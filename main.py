import re
import gc
import time
import random
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import os
import pickle
import warnings
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    StoppingCriteria, 
    StoppingCriteriaList
)

# Suppress warnings
warnings.filterwarnings("ignore", message=".*HybridCache.*")
torch._dynamo.config.suppress_errors = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
#  Utility Functions                                                          #
# --------------------------------------------------------------------------- #

def init_model(args):
    """Initialize the Huggingface model."""
    return HuggingfaceModel(args.model_name, max_new_tokens=args.model_max_new_tokens)


def save(obj, file, save_dir):
    """Save object to file in specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Saved] {file} → {path}")


def get_parser():
    """Return a minimal argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--model_max_new_tokens", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser


# --------------------------------------------------------------------------- #
#  Huggingface Model Implementation                                           #
# --------------------------------------------------------------------------- #

class StopWordsCriteria(StoppingCriteria):
    """
    Stop generation once any of the stop strings appears **after** the prompt.
    """
    def __init__(self, stop_strings, tokenizer, input_ids):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.input_len = input_ids.shape[1]

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.batch_decode(
            input_ids[:, self.input_len:], skip_special_tokens=True
        )
        return any(any(s in d for s in self.stop_strings) for d in decoded)


class HuggingfaceModel:
    """Simplified wrapper for generation + hidden-state capture."""
    def __init__(self, model_name: str, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.token_limit = 8192

        model_id = self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            device_map="auto",
            token_type_ids=None,
            clean_up_tokenization_spaces=False,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def batch_predict(
        self,
        prompts,
        temperature: float,
        *,
        return_latent: bool = False,
        batch_size: int = 10,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = []

        for b0 in range(0, len(prompts), batch_size):
            batch = prompts[b0 : b0 + batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.token_limit - self.max_new_tokens,
                return_tensors="pt",
            ).to(device)

            # Ensure input_ids is a tensor
            assert isinstance(enc["input_ids"], torch.Tensor), "input_ids must be a tensor"

            criteria = StoppingCriteriaList(
                [StopWordsCriteria(["Q:", "Context:", "END"], self.tokenizer, enc["input_ids"])]
            )

            with torch.no_grad():
                gen = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    stopping_criteria=criteria,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.85,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            hid_steps = gen.hidden_states  # tuple(len = #generated tokens)

            for idx, prompt in enumerate(batch):
                full = self.tokenizer.decode(gen.sequences[idx], skip_special_tokens=True)
                tail = full[len(prompt):]
                pos = tail.find("END")
                slice_txt = tail[:pos].strip() if pos != -1 else tail.strip()

                tok_ids = self.tokenizer(
                    full[: len(prompt) + len(slice_txt)], return_tensors="pt"
                )["input_ids"]
                tok_stop = tok_ids.shape[1]
                n_prompt = (enc["input_ids"][idx] != self.tokenizer.pad_token_id).sum().item()
                n_gen = max(tok_stop - n_prompt, 1)
                if n_gen > len(hid_steps):
                    n_gen = len(hid_steps)  # Prevent index out of range

                # Extract embeddings with bounds checking
                if n_gen > 0 and len(hid_steps) > 0:
                    last_emb = hid_steps[n_gen - 1][-1][idx, -1, :].cpu()
                else:
                    last_emb = None

                if n_gen >= 2 and len(hid_steps) >= 2:
                    sec_emb = torch.stack([l[idx, -1, :] for l in hid_steps[n_gen - 2]]).cpu()
                else:
                    sec_emb = None

                if len(hid_steps) > 0:
                    pre_emb = torch.stack([l[idx, -1, :] for l in hid_steps[0]]).cpu()
                else:
                    pre_emb = None

                if n_gen >= 2 and len(hid_steps) >= 2:
                    # **Fixed Line**: Correctly access sequence length from tensor shape
                    seq_len = hid_steps[n_gen - 2][-1].shape[1]  # Get sequence length from last layer
                    if seq_len >= 2:
                        slt_emb = torch.stack([l[idx, -2, :] for l in hid_steps[n_gen - 2]]).cpu()
                    else:
                        slt_emb = None
                else:
                    slt_emb = None

                trans = self.model.compute_transition_scores(
                    gen.sequences, gen.scores, normalize_logits=True
                )
                log_liks = [s.item() for s in trans[idx][:n_gen]]

                if return_latent:
                    lat = {
                        "last_emb": last_emb,
                        "sec_emb": sec_emb,
                        "pre_emb": pre_emb,
                        "slt_emb": slt_emb,
                    }
                else:
                    lat = None
                results.append((slice_txt, log_liks, lat))

        return results


# --------------------------------------------------------------------------- #
#  Semantic Entropy Implementation                                            #
# --------------------------------------------------------------------------- #

random.seed(10)

class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        logits = outputs.logits
        largest_index = torch.argmax(F.softmax(logits, dim=1))
        prediction = largest_index.cpu().item()
        return prediction


def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""
    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
        return semantically_equivalent

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1
    assert -1 not in semantic_set_ids
    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum'):
    """Sum probabilities with the same semantic id."""
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []
    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum':
            logsumexp_value = np.log(np.sum(np.exp(id_log_likelihoods))) - 5.0
        elif agg == 'sum_normalized':
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        elif agg == 'mean':
            logsumexp_value = np.log(np.mean(np.exp(id_log_likelihoods)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)
    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy."""
    entropy = -np.sum(log_probs) / len(log_probs)
    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from cluster assignments."""
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy


# --------------------------------------------------------------------------- #
#  Main Program                                                               #
# --------------------------------------------------------------------------- #
def main(args):
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # -------- 1. LOAD & SPLIT DATASET ---------------------------------------
    ds = load_dataset("openbmb/UltraFeedback")["train"].train_test_split(
        test_size=0.2, seed=42
    )
    train_raw, test_raw = ds["train"], ds["test"]

    # -------- 2. REFORMAT ----------------------------------------------------
    def reformat(ex, j):
        try:
            comp = ex["completions"][j]
            resp = comp.get("response", "No response found")
            ann  = comp.get("annotations", {}).get("helpfulness", {})
            md5  = lambda s: str(int(hashlib.md5(s.encode()).hexdigest(), 16))
            return {
                "question":   ex["instruction"],
                "response":   resp,
                "evaluation": f"Rating: {ann.get('Rating', '?')}\n"
                              f"Rationale: {ann.get('Rationale', '?')}",
                "id":         md5(ex["instruction"]),
            }
        except Exception:
            return None

    def unpack(raw):
        return [
            x for d in raw for j in range(4)
            if (x := reformat(d, j)) is not None
        ]

    train_ds, test_ds = unpack(train_raw), unpack(test_raw)

    # -------- Shuffle datasets -----------------------------------------------
    random.shuffle(train_ds)
    random.shuffle(test_ds)

    # -------- 3. 0-SHOT PROMPT -----------------------------------------------
    def fewshot(dataset, k=0, limit=1000):
        prompt = (
            "You are a helpful assistant evaluating how helpful a response is to a question.\n"
            "For each example, you will see:\n"
            "Question: <the question>\n"
            "Response: <the response>\n"
            "Rate the response's helpfulness.\n"
            "Your answer MUST be exactly two lines in this format:\n"
            "Rating: <1-5>\n"
            "Rationale: <very brief explanation for rating, only a few words>\n"
            "Do not repeat the question or response. Do not add extra text. End after the rationale.\n\n"
        )
        return prompt

    zero_shot_prompt = fewshot(train_ds, k=0)

    # -------- 4. INITIALISE MODELS -------------------------------------------
    model            = init_model(args)
    entailment_model = EntailmentDeberta()

    # -------- 5. MAIN LOOP ---------------------------------------------------
    for split_name, data in [("train", train_ds), ("validation", test_ds)]:
        print(f"\n========== Generating evaluations for {split_name} split ==========")
        print(f"Dataset size: {len(data)}")
        generations, collected, idx = {}, 0, 0

        while collected < args.num_samples and idx < len(data):
            ex = data[idx]
            idx += 1

            q, resp = ex["question"], ex["response"]
            lp = f"{zero_shot_prompt}Question: {q}\nResponse: {resp}\nEvaluation:"

            print(f"\nProcessing example {idx}/{len(data)}")

            # ----- Greedy ----------------------------------------------------
            try:
                g_ans, _, latents = model.batch_predict(
                    [lp], temperature=0.1, return_latent=True
                )[0]
                print(f"Greedy answer: {repr(g_ans)}")
                greedy = g_ans  # No filtering
            except Exception as e:
                print(f"Exception in greedy generation: {e}")
                continue

            # ----- Sampling --------------------------------------------------
            responses, log_liks, embeds = [], [], []
            attempts = 0
            print(f"Starting sampling...")
            while len(responses) < 10 and attempts < 40:
                attempts += 1
                try:
                    ans, tls, lat = model.batch_predict(
                        [lp], temperature=args.temperature, return_latent=True
                    )[0]
                    responses.append(ans)  # No filtering
                    log_liks.append(tls)
                    embeds.append(lat["slt_emb"])
                    print(f"Attempt {attempts}: Got sample ({len(responses)}/10)")
                except Exception as e:
                    print(f"Attempt {attempts}: Exception in sampling: {e}")
                    continue

            print(f"Sampling complete: {len(responses)} responses from {attempts} attempts")
            if len(responses) < 3:
                print("Not enough responses, skipping...")
                continue

            # ----- Entropy ---------------------------------------------------
            try:
                print("Computing semantic entropy...")
                sem_ids = get_semantic_ids(
                    responses, entailment_model,
                    strict_entailment=False, example=ex
                )
                entropy = cluster_assignment_entropy(sem_ids)
                print(f"Semantic IDs: {sem_ids}")
                print(f"Entropy: {entropy:.4f}")
            except Exception as e:
                print(f"Exception in entropy calculation: {e}")
                continue

            # ----- Print to terminal -----------------------------------------
            print("\n---------------- PROMPT ----------------")
            print(lp)
            print("------------ MODEL ANSWERS -------------")
            for i, r in enumerate(responses, 1):
                print(f"{i}. {r}")
            print(f"Entropy: {entropy:.4f}")
            print("----------------------------------------\n")

            # ----- Store -----------------------------------------------------
            generations[ex["id"]] = {
                "context": q,
                "question": "Evaluate the following model response: " + resp,
                "responses": list(zip(responses, log_liks, embeds)),
                "most_likely_answer": {
                    "response": greedy,
                    "last_embedding": latents["last_emb"],
                    "sec_last_embedding": latents["sec_emb"],
                    "prompt_last_embedding": latents["pre_emb"],
                    "slt_embedding": latents["slt_emb"],
                },
                "entropy": entropy,
                "reference": resp,
            }
            collected += 1

            # --- Progress ----------------------------------------------------
            print(f"Progress: {collected}/{args.num_samples} examples processed")

        # -------- 6. BINARIZE ENTROPY ---------------------------------------
        if generations:
            entropies = [g["entropy"] for g in generations.values()]

            def find_optimal_threshold(entropies):
                best_gamma, best_mse = None, float("inf")
                for gamma in np.linspace(min(entropies), max(entropies), 100):
                    low = [e for e in entropies if e < gamma]
                    high = [e for e in entropies if e >= gamma]
                    if not low or not high:
                        continue
                    mse = sum((e - np.mean(low))**2 for e in low) + sum((e - np.mean(high))**2 for e in high)
                    if mse < best_mse:
                        best_mse, best_gamma = mse, gamma
                return best_gamma

            gamma_star = find_optimal_threshold(entropies)
            for gid, gen in generations.items():
                gen["binarized_entropy"] = 1 if gen["entropy"] > gamma_star else 0

        save(
            generations, f"{split_name}_generations.pkl",
            save_dir="/workspace/sep-probe-judge"
        )

    print("Run complete.")
    del model
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  Entry-point                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--num_few_shot", type=int, default=0,
                        help="number of few-shot examples in the prompt (0 for zero-shot)")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)

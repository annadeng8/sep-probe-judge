#!/usr/bin/env python3
"""
Generate evaluations with LLM, cache activations, and compute entropy with few-shot prompting.

Revision notes
--------------
• keep the entire two-line evaluation in `responses`
• print the prompt and *every* sampled answer to stdout
• **strict `clean_evaluation()`** – only accept answers that
    ▸ have “Rating:” 1-5
    ▸ have a non-empty “Rationale: …”
• malformed answers are silently skipped
• **NEW**: print the entropy value for every kept batch
"""
import re
import gc
import random
import hashlib
import numpy as np
import torch
from datasets import load_dataset
from uncertainty.utils import utils
from uncertainty.semantic_entropy import (
    get_semantic_ids,
    cluster_assignment_entropy,
    EntailmentDeberta,
)

# --------------------------------------------------------------------------- #
#  Strict evaluation-line regexes                                             #
# --------------------------------------------------------------------------- #
RATING_RE    = re.compile(r"^Rating:\s*[1-5]\s*$",  re.I)
RATIONALE_RE = re.compile(r"^Rationale:\s*\S.*$",   re.I)


def clean_evaluation(text: str) -> str | None:
    """
    Return the cleaned two-line evaluation if it exactly matches
        Rating: <1-5>
        Rationale: <non-empty …>
    else return None.
    """
    if "Rating:" in text:
        text = text[text.index("Rating:") :]
    if "END" in text:
        text = text[: text.index("END")]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    rating, rationale = lines[:2]
    if not RATING_RE.fullmatch(rating):
        return None
    if not RATIONALE_RE.fullmatch(rationale):
        return None
    return f"{rating}\n{rationale}"


# --------------------------------------------------------------------------- #
#  Main program                                                               #
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
            comp  = ex["completions"][j]
            resp  = comp.get("response", "No response found")
            ann   = comp.get("annotations", {}).get("helpfulness", {})
            md5   = lambda s: str(int(hashlib.md5(s.encode()).hexdigest(), 16))
            return {
                "question":   ex["instruction"],
                "response":   resp,
                "evaluation": f"Rating: {ann.get('Rating','?')}\n"
                              f"Rationale: {ann.get('Rationale','?')}",
                "id":         md5(ex["instruction"]),
            }
        except Exception:
            return None

    unpack = lambda raw: [
        x for d in raw for j in range(4)
        if (x := reformat(d, j)) is not None
    ]
    train_ds, test_ds = unpack(train_raw), unpack(test_raw)

    # -------- 3. FEW-SHOT PROMPT --------------------------------------------
    def fewshot(dataset, k=3, limit=1000):
        prompt = (
            "You are an evaluator of text quality. Your task is to evaluate the helpfulness of responses.\n\n"
            "CRITICAL FORMAT RULES:\n"
            "1. Your response MUST be exactly two lines:\n"
            "   Rating: <number 1-5>\n"
            "   Rationale: <one sentence explanation>\n"
            "2. Do not include any other text, labels, or information\n"
            "3. Keep rationales brief and focused\n"
            "4. Do not repeat the question or instruction in your rationale\n"
            "5. Do not include 'Evaluation:' or any other prefix\n"
            "6. Do not include any text after the rationale\n"
            "7. Your response MUST end after the rationale\n\n"
            "Examples:\n\n"
        )
        for ex in random.sample(dataset, k):
            snippet = (
                f"Question: {ex['question']}\n"
                f"Response: {ex['response']}\n"
                f"Evaluation: {ex['evaluation']} END\n\n"
            )
            prompt += snippet[:limit]
        prompt += (
            "Now evaluate the following response. Remember the exact two-line format. "
            "Write END after your response.\n\n"
        )
        return prompt

    few_shot_prompt = fewshot(train_ds, k=args.num_few_shot)

    # -------- 4. INITIALISE MODELS ------------------------------------------
    model            = utils.init_model(args)
    entailment_model = EntailmentDeberta()

    # -------- 5. MAIN LOOP ---------------------------------------------------
    for split_name, data in [("train", train_ds), ("validation", test_ds)]:
        print(f"\n========== Generating evaluations for {split_name} split ==========")
        generations, collected, idx = {}, 0, 0

        while collected < args.num_samples and idx < len(data):
            ex = data[idx]
            idx += 1

            q, resp = ex["question"], ex["response"]
            lp = f"{few_shot_prompt}Question: {q}\nResponse: {resp}\nEvaluation:"

            # ----- Greedy ----------------------------------------------------
            try:
                g_ans, _, (g_last, g_sec, g_pre) = model.batch_predict(
                    [lp], temperature=0.1, return_latent=True
                )[0]
                greedy = clean_evaluation(g_ans)
                if greedy is None:
                    continue
            except Exception:
                continue

            # ----- Sampling --------------------------------------------------
            responses, log_liks, embeds = [], [], []
            attempts = 0
            while len(responses) < 10 and attempts < 40:
                attempts += 1
                ans, tls, (e_last, *_rest) = model.batch_predict(
                    [lp], temperature=args.temperature, return_latent=True
                )[0]
                clean = clean_evaluation(ans)
                if clean is None:
                    continue
                responses.append(clean)
                log_liks.append(tls)
                embeds.append(e_last)

            if len(responses) < 3:
                continue

            # ----- Entropy ---------------------------------------------------
            try:
                sem_ids = get_semantic_ids(responses, entailment_model, ex)
                entropy = cluster_assignment_entropy(sem_ids)
            except Exception:
                continue

            # ----- Print to terminal ----------------------------------------
            print("\n---------------- PROMPT ----------------")
            print(lp)
            print("------------ MODEL ANSWERS -------------")
            for i, r in enumerate(responses, 1):
                print(f"{i}. {r}")
            print(f"Entropy: {entropy:.4f}")            # <--- New line
            print("----------------------------------------\n")

            # ----- Store -----------------------------------------------------
            generations[ex["id"]] = {
                "context": q,
                "question": "Evaluate the following model response: " + resp,
                "responses": list(zip(responses, log_liks, embeds)),
                "most_likely_answer": {
                    "response": greedy,
                    "last_embedding": g_last,
                    "sec_last_embedding": g_sec,
                    "prompt_last_embedding": g_pre,
                },
                "entropy": entropy,
                "reference": resp,
            }
            collected += 1

        utils.save(
            generations, f"{split_name}_generations.pkl",
            save_dir="/workspace/sep-temp"
        )

    print("Run complete.")
    del model
    torch.cuda.empty_cache()
    time.sleep(600000)


# --------------------------------------------------------------------------- #
#  Entry-point                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=3,
                        help="number of few-shot examples in the prompt")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)

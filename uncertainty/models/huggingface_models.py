"""Minimal Huggingface model implementation (stop at END, full latents, clipped n_gen without error logs)."""

import warnings
warnings.filterwarnings("ignore", message=".*HybridCache.*")

import torch
import torch._dynamo
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

torch._dynamo.config.suppress_errors = True


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

        model_id = "google/gemma-2-2b"  # hard-wired judge model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            device_map="auto",
            token_type_ids=None,
            clean_up_tokenization_spaces=False,
        )
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
                # clip without logging
                if n_gen > len(hid_steps):
                    n_gen = len(hid_steps)

                last_emb = hid_steps[n_gen - 1][-1][idx, -1, :].cpu()
                sec_emb = (
                    torch.stack([l[idx, -1, :] for l in hid_steps[n_gen - 2]]).cpu()
                    if n_gen >= 2
                    else None
                )
                pre_emb = torch.stack([l[idx, -1, :] for l in hid_steps[0]]).cpu()

                trans = self.model.compute_transition_scores(
                    gen.sequences, gen.scores, normalize_logits=True
                )
                log_liks = [s.item() for s in trans[idx][:n_gen]]

                lat = (last_emb, sec_emb, pre_emb) if return_latent else None
                results.append((slice_txt, log_liks, lat))

        return results
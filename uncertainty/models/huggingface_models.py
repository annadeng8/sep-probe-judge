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

        model_id = "google/gemma-2-9b-it"  # hard-wired judge model
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
        stop_sequences: list = None,
        min_tokens: int = 20,
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

            # Create stopping criteria - only if stop_sequences is not empty
            criteria_list = []
            if stop_sequences:
                criteria_list.append(
                    StopWordsCriteria(stop_sequences, self.tokenizer, enc["input_ids"])
                )
            criteria = StoppingCriteriaList(criteria_list) if criteria_list else None

            with torch.no_grad():
                gen_kwargs = {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "max_new_tokens": max(self.max_new_tokens, min_tokens),
                    "min_new_tokens": min_tokens,  # Force minimum generation
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "output_hidden_states": True,
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 50,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                
                # Only add stopping criteria if we have any
                if criteria:
                    gen_kwargs["stopping_criteria"] = criteria
                
                gen = self.model.generate(**gen_kwargs)

            hid_steps = gen.hidden_states  # tuple(len = #generated tokens)

            for idx, prompt in enumerate(batch):
                full = self.tokenizer.decode(gen.sequences[idx], skip_special_tokens=True)
                tail = full[len(prompt):]
                
                # Check for END token or use the full generation
                pos = tail.find("END")
                slice_txt = tail[:pos].strip() if pos != -1 else tail.strip()
                
                # Additional safety: if still blank, take more of the generation
                if not slice_txt and len(tail) > 0:
                    slice_txt = tail.strip()

                tok_ids = self.tokenizer(
                    full[: len(prompt) + len(slice_txt)], return_tensors="pt"
                )["input_ids"]
                tok_stop = tok_ids.shape[1]
                n_prompt = (enc["input_ids"][idx] != self.tokenizer.pad_token_id).sum().item()
                n_gen = max(tok_stop - n_prompt, 1)
                # clip without logging
                if n_gen > len(hid_steps):
                    n_gen = len(hid_steps)

                # FIXED: Extract hidden states from the correct positions
                # TBG (Token Before Generation) - last token of the prompt
                tbg_embedding = hid_steps[0][-1][idx, -1, :].detach().cpu()
                
                # SLT (Second Last Token) - second-to-last generated token
                if n_gen >= 2:
                    slt_embedding = hid_steps[n_gen - 2][-1][idx, -1, :].detach().cpu()
                else:
                    # If only one token generated, use the first generated token
                    slt_embedding = hid_steps[0][-1][idx, -1, :].detach().cpu()
                
                # Last generated token (for compatibility, though paper doesn't use this)
                last_embedding = hid_steps[n_gen - 1][-1][idx, -1, :].detach().cpu()

                trans = self.model.compute_transition_scores(
                    gen.sequences, gen.scores, normalize_logits=True
                )
                log_liks = [s.item() for s in trans[idx][:n_gen]]

                # Return in order: (last_embedding, slt_embedding, tbg_embedding)
                lat = (last_embedding, slt_embedding, tbg_embedding) if return_latent else None
                results.append((slice_txt, log_liks, lat))

        return results

    def get_top_tokens(self, prompts, k=5):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = []
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :]  # logits for the next token
                probs = torch.softmax(logits, dim=-1)
                topk = torch.topk(probs, k)
                top_tokens = [self.tokenizer.decode([idx]) for idx in topk.indices[0]]
                top_probs = topk.values[0].cpu().to(torch.float32).numpy()
                results.append(list(zip(top_tokens, top_probs)))
        return results

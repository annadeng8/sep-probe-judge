"""Minimal Huggingface model implementation."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer, input_ids):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.input_len = input_ids.shape[1]

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only the newly generated tokens
        decoded = self.tokenizer.batch_decode(input_ids[:, self.input_len:], skip_special_tokens=True)
        for text in decoded:
            if any(stop_string in text for stop_string in self.stop_strings):
                return True  # Stop generation
        return False

class HuggingfaceModel:
    """Simplified Huggingface model for generation and activation caching."""
    def __init__(self, model_name, max_new_tokens):
        self.max_new_tokens = max_new_tokens * 2  # Double the max tokens to ensure enough space
        model_id = "google/gemma-2b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, device_map='auto', token_type_ids=None, clean_up_tokenization_spaces=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
        self.model_name = model_name
        self.token_limit = 8192
    def batch_predict(self, prompts, temperature, return_latent=False, batch_size=10, min_new_tokens=1):
        """Generate answers for a batch of prompts and return text, log-likelihoods, and embeddings if requested."""
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]

            # Tokenize batch prompts
            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=self.token_limit - self.max_new_tokens,
                return_tensors="pt"
            ).to(device)

            allowed_keys = {"input_ids", "attention_mask"}
            safe_encoded = {k: v for k, v in encoded.items() if k in allowed_keys}

            # Now call generate without stopping_criteria
            with torch.no_grad():
                outputs = self.model.generate(
                    **safe_encoded,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # ---- FIXED SECTION ----
            # last layer → shape (batch, seq_len, hidden_size)
            
            hidden_states      = outputs.hidden_states[0][-1][0,0]
            last_token_embedding = hidden_states.cpu()
            
            for i, prompt in enumerate(batch_prompts):
                full_answer   = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
                sliced_answer = full_answer[len(prompt):].strip()

                # Prompt / generation lengths
                token_stop_index = self.tokenizer(full_answer, return_tensors="pt")["input_ids"].shape[1]
                n_input_token    = encoded["input_ids"][i].ne(self.tokenizer.pad_token_id).sum().item()
                n_generated      = token_stop_index - n_input_token or 1

                # -----------------------

                # Log-likelihoods for generated tokens
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                log_likelihoods = [score.item() for score in transition_scores[i][:n_generated]]

                if return_latent:
                    results.append((sliced_answer, log_likelihoods, (last_token_embedding, None, None)))
                else:
                    results.append((sliced_answer, log_likelihoods, None))

        return results

"""Minimal Huggingface model implementation."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
class HuggingfaceModel:
    """Simplified Huggingface model for generation and activation caching."""
    def __init__(self, model_name, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        model_id = "google/gemma-2b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, device_map='auto', token_type_ids=None, clean_up_tokenization_spaces=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
        self.model_name = model_name
        self.token_limit = 8192
    def predict(self, input_data, temperature, return_latent=False):
        """Generate answer and return text, log likelihoods, and embeddings if requested."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = self.tokenizer.encode(input_data)
        max_input_tokens = self.token_limit - self.max_new_tokens
        if len(input_ids) > max_input_tokens:
            input_ids = input_ids[:max_input_tokens]
            input_data = self.tokenizer.decode(input_ids)
        inputs = {
            "input_ids": torch.tensor([input_ids], device=device),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long, device=device).unsqueeze(0)
        }
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        sliced_answer = full_answer[len(input_data):].strip()
        # Compute token stop index
        token_stop_index = self.tokenizer(full_answer, return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token or 1
        # Extract last token embedding
        hidden = outputs.hidden_states
        print("Shape",hidden.shape)
        last_input = hidden[min(n_generated - 1, len(hidden) - 1)][-1]
        last_token_embedding = last_input[:, -1, :].cpu()
        # Compute log likelihoods
        transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = [score.item() for score in transition_scores[0]][:n_generated]
        if return_latent:
            return sliced_answer, log_likelihoods, (last_token_embedding, None, None)
        return sliced_answer, log_likelihoods, None
    def batch_predict(self, prompts, temperature, return_latent=False, batch_size=10):
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

            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
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
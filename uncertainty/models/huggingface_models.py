"""Minimal Huggingface model implementation."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class HuggingfaceModel:
    """Simplified Huggingface model for generation and activation caching."""
    def __init__(self, model_name, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        model_id = "google/gemma-2b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
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
        last_input = hidden[min(n_generated - 1, len(hidden) - 1)][-1]
        last_token_embedding = last_input[:, -1, :].cpu()

        # Compute log likelihoods
        transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = [score.item() for score in transition_scores[0]][:n_generated]

        if return_latent:
            return sliced_answer, log_likelihoods, (last_token_embedding, None, None)
        return sliced_answer, log_likelihoods, None
    
    def predict_batch(self, prompts, temperature=1.0, return_latent=False):
        """Generate outputs for a batch of prompts."""
        device = self.model.device
        max_input_tokens = self.token_limit - self.max_new_tokens

        # Tokenize with padding/truncation
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens
        ).to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        sequences = outputs.sequences
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        results = []
        for i, prompt in enumerate(prompts):
            full = decoded[i]
            generated = full[len(prompt):].strip()

            # Skip latent/log likelihood if not needed
            if not return_latent:
                results.append((generated, None, None))
                continue

            # Compute log probs and embedding (optional)
            transition_scores = self.model.compute_transition_scores([sequences[i]], [s[i:i+1] for s in outputs.scores], normalize_logits=True)
            log_likelihoods = [score.item() for score in transition_scores[0]]
            last_input = outputs.hidden_states[-1][i][-1].unsqueeze(0).cpu()
            results.append((generated, log_likelihoods, (last_input, None, None)))

        return results
        
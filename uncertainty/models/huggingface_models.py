"""Minimal Huggingface model implementation."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch._dynamo
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F


torch._dynamo.config.suppress_errors = True

from transformers import LogitsProcessor
from transformers import LogitsProcessorList


class PenalizeTokenProcessor(LogitsProcessor):
    def __init__(self, token_id: int, penalty: float):
        self.token_id = token_id
        self.penalty = penalty  # positive value → lower probability

    def __call__(self, input_ids, scores):
        # Subtract penalty from the target token
        scores[:, self.token_id] -= self.penalty
        return scores

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
                print("STOPPING CRITERIA EVOKED")
                return True  # Stop generation
        return False

class HuggingfaceModel:
    """Simplified Huggingface model for generation and activation caching."""
    def __init__(self, model_name, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        model_id = "google/gemma-2-2b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, device_map='auto', token_type_ids=None, clean_up_tokenization_spaces=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
        self.model_name = model_name
        self.token_limit = 8192
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

                # Define your stop strings
            # stop_strings = ["\n\n", "Question:", "Q:", "Context:", "Now, provide", "Use exactly", "Evaluation", "Thank you"]
            stop_strings = ["Q:", "Context:"]

            # Build stopping criteria
            stop_criteria = StoppingCriteriaList([
                StopWordsCriteria(stop_strings, self.tokenizer, encoded["input_ids"])
            ])

            allowed_keys = {"input_ids", "attention_mask"}
            safe_encoded = {k: v for k, v in encoded.items() if k in allowed_keys}


            target_token = self.tokenizer.encode("Question:", add_special_tokens=False)[0]
            penalty = 10.0  # large → lower probability a lot

            """
            logits_processor = LogitsProcessorList([
                PenalizeTokenProcessor(token_id=target_token, penalty=penalty)
            ])
            """


            # Now call generate with stopping_criteria
            with torch.no_grad():
                outputs = self.model.generate(
                    **safe_encoded,
                    max_new_tokens=self.max_new_tokens,
                    #logits_processor=logits_processor,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stop_criteria,
                    top_p = 0.85,
                    top_k = 50,
                    )
            
            """
            for step_idx, step_logits in enumerate(outputs.scores):  # one per generated token
                probs = F.softmax(step_logits, dim=-1)  # shape: (batch_size, vocab_size)

                # Example: log top 5 tokens for sample 0
                topk = torch.topk(probs[0], k=5)
                print(f"\n[Step {step_idx}] Top-5 token probabilities:")
                for prob, token_id in zip(topk.values.tolist(), topk.indices.tolist()):
                    decoded = self.tokenizer.decode([token_id])
                    print(f"  {decoded!r}: {prob:.4f}")
            """
            
            """
            eos_token_id = self.tokenizer.eos_token_id
            for i, sequence in enumerate(outputs.sequences):
                if eos_token_id in sequence:
                    print(f"[EOS Detected] Sample {i} includes eos_token_id ({eos_token_id})")
                    # Optional: where in the sequence
                    eos_positions = (sequence == eos_token_id).nonzero(as_tuple=True)[0].tolist()
                    print(f" → EOS token found at indices: {eos_positions}")

                else:
                    print(f"[EOS Not Found] Sample {i} did not include eos_token_id")
            """

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
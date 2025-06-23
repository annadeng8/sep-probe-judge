import numpy as np
import torch
from datasets import load_dataset
from uncertainty.utils import utils
import hashlib
import random
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Entailment model for semantic clustering
class EntailmentDeberta:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli", 
            use_safetensors=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return prediction

# Function to cluster evaluations semantically
def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2)
        implication_2 = model.check_implication(text2, text1)
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
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1
    return semantic_set_ids

# Function to compute log probabilities per semantic cluster
def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum'):
    unique_ids = sorted(list(set(semantic_ids)))
    log_likelihood_per_semantic_id = []
    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum':
            logsumexp_value = np.log(np.sum(np.exp(id_log_likelihoods)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)
    return log_likelihood_per_semantic_id

def main(args):
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    dataset = load_dataset("openbmb/UltraFeedback")["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def reformat(example, j):
        try:
            completion = example['completions'][j]
            response = completion.get('response', 'No response found')
            annotations = completion.get('annotations', {})
            instruction_following = annotations.get('instruction_following', {})
            rating = instruction_following.get('Rating', 'Unknown rating')
            rationale = instruction_following.get('Rationale', 'No rationale provided')
            md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
            return {
                'question': example['instruction'],
                'response': response,
                'evaluation': f"Rating: {rating}\nRationale: {rationale}",
                'id': md5hash(str(example['instruction']))
            }
        except:
            return None

    train_dataset = [x for d in train_dataset for j in range(4) if (x := reformat(d, j)) is not None]
    test_dataset = [x for d in test_dataset for j in range(4) if (x := reformat(d, j)) is not None]

    def construct_fewshot_prompt(dataset, num_examples=3):
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
        sampled_indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        for idx in sampled_indices:
            example = dataset[idx]
            prompt += (
                f"Instruction: {example['question']}\n"
                f"Response: {example['response']}\n"
                f"{example['evaluation']}\n"
                f"END\n\n"
            )
        prompt += (
            "Now evaluate the following response. Remember:\n"
            "- Use EXACTLY two lines\n"
            "- First line: Rating: <number 1-5>\n"
            "- Second line: Rationale: <one sentence>\n"
            "- Do not include any other text\n"
            "- Do not include 'Evaluation:' or any prefix\n"
            "- Your response MUST end after the rationale\n"
            "- Write END after your response\n\n"
        )
        return prompt

    def clean_evaluation(text):
        if "Rating:" in text:
            text = text[text.index("Rating:"):]
        if "END" in text:
            text = text[:text.index("END")]
        lines = text.strip().split('\n')
        if len(lines) >= 2:
            return '\n'.join(lines[:2])
        return text

    model = utils.init_model(args)
    entailment_model = EntailmentDeberta()  # Initialize entailment model

    for dataset_split, dataset in [('train', train_dataset), ('validation', test_dataset)]:
        print(f"Generating evaluations for {dataset_split} split")
        generations = {}
        indices = range(min(args.num_samples, len(dataset)))
        for index in indices:
            example = dataset[index]
            question = example["question"]
            test_answer = example["response"]
            generations[example['id']] = {
                'context': question,
                'question': "Evaluate the following model response: " + test_answer,
                'responses': []
            }
            current_input = f"Instruction: {question}\nResponse: {test_answer}\n"
            local_prompt = construct_fewshot_prompt(train_dataset, num_examples=args.num_few_shot) + current_input
            full_evaluations = []
            num_generations = 10
            prompts = [local_prompt] * num_generations
            results = model.batch_predict(
                prompts,
                temperature=0.3,
                return_latent=True
            )
            for predicted_evaluation, token_log_likelihoods, (embedding, _, _) in results:
                cleaned_evaluation = clean_evaluation(predicted_evaluation)
                embedding = embedding.cpu() if embedding is not None else None
                full_evaluations.append((cleaned_evaluation, token_log_likelihoods, embedding))
                print(f"Evaluation: {cleaned_evaluation.replace(chr(10), ' ')}")

            # Compute semantic entropy using rating + rationale
            evaluation_texts = [eval_text for eval_text, _, _ in full_evaluations]
            log_liks = [np.sum(token_log_likelihoods) for _, token_log_likelihoods, _ in full_evaluations]
            semantic_ids = get_semantic_ids(evaluation_texts, model=entailment_model)
            log_p_clusters = logsumexp_by_id(semantic_ids, log_liks, agg='sum')
            if log_p_clusters:
                entropy = -np.mean(log_p_clusters)
            else:
                entropy = 0
            print(f"Entropy: {entropy:.4f}")

            generations[example['id']]['responses'] = full_evaluations
            generations[example['id']]['entropy'] = entropy

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            del predicted_evaluation, token_log_likelihoods, embedding, results

        utils.save(generations, f'{dataset_split}_generations.pkl', save_dir="/workspace/sep-temp")

    print("Run complete.")
    del model

if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=3, help="Number of few-shot examples")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)

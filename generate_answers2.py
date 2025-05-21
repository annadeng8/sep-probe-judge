"""Generate evaluations with LLM, cache activations, and compute entropy with few-shot prompting."""
import numpy as np
import torch
from datasets import load_dataset
from uncertainty.utils import utils
import hashlib
import random
import gc
def main(args):
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    # Load and split dataset
    dataset = load_dataset("openbmb/UltraFeedback")["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    def reformat(example, j):
        try:
            completion = example['completions'][j]
            response = completion.get('response', 'No response found')
            annotations = completion.get('annotations', {})
            inst = annotations.get('instruction_following', {})
            helpf = annotations.get('helpfulness', {})
            md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
            return {
                'question': example['instruction'],
                'response': response,
                'instruction_following': {
                    'Rating': inst.get('Rating', 'Unknown rating'),
                    'Rationale': inst.get('Rationale', 'No rationale provided')
                },
                'helpfulness': {
                    'Rating': helpf.get('Rating', 'Unknown rating'),
                    'Rationale': helpf.get('Rationale', 'No rationale provided')
                },
                'id': md5hash(str(example['instruction']) + response)
            }
        except:
            return None
    train_dataset = [x for d in train_dataset for j in range(4) if (x := reformat(d, j)) is not None]
    test_dataset = [x for d in test_dataset for j in range(4) if (x := reformat(d, j)) is not None]
    def construct_prompt(dataset, field, num_examples=3):
        prompt = f"You are an evaluator of text quality. Below are examples to guide your evaluation of {field.replace('_', ' ')}.\n\n"
        for ex in random.sample(dataset, min(num_examples, len(dataset))):
            prompt += f"Instruction: {ex['question']}\nQuestion: {ex['question']}\nAnswer: {ex['response']}\nEvaluation: Rating: {ex[field]['Rating']}\nRationale: {ex[field]['Rationale']}\n\n"
        prompt += "Now, provide your evaluation for the following. Use the same format exactly:\nRating: <1-5> Rationale: <brief explanation>\nDO NOT include anything else.\n"
        return prompt
    model = utils.init_model(args)
    for dataset_split, dataset in [('train', train_dataset), ('validation', test_dataset)]:
        print(f"Generating evaluations for {dataset_split} split")
        generations = {}
        indices = range(min(args.num_samples, len(dataset)))
        for index in indices:
            example = dataset[index]
            question = example["question"]
            answer = example["response"]
            generations[example['id']] = {
                'context': question,
                'question': answer
            }
            for field in ['instruction_following', 'helpfulness']:
                prompt = construct_prompt(train_dataset, field, num_examples=args.num_few_shot)
                full_input = prompt + f"Instruction: {question}\nQuestion: {question}\nAnswer: {answer}\nEvaluation:"
                prompts = [full_input] * args.num_generations
                results = model.batch_predict(prompts, temperature=args.temperature, return_latent=True)
                evaluations = []
                ratings = []
                for pred, logp, (embedding, _, _) in results:
                    embedding = embedding.cpu() if embedding is not None else None
                    evaluations.append((pred, logp, embedding))
                    try:
                        rating_str = pred.split("Rating: ")[1].split("\n")[0].strip()
                        rating = int(rating_str)
                    except (IndexError, ValueError):
                        rating = None
                    ratings.append(rating)
                    print(f"{field} Evaluation: {pred.replace(chr(10), ' ')}")
                valid_ratings = [r for r in ratings if r is not None]
                if valid_ratings:
                    counts = np.unique(valid_ratings, return_counts=True)[1]
                    probs = counts / len(valid_ratings)
                    entropy = -np.sum(probs * np.log(probs))
                else:
                    entropy = 0
                generations[example['id']][f"{field}_responses"] = evaluations
                generations[example['id']][f"{field}_entropy"] = entropy
                print(f"{field} Entropy: {entropy:.4f}")
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
        utils.save(generations, f"{dataset_split}_generations.pkl", save_dir="/workspace/sep-temp")
    print("Run complete.")
    del model
if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=3)
    # parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=10)
    # parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)
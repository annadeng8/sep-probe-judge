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

    # Reformat dataset to include responses and annotations
    def reformat(example, j):
        try:
            completion = example['completions'][j]
            response = completion.get('response', 'No response found')
            annotations = completion.get('annotations', {})
            instruction_following = annotations.get('helpfulness', {})
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

    # Construct few-shot prompt using Instruction, Question, Answer, and Rationale
    def construct_fewshot_prompt(dataset, num_examples=2):
        prompt = "You are an evaluator of model response quality. Below are examples to guide your evaluation of helpfulness.\n\n"
        sampled_indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        # sampled_indices = [1,2,3]
        for idx in sampled_indices:
            example = dataset[idx]
            question = example['question']    
            answer = example['response']
            evaluation = example['evaluation']
            prompt += f"Question: {question}\nAnswer: {answer}\nEvaluation: {evaluation}\n\n"
        prompt += "Now, provide your evaluation for the following. Use the same format exactly:\nRating: <1-5>\nRationale: <brief explanation>\nDO NOT include anything else.\n\n"
        return prompt

     # Initialize model
    model = utils.init_model(args)

    # Process each split
    # Construct few-shot prompt for this specific example
    few_shot_prompt = construct_fewshot_prompt(train_dataset, num_examples=args.num_few_shot)

    for dataset_split, dataset in [('train', train_dataset), ('validation', test_dataset)]:
        print(f"Generating evaluations for {dataset_split} split")
        generations = {}

        # Limit to a small subset for efficiency
        indices = range(min(args.num_samples, len(dataset)))

        for index in indices:
            example = dataset[index]
            question = example["question"]
            test_answer = example["response"]  # Use the dataset's response as the answer to evaluate
            generations[example['id']] = {
                'context': question,
                'question': "Evaluate the following model response: " + test_answer,
                'responses': []  # initialize the responses key
            }

            # Combine few-shot prompt with current input
            current_input = f"Question: {question}\nAnswer: {test_answer}\nEvaluation:"
            local_prompt = few_shot_prompt + current_input

            # Print the full prompt before generating evaluations
            print("Few-shot prompt constructed:")
            print(local_prompt)

            # Generate n evaluations
            full_evaluations = []
            ratings = []
            num_generations = 10
            
            # Prepare N copies of the same prompt
            prompts = [local_prompt] * num_generations
            results = model.batch_predict(prompts, temperature=args.temperature, return_latent=True)

            full_evaluations = []
            ratings = []
            for predicted_evaluation, token_log_likelihoods, (embedding, _, _) in results:
                embedding = embedding.cpu() if embedding is not None else None
                full_evaluations.append((predicted_evaluation, token_log_likelihoods, embedding))
                try:
                    rating_str = predicted_evaluation.split("Rating: ")[1].split("\n")[0].strip()
                    rating = int(rating_str)
                except (IndexError, ValueError):
                    rating = None
                ratings.append(rating)
                print(f"Evaluation: {predicted_evaluation.replace(chr(10), ' ')}")

            # Compute entropy based on ratings
            valid_ratings = [r for r in ratings if r is not None]
            if valid_ratings:
                counts = np.unique(valid_ratings, return_counts=True)[1]
                probs = counts / len(valid_ratings)
                entropy = -np.sum(probs * np.log(probs))
            else:
                entropy = 0  # Default entropy if no valid ratings
            print(f"Entropy: {entropy:.4f}")

            # Minimal change: use key "responses" instead of "evaluations"
            generations[example['id']]['responses'] = full_evaluations
            generations[example['id']]['entropy'] = entropy
            # Clean up memory after each sample
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            del predicted_evaluation, token_log_likelihoods, embedding, results



        # Save generations
        utils.save(generations, f'{dataset_split}_generations.pkl', save_dir="/workspace/sep-temp-1")

    print("Run complete.")
    del model

if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=2, help="Number of few-shot examples")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)
"""Generate evaluations with LLM, cache activations, and compute entropy with few-shot prompting."""
import numpy as np
import torch
from datasets import load_dataset
from uncertainty.utils import utils
import hashlib
import random
import gc

from uncertainty.semantic_entropy import (
    get_semantic_ids,
    cluster_assignment_entropy,
    EntailmentDeberta
)

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

    # Construct few-shot prompt using Instruction, Question, Response, and Rationale
    def construct_fewshot_prompt(dataset, num_examples=5, char_limit=1000, max_attempts=50):
        prompt = f"You are an evaluator of text quality. Your task is to evaluate the helpfulness of responses.\n\nCRITICAL FORMAT RULES:\n1. Your response MUST be exactly two lines:\n   Rating: <number 1-5>\n   Rationale: <one sentence explanation>\n2. Do not include any other text, labels, or information\n3. Keep rationales brief and focused\n4. Do not repeat the question or instruction in your rationale\n5. Do not include any text after the rationale\n6. Your response MUST end after the rationale\n\n"
        used_indices = set()
        added = 0
        attempts = 0

        while added < num_examples and attempts < max_attempts:
            idx = random.randint(0, len(dataset) - 1)
            if idx in used_indices:
                attempts += 1
                continue

            used_indices.add(idx)
            example = dataset[idx]
            question = example['question']
            response = example['response']
            evaluation = example['evaluation']

            example_text = f"Question: {question}\nResponse: {response}\nEvaluation: {evaluation}\n\n"

            if len(example_text) <= char_limit:
                prompt += example_text
                added += 1
            else:
                # If too long, skip and try another example
                attempts += 1

        if added < num_examples:
            print(f"Warning: Only added {added}/{num_examples} examples due to character limit.")

        prompt += (
            "Now evaluate the following response. Remember:\n"
            "- Use EXACTLY two lines\n"
            "- First line: Rating: <number 1-5>\n"
            "- Second line: Rationale: <one sentence>\n"
            "- Do not include any other text\n"
            "- Do not include 'Evaluation:' or any prefix\n"
            "- Your response MUST end after the rationale\n"
        )
        return prompt


     # Initialize model
    model = utils.init_model(args)
    entailment_model = EntailmentDeberta()
    few_shot_prompt = construct_fewshot_prompt(train_dataset, num_examples=args.num_few_shot)


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
            test_response = example["response"]  # Use the dataset's response as the answer to evaluate
            generations[example['id']] = {
                'context': question,
                'question': "Evaluate the following model response: " + test_response,
                'responses': []  # initialize the responses key
            }

            # Combine few-shot prompt with current input
            current_input = f"Question: {question}\Response: {test_response}\nEvaluation:"
            local_prompt = few_shot_prompt + current_input

            # Print the full prompt before generating evaluations
            print("Few-shot prompt constructed:")
            print(local_prompt)

            num_generations = 10
            prompts = [local_prompt] * num_generations
            results = model.batch_predict(prompts, temperature=args.temperature, return_latent=True)

            responses, log_liks, embeddings = [], [], []
            for predicted_answer, token_log_likelihoods, (embedding, _, _) in results:
                embedding = embedding.cpu() if embedding is not None else None
                responses.append(predicted_answer)
                log_liks.append(token_log_likelihoods)
                embeddings.append(embedding)
                print(f"Answer: {predicted_answer.replace(chr(10), ' ')}")

            # Compute semantic entropy using entailment-based clustering
            try:
                semantic_ids = get_semantic_ids(responses, model=entailment_model, example=example)
                entropy = cluster_assignment_entropy(semantic_ids)
            except Exception as e:
                print(f"Error computing semantic entropy: {e}")
                entropy = 0.0

            print(f"Semantic Entropy: {entropy:.4f}")

            generations[example['id']].update({
                'responses': list(zip(responses, log_liks, embeddings)),
                'most_likely_answer': {
                    'response': responses[0],
                    'embedding': embeddings[0],
                },
                'entropy': entropy,
                'reference': example['response']
            })

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

        utils.save(generations, f'{dataset_split}_generations.pkl', save_dir="/workspace/sep-temp")

    print("Run complete.")
    del model

if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=2, help="Number of few-shot examples")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)
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
    def construct_fewshot_prompt(dataset, num_examples=3, char_limit=1000, max_attempts=50):
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

            example_text = f"Question: {question}\nResponse: {response}\nEvaluation: {evaluation} END\n\n"

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

        # To ensure exactly num_samples valid datapoints
        collected = 0
        index = 0
        dataset_len = len(dataset)
        max_trials = 2 * args.num_samples  # failsafe, to avoid infinite loop

        while collected < args.num_samples and index < dataset_len and index < max_trials:
            print(f"\n>>> Starting example {collected+1}/{args.num_samples} (dataset index {index})")
            example = dataset[index]
            index += 1

            question = example["question"]
            test_response = example["response"]
            current_input = f"Question: {question}\nResponse: {test_response}\nEvaluation:"
            local_prompt = few_shot_prompt + current_input

            # 1. Greedy generation
            print(" - [Step 1] Getting greedy generation...")
            try:
                greedy_result = model.batch_predict([local_prompt], temperature=0.1, return_latent=True)
                greedy_predicted_answer, _, (greedy_embedding, _, _) = greedy_result[0]
                greedy_embedding = greedy_embedding.cpu() if greedy_embedding is not None else None
            except Exception as e:
                print(f"Greedy generation failed: {e}")
                continue

            # 2. Sampling for SE
            print(" - [Step 2] Sampling for entropy...")
            try:
                num_generations = 10
                prompts = [local_prompt] * num_generations
                results = model.batch_predict(prompts, temperature=args.temperature, return_latent=True)

                responses, log_liks, embeddings = [], [], []
                for predicted_answer, token_log_likelihoods, (embedding, _, _) in results:
                    predicted_answer = clean_evaluation(predicted_answer)
                    rationale_only = predicted_answer.split("Rationale:")[-1].strip()
                    responses.append(rationale_only)
                    log_liks.append(token_log_likelihoods)
                    embeddings.append(embedding.cpu() if embedding is not None else None)
            except Exception as e:
                print(f"   ✖ Sampling failed: {e}")
                continue

            # 3. Entropy
            print(" - [Step 3] Computing semantic entropy...")
            try:
                semantic_ids = get_semantic_ids(responses, model=entailment_model, example=example)
                entropy = cluster_assignment_entropy(semantic_ids)
            except Exception as e:
                print(f"   ✖ Entropy computation failed: {e}")
                continue

            print(f" - ✓ Finished example {collected+1}/{args.num_samples}, entropy = {entropy:.3f}")


            # 4. Save the data point (greedy embedding for probe, responses for SE)
            generations[example['id']] = {
                'context': question,
                'question': "Evaluate the following model response: " + test_response,
                'responses': list(zip(responses, log_liks, embeddings)),
                'most_likely_answer': {
                    'response': greedy_predicted_answer,
                    'embedding': greedy_embedding,  # <--- always the greedy one!
                },
                'entropy': entropy,
                'reference': example['response']
            }

            collected += 1  # Only increment if all above succeeds!

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            if collected % 50 == 0:
                print(f"Collected {collected}/{args.num_samples} valid examples so far...")

        # Save output for this split
        utils.save(generations, f'{dataset_split}_generations.pkl', save_dir="/workspace/sep-temp")



    print("Run complete.")
    del model

if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=3, help="Number of few-shot examples")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)
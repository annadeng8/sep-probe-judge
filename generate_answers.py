"""Generate answers with LLM, cache activations, and compute semantic entropy using few-shot prompting."""
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

    # Load TriviaQA with streaming
    train_stream = load_dataset("mandarjoshi/trivia_qa", "rc", split="train", streaming=True)
    val_stream = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation", streaming=True)

    # Sample 500 examples from each stream
    train_dataset = [x for _, x in zip(range(500), train_stream)]
    test_dataset = [x for _, x in zip(range(500), val_stream)]

    def reformat(example):
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))
        return {
            'question': example['question'],
            'response': example['answer']['value'],
            'id': md5hash(str(example['question']))
        }

    train_dataset = [reformat(ex) for ex in train_dataset]
    test_dataset = [reformat(ex) for ex in test_dataset]

    def construct_fewshot_prompt(dataset, num_examples=5):
        prompt = "Answer briefly.\n\n"
        used = set()
        while len(used) < num_examples:
            idx = random.randint(0, len(dataset) - 1)
            if idx in used:
                continue
            used.add(idx)
            q = dataset[idx]['question']
            a = dataset[idx]['response']
            prompt += f"Question: {q}\nAnswer: {a}\n\n"
        return prompt

    model = utils.init_model(args)
    entailment_model = EntailmentDeberta()
    few_shot_prompt = construct_fewshot_prompt(train_dataset, num_examples=args.num_few_shot)

    for dataset_split, dataset in [('train', train_dataset), ('validation', test_dataset)]:
        print(f"Generating answers for {dataset_split} split")
        generations = {}
        indices = range(min(args.num_samples, len(dataset)))

        for index in indices:
            example = dataset[index]
            question = example["question"]

            generations[example['id']] = {
                'question': question,
                'responses': []
            }

            current_input = f"Q: {question}\nA:"
            local_prompt = few_shot_prompt + current_input
            print("Prompt:\n", local_prompt)

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

        utils.save(generations, f'{dataset_split}_generations.pkl', save_dir="/workspace/sep-temp-1")

    print("Run complete.")
    del model

if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=5, help="Number of few-shot examples")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)
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

    # Construct few-shot prompt using Instruction, Question, Answer, and Rationale
    def construct_fewshot_prompt(dataset, num_examples=5):
        prompt = """You are an evaluator of text quality. Below are examples to guide your evaluation. Each example includes an Instruction (the task), Question (the specific query), Answer (the response), and Rationale (the reasoning behind the rating). Use these to provide your evaluation.\n\n"""
        sampled_indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
        for idx in sampled_indices:
            example = dataset[idx]
            instruction = example['question']  # Instruction is the same as the question
            question = example['question']     # Question is the same as instruction
            answer = example['response']
            evaluation = example['evaluation']
            prompt += f"Instruction: {instruction}\nQuestion: {question}\nAnswer: {answer}\nEvaluation: {evaluation}\n\n"
        prompt += "Now, provide your evaluation for the following AND GIVE NOTHING ELSE:\n"
        return prompt

     # Initialize model
    model = utils.init_model(args)

    # Process each split
    # Construct few-shot prompt for this specific example
    few_shot_prompt = construct_fewshot_prompt(train_dataset, num_examples=args.num_few_shot)

    for dataset_split, dataset in [('train', train_dataset), ('validation', test_dataset)]:
        print(f"Generating evaluations for {dataset_split} split")
        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        # Evaluate over random subset of the datasets.
        indices = random.sample(range(0, len(dataset)), min(args.num_samples, len(dataset)))

        for index in indices:
            example = dataset[index]
            question = example["question"]
            test_answer = example["response"]  # Use the dataset's response as the answer to evaluate
            # Minimal change: use "context" for the original instruction, and prepend the evaluation prompt to the response
            generations[example['id']] = {
                'context': question,
                'question': "Evaluate the following model response: " + test_answer,
                'responses': []  # initialize the responses key
            }

            # Combine few-shot prompt with current input
            current_input = f"Instruction: {question}\nQuestion: {question}\nAnswer: {test_answer}\nEvaluation:"
            local_prompt = few_shot_prompt + current_input

            # Print the full prompt before generating evaluations
            print("Few-shot prompt constructed:")
            # print(local_prompt)

            # Generate n evaluations
            full_responses = []
            
            if dataset_split == 'train':
                num_generations = 1
            else:
                num_generations = 10
            
            # Prepare N copies of the same prompt
            prompts = [local_prompt] * num_generations
            # results = model.batch_predict(prompts, temperature=args.temperature, return_latent=True)
            
            for i in range(num_generations):

                # Temperature for first generation is always `0.1`.
                temperature = 0.1 if i == 0 else args.temperature

                predicted_answer, token_log_likelihoods, (embedding, emb_last_before_gen, emb_before_eos) = model.batch_predict(prompts, temperature, return_latent=True) 
                
                # Last token embedding
                embedding = embedding.cpu() if embedding is not None else None
                # emb_last_before_gen = emb_last_before_gen.cpu() if emb_last_before_gen is not None else None
                # emb_before_eos = emb_before_eos.cpu() if emb_before_eos is not None else None
                
                """
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0  # pylint: disable=invalid-name
                """
                
                if i == 0:
                    # accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        # 'accuracy': acc,
                        'emb_last_tok_before_gen': emb_last_before_gen,
                        'emb_tok_before_eos': emb_before_eos, 
                    }

                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example),
                    })
                else:
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, None))

            # Append all predictions for this example to `generations`.
            generations[example['id']]['responses'] = full_responses
            
            # Clean up memory after each sample
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            del token_log_likelihoods, embedding


        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy.
        # accuracy = np.mean(accuracies)
        # print(f"Overall {dataset_split} split accuracy: {accuracy}")

        if dataset_split == 'validation':
            utils.save(results_dict, 'uncertainty_measures.pkl')

    # utils.save(experiment_details, 'experiment_details.pkl')
    del model
            
if __name__ == '__main__':
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=5, help="Number of few-shot examples")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)
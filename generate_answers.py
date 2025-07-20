#!/usr/bin/env python3
"""
Generate evaluations with LLM, cache activations, and compute entropy with few-shot prompting.

Revision notes
--------------
• keep the entire two-line evaluation in `responses`
• print the prompt and *every* sampled answer to stdout
• **strict `clean_evaluation()`** – only accept answers that
    ▸ have “Rating:” 1-5
    ▸ have a non-empty “Rationale: …”
• malformed answers are silently skipped
• **prints entropy** for every kept batch
• **NEW**: shuffle train/validation example order so the same question
  is not repeatedly pulled

***Current hot-fix***
  – format filtering turned **off** (no answers discarded for bad format)
  – now prints progress `example/total` after every kept batch
"""
import sys
import re
import gc
import time
import random
import hashlib
import numpy as np
import torch
from datasets import load_dataset
from uncertainty.utils import utils
from uncertainty.semantic_entropy import (
    get_semantic_ids,
    cluster_assignment_entropy,
    EntailmentDeberta,
)

# Redirect all stdout (and stderr) to a file to capture terminal output
output_file = open('/workspace/sep-probe-judge/generate_answers_output.txt', 'w')
sys.stdout = output_file
sys.stderr = output_file

# --------------------------------------------------------------------------- #
#  Regexes kept for possible future use                                       #
# --------------------------------------------------------------------------- #
RATING_RE    = re.compile(r"^Rating:\s*[1-5]\s*$", re.I)
RATIONALE_RE = re.compile(r"^Rationale:\s*\S.*$",  re.I)


def clean_evaluation(text: str) -> str | None:
    """
    **Relaxed**: simply return the first two non-empty lines (if any).
    No regex validation – everything passes the filter.
    """
    if "Rating:" in text:
        text = text[text.index("Rating:") :]
    if "END" in text:
        text = text[: text.index("END")]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None  # still need at least two lines
    return f"{lines[0]}\n{lines[1]}"


# --------------------------------------------------------------------------- #
#  Main program                                                               #
# --------------------------------------------------------------------------- #
def main(args):
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # -------- 1. LOAD & SPLIT DATASET ---------------------------------------
    ds = load_dataset("openbmb/UltraFeedback")["train"].train_test_split(
        test_size=0.2, seed=42
    )
    train_raw, test_raw = ds["train"], ds["test"]

    # -------- 2. REFORMAT ----------------------------------------------------
    def reformat(ex, j):
        try:
            comp = ex["completions"][j]
            resp = comp.get("response", "No response found")
            ann  = comp.get("annotations", {}).get("helpfulness", {})
            md5  = lambda s: str(int(hashlib.md5(s.encode()).hexdigest(), 16))
            return {
                "question":   ex["instruction"],
                "response":   resp,
                "evaluation": f"Rating: {ann.get('Rating', '?')}\n"
                              f"Rationale: {ann.get('Rationale', '?')}",
                "id":         md5(ex["instruction"]),
            }
        except Exception:
            return None

    def unpack(raw):
        return [
            x for d in raw for j in range(4)
            if (x := reformat(d, j)) is not None
        ]

    train_ds, test_ds = unpack(train_raw), unpack(test_raw)

    # -------- NEW: shuffle so we don’t keep seeing the same example ----------
    random.shuffle(train_ds)
    random.shuffle(test_ds)

    # -------- 3. FEW-SHOT PROMPT --------------------------------------------
    def fewshot(dataset, k=3, limit=1000):
        prompt = (
            "You are an evaluator of text quality. Your task is to evaluate the helpfulness of responses.\n\n"
            "CRITICAL FORMAT RULES:\n"
            "1. Your response MUST be exactly two lines:\n"
            "   Rating: <number 1-5>\n"
            "   Rationale: <one sentence explanation>\n"
            "2. Do not include any other text, labels, or information\n"
            "3. Keep rationales brief and focused\n"
            "4. Do NOT repeat the question or instruction in your rationale\n"
            "5. Do NOT include 'Question:' or any prefix\n"
            "6. Do NOT include any text after the rationale\n"
            "7. Your response MUST end after the rationale\n\n"
            "Here are some examples of how to evaluate responses:\n\n"
        )
        for ex in random.sample(dataset, k):
            snippet = (
                f"Example Question: {ex['question']}\n"
                f"Example Response: {ex['response']}\n"
                f"Example Evaluation: {ex['evaluation']}\n\n"
            )
            prompt += snippet[:limit]
        prompt += (
            "Now evaluate the following NEW question and response. "
            "Focus ONLY on the question and response below. "
            "Do NOT reference any of the examples above.\n\n"
        )
        return prompt

    few_shot_prompt = fewshot(train_ds, k=args.num_few_shot)

    # -------- 4. INITIALISE MODELS ------------------------------------------
    model            = utils.init_model(args)
    entailment_model = EntailmentDeberta()

    # -------- 5. MAIN LOOP ---------------------------------------------------
    for split_name, data in [("train", train_ds), ("validation", test_ds)]:
        print(f"\n========== Generating evaluations for {split_name} split ==========")
        generations, collected, idx = {}, 0, 0

        while collected < args.num_samples and idx < len(data):
            ex = data[idx]
            idx += 1

            q, resp = ex["question"], ex["response"]
            lp = f"{few_shot_prompt}Question: {q}\nResponse: {resp}\nEvaluation:"

            # ----- Greedy ----------------------------------------------------
            try:
                g_ans, _, (g_last, g_sec, g_pre) = model.batch_predict(
                    [lp], temperature=0.1, return_latent=True
                )[0]
                greedy = clean_evaluation(g_ans)
                if greedy is None:
                    continue
            except Exception:
                continue

            # ----- Sampling --------------------------------------------------
            responses, log_liks, embeds = [], [], []
            blank_response_infos = []  # To store info about blank responses
            max_resample_rounds = 3
            resample_round = 0
            while resample_round <= max_resample_rounds:
                batch_responses, batch_log_liks, batch_embeds = [], [], []
                attempts = 0
                batch_blank_infos = []
                while len(batch_responses) < 10 and attempts < 40:
                    attempts += 1
                    ans, tls, (e_last, slt_embedding, tbg_embedding) = model.batch_predict(
                        [lp], temperature=args.temperature, return_latent=True
                    )[0]
                    clean = clean_evaluation(ans)
                    if clean is None or clean.strip() == "":
                        # Output info for blank model responses
                        print(f"[BLANK RESPONSE] Attempt {attempts} (Resample round {resample_round}): Raw model output was blank or malformed.")
                        print(f"Raw output: {repr(ans)}")
                        # Investigate top tokens if possible
                        if hasattr(model, 'get_top_tokens'):
                            top_tokens = model.get_top_tokens([lp])
                            print(f"Top tokens for blank response: {top_tokens}")
                        else:
                            print("[INFO] Model does not support top token extraction.")
                        batch_blank_infos.append({'attempt': attempts, 'raw_output': ans, 'resample_round': resample_round})
                        continue
                    batch_responses.append(clean)
                    batch_log_liks.append(tls)
                    batch_embeds.append(tbg_embedding)
                # Accumulate all responses and blank infos
                responses.extend(batch_responses)
                log_liks.extend(batch_log_liks)
                embeds.extend(batch_embeds)
                blank_response_infos.extend(batch_blank_infos)
                # Optionally, log blank response info to file
                if batch_blank_infos:
                    with open('/workspace/sep-probe-judge/generate_answers_output.txt', 'a', encoding='utf-8') as outf:
                        for info in batch_blank_infos:
                            try:
                                outf.write(f"[BLANK RESPONSE] Attempt {info['attempt']} (Resample round {info['resample_round']}): Raw model output was blank or malformed.\n")
                                outf.write(f"Raw output: {repr(info['raw_output'])}\n")
                            except UnicodeEncodeError:
                                # Skip writing this line if encoding fails
                                continue
                # Check if we need to resample
                if len([r for r in batch_responses if r.strip() == ""]) > 1 and resample_round < max_resample_rounds:
                    print(f"[RESAMPLE] More than one blank response in batch (round {resample_round}). Resampling...")
                    resample_round += 1
                else:
                    break
            if len(responses) < 3:
                continue

            # ----- Entropy ---------------------------------------------------
            try:
                sem_ids = get_semantic_ids(responses, entailment_model, strict_entailment=False, example=ex)
                from collections import defaultdict
                clusters = defaultdict(list)
                for resp, sid in zip(responses, sem_ids):
                    clusters[sid].append(resp)
                print("Clusters formed:")
                for cid, resps in clusters.items():
                    print(f"Cluster {cid}:")
                    for r in resps:
                        print(f"  - {r}")
                print(f"Cluster IDs: {sem_ids}")
                entropy = cluster_assignment_entropy(sem_ids)
                # Save outputs with 0 entropy in a separate file
                if entropy == 0:
                    with open('/workspace/sep-probe-judge/zero_entropy_outputs.txt', 'a') as zf:
                        zf.write(f"Split: {split_name}\n")
                        zf.write(f"Example ID: {ex['id']}\n")
                        zf.write("---------------- PROMPT ----------------\n")
                        zf.write(lp + "\n")
                        zf.write("------------ MODEL ANSWERS -------------\n")
                        for i, r in enumerate(responses, 1):
                            zf.write(f"{i}. {r}\n")
                        zf.write(f"Entropy: {entropy:.4f}\n")
                        zf.write(f"Reference Response: {resp}\n")
                        zf.write("----------------------------------------\n\n")
            except Exception:
                continue

            # ----- Print to terminal ----------------------------------------
            print("\n---------------- PROMPT ----------------")
            print(lp)
            print("------------ MODEL ANSWERS -------------")
            for i, r in enumerate(responses, 1):
                print(f"{i}. {r}")
            print(f"Entropy: {entropy:.4f}")
            print("----------------------------------------\n")

            # ----- Store -----------------------------------------------------
            generations[ex["id"]] = {
                "context": q,
                "question": "Evaluate the following model response: " + resp,
                "responses": list(zip(responses, log_liks, embeds)),
                "most_likely_answer": {
                    "response": greedy,
                    "last_embedding": g_last,
                    "sec_last_embedding": g_sec,
                    "prompt_last_embedding": g_pre,
                },
                "entropy": entropy,
                "reference": resp,
            }
            collected += 1

            # --- progress ----------------------------------------------------
            print(f"Progress: {collected}/{args.num_samples} examples processed")  # <-- NEW

        utils.save(
            generations, f"{split_name}_generations.pkl",
            save_dir="/workspace/sep-probe-judge"
        )

    print("Run complete.")
    del model
    torch.cuda.empty_cache()



# --------------------------------------------------------------------------- #
#  Entry-point                                                                #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = utils.get_parser()
    parser.add_argument("--num_few_shot", type=int, default=2,
                        help="number of few-shot examples in the prompt")
    args = parser.parse_args()
    print(f"Starting run with args: {args}")
    main(args)
"""Implement semantic entropy with DeBERTa only (no OpenAI dependency)."""
import logging
import numpy as np
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(10)

class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        logits = outputs.logits
        largest_index = torch.argmax(F.softmax(logits, dim=1))
        prediction = largest_index.cpu().item()
        return prediction


def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

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
            for j in range(i+1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum'):
    """Sum probabilities with the same semantic id."""
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum':
            logsumexp_value = np.log(np.sum(np.exp(id_log_likelihoods))) - 5.0
        elif agg == 'sum_normalized':
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        elif agg == 'mean':
            logsumexp_value = np.log(np.mean(np.exp(id_log_likelihoods)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy."""
    entropy = -np.sum(log_probs) / len(log_probs)
    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from cluster assignments."""
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy


def binarize_entropy(entropy_values, threshold=None):
    """
    Binarize entropy values using optimal threshold selection.
    
    If threshold is provided, use it. Otherwise, find optimal threshold
    using the paper's Equation 5 approach.
    """
    if threshold is not None:
        return entropy_values > threshold
    
    # Find optimal threshold as in paper's Equation 5
    best_threshold = None
    min_mse = float('inf')
    
    # Try different thresholds
    for threshold in np.linspace(np.min(entropy_values), np.max(entropy_values), 100):
        low_mask = entropy_values < threshold
        high_mask = ~low_mask
        
        # Need at least one sample in each group
        if np.any(low_mask) and np.any(high_mask):
            low_mean = np.mean(entropy_values[low_mask])
            high_mean = np.mean(entropy_values[high_mask])
            
            # Calculate MSE as in Equation 5
            mse = np.sum((entropy_values[low_mask] - low_mean)**2) + \
                  np.sum((entropy_values[high_mask] - high_mean)**2)
            
            if mse < min_mse:
                min_mse = mse
                best_threshold = threshold
    
    if best_threshold is None:
        # Fallback to median if no valid threshold found
        best_threshold = np.median(entropy_values)
    
    return entropy_values > best_threshold, best_threshold

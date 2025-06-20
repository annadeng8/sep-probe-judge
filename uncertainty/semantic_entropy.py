"""Implement semantic entropy with DeBERTa only (no OpenAI dependency)."""
import logging
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        # DeBERTa-MNLI returns: [contradiction, neutral, entailment]
        prediction = torch.argmax(F.softmax(logits, dim=1)).item()
        return prediction


def context_entails_response(context, responses, model):
    votes = [model.check_implication(context, response) for response in responses]
    return 2 - np.mean(votes)


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic clusters based on entailment."""

    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)

        if strict_entailment:
            return (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            return (0 not in implications) and ([1, 1] != implications)

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum'):
    """Aggregate log probabilities per semantic cluster ID using log-sum-exp logic."""
    unique_ids = sorted(set(semantic_ids))
    assert unique_ids == list(range(len(unique_ids)))

    log_likelihood_per_semantic_id = []
    for uid in unique_ids:
        idxs = [i for i, x in enumerate(semantic_ids) if x == uid]
        cluster_log_liks = [log_likelihoods[i] for i in idxs]

        if agg == 'sum':
            val = np.log(np.sum(np.exp(cluster_log_liks))) - 5.0
        elif agg == 'sum_normalized':
            normed = cluster_log_liks - np.log(np.sum(np.exp(log_likelihoods)))
            val = np.log(np.sum(np.exp(normed)))
        elif agg == 'mean':
            val = np.log(np.mean(np.exp(cluster_log_liks)))
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

        log_likelihood_per_semantic_id.append(val)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Standard entropy over log-likelihoods (negative average log probability)."""
    return -np.mean(log_probs)


def predictive_entropy_rao(log_probs):
    """Entropy weighted by exponentiated probabilities (for semantic mixture)."""
    return -np.sum(np.exp(log_probs) * log_probs)


def cluster_assignment_entropy(semantic_ids):
    """Entropy over semantic cluster distribution."""
    counts = np.bincount(semantic_ids)
    probs = counts / len(semantic_ids)
    assert np.isclose(probs.sum(), 1)
    return -np.sum(probs * np.log(probs))
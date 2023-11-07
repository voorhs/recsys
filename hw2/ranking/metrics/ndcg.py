import numpy as np
from math import log2

def gain(r):
    """Helper for `DCG`.
    - `r` is a relevance label of url"""
    return 2 ** r - 1

def discount(i):
    """Helper for `DCG`.
    - `i` is a rank of url (predicted or gold)"""
    return 1 / log2(i + 2)

def DCG(y_true):
    """Helper for `NDCG`. Calculates discounted cumulative gain.
    - `y_true`: list of relevance labels of ranked urls"""
    return sum(gain(r) * discount(i) for i, r in enumerate(y_true))

def IDCG(y_true):
    """Helper for `NDCG`. Calculates ideal discounted cumulative gain.
    - `y_true`: list of relevance labels of ranked urls"""
    return DCG(sorted(y_true, reverse=True))

def NDCG(y_true, y_scores, k=None, as_indices=False):
    """Calculates normalized discounted cumulative gain metric. Gain is 2^r-1, where r is a relevance label
    - `y_true`: mapping from query id to list of relevance labels for each ranked url
    - `y_score`: predictions following the same format
    - `as_indices`: set `True` when `y_scores` are preticted indices (not predicted scores), handy for lambda rank"""
    query_ids = y_true.keys()
    assert query_ids == y_scores.keys()

    res = 0
    for qid in query_ids:
        targets = y_true[qid]
        scores = y_scores[qid]

        if as_indices:
            indices = scores
        else:
            indices = np.argsort(scores)[::-1]
        if k is not None:
            indices = indices[:k]
        targets = [targets[i] for i in indices]
        
        idcg = IDCG(targets)
        if idcg == 0:
            ndcg = 0
        else:
            ndcg = DCG(targets) / idcg
        res += ndcg
    
    res /= len(query_ids)
    return res

def delta_NDCG(relevance_labels, scores):
    indices = np.argsort(scores)[::-1]
    discount_diff = 1 / np.log2(indices[:, None] + 2) - 1 / np.log2(indices[None, :] + 2)
    gain_diff = 2 ** relevance_labels[:, None] - 2 ** relevance_labels[None, :]
    idcg = IDCG(relevance_labels)
    return abs(gain_diff * discount_diff / idcg)

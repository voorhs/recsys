import numpy as np
from math import log2

def gain(r):
    """Helper for `DCG`.
    - `r` is a relevance label of url"""
    return 2 ** r - 1

def discount(i):
    """Helper for `DCG`.
    - `i` is a rank of url (predicted or gold)"""
    return 1 / log2(i + 1)

def DCG(y_true):
    """Helper for `NDCG`. Calculates discounted cumulative gain.
    - `y_true`: list of relevance labels of ranked urls"""
    return sum(gain(r) * discount(i) for i, r in enumerate(y_true))

def IDCG(y_true):
    """Helper for `NDCG`. Calculates ideal discounted cumulative gain.
    - `y_true`: list of relevance labels of ranked urls"""
    return DCG(sorted(y_true, reverse=True))

def NDCG(y_true, y_scores):
    """Calculates normalized discounted cumulative gain metric. Gain is 2^r-1, where r is a relevance label
    - `y_true`: mapping from query id to list of relevance labels for each ranked url
    - `y_score`: predictions following the same format"""
    query_ids = y_true.keys()
    assert query_ids == y_scores.keys()

    res = 0
    for qid in query_ids:
        targets = y_true[qid]
        scores = y_scores[qid]

        indices = np.argsort(scores)[::-1]
        targets = [targets[i] for i in indices]
        
        ndcg = DCG(targets) / IDCG(targets)
        res += ndcg
    
    res /= len(query_ids)
    return res

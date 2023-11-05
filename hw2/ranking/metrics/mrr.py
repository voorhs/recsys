import numpy as np

def MRR(y_true, y_scores, as_indices=False):
    """Calculate mean reciprocal rank metric.
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
        is_relevant = [targets[i] > 0 for i in indices]
        
        rank_of_first_relevant = is_relevant.index(True) + 1
        res += 1 / rank_of_first_relevant
    
    res /= len(query_ids)
    return res

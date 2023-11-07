import numpy as np

def precision(y_true):
    """Helper for `MAP`.
    - `y_true`: list of relevance labels of top retrieved urls"""
    num_of_relevant = sum(y > 0 for y in y_true)
    num_of_retrieved = len(y_true)
    return num_of_relevant / num_of_retrieved

def MAP(y_true, y_scores, k=None, as_indices=False):
    """Calculate mean average precision metric.
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
        
        sum_prec = sum(precision(targets[:i+1]) for i, t in enumerate(targets) if t > 0)
        num_of_relevant = sum(t > 0 for t in targets)
        
        if num_of_relevant == 0:
            avg_prec = 0
        else:
            avg_prec = sum_prec / num_of_relevant
        res += avg_prec
    
    res /= len(query_ids)
    return res

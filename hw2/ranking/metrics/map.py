import numpy as np

def precision(y_true):
    """Helper for `MAP`.
    - `y_true`: list of relevance labels of top retrieved urls"""
    num_of_relevant = sum(y > 0 for y in y_true)
    num_of_retrieved = len(y_true)
    return num_of_relevant / num_of_retrieved

def MAP(y_true, y_scores):
    """Calculate mean average precision metric.
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
        scores = [scores[i] for i in indices]
        
        sum_prec = sum(precision(targets[:i]) for i, t in enumerate(targets) if t > 0)
        num_of_relevant_urls = sum(t > 0 for t in targets)
        
        avg_prec = sum_prec / num_of_relevant_urls
        res += avg_prec
    
    res /= len(query_ids)
    return res

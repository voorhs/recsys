import numpy as np

def MRR(y_true, y_scores):
    """Calculate mean reciprocal rank metric.
    - `y_true`: mapping from query id to list of relevance labels for each ranked url
    - `y_score`: predictions following the same format"""
    query_ids = y_true.keys()
    assert query_ids == y_scores.keys()

    res = 0
    for qid in query_ids:
        targets = y_true[qid]
        scores = y_scores[qid]

        indices = np.argsort(scores)[::-1]
        is_relevant = [targets[i] > 0 for i in indices]
        
        rank_of_first_relevant = is_relevant.index(True) + 1
        res += 1 / rank_of_first_relevant
    
    res /= len(query_ids)
    return res

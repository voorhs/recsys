import numpy as np
from math import log2


def gain(r):
    return 2 ** r - 1

def discount(i):
    return 1 / log2(i + 2)

def DCG(y_true):
    return sum(gain(r) * discount(i) for i, r in enumerate(y_true))

def IDCG(y_true):
    return DCG(sorted(y_true, reverse=True))

def normalized_discounted_cumulative_gain(y_true, y_scores, top_k=10):
    res = []
    for targets, scores in zip(y_true, y_scores):
        idcg = IDCG(targets)
        if idcg == 0:
            ndcg = 0
        else:
            indices = np.argpartition(scores, kth=np.arange(-top_k, 0))[-top_k:][::-1]
            targets = [targets[i] for i in indices]
            ndcg = DCG(targets) / idcg
        res.append(ndcg)
    
    return np.mean(res)

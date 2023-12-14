import numpy as np

def hit_ratio(results, top_k=10, ref_idx=0):
    return np.mean([ref_idx in np.argpartition(feed, kth=-top_k)[-top_k:] for feed in results])

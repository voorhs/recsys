import torch
import torch.nn.functional as F
from .ranknet import RankNet
from typing import Literal
from ..metrics import MAP, MRR, NDCG, delta_NDCG
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

class LambdaRank(RankNet):
    def __init__(self, input_size, hidden_size, temperature, metric_to_optimize: Literal['map', 'mrr', 'ndcg']):
        super().__init__(input_size, hidden_size, temperature)

        self.metric_to_optimize = metric_to_optimize

        if metric_to_optimize == 'map':
            self.metric_fn = MAP
        elif metric_to_optimize == 'mrr':
            self.metric_fn = MRR
        elif metric_to_optimize == 'ndcg':
            self.metric_fn = NDCG

    def _calc_lambdas(self, relevance_labels, scores):
        is_less = relevance_labels[:, None] < relevance_labels[None, :]
        is_eq = relevance_labels[:, None] == relevance_labels[None, :]
        diff = scores[:, None] - scores[None, :]

        abs_delta_metric = self._calc_metric_deltas(relevance_labels, scores).to(scores.device)

        lambda_ij = -1 * F.sigmoid(diff / self.temperature) / self.temperature * abs_delta_metric
        lambda_ij[is_less] *= -1
        lambda_ij[is_eq] *= 0
        return torch.sum(lambda_ij, dim=1)
    
    def _calc_metric_deltas(self, relevance_labels, scores):
        """this function is not optimized but straightforward and generic"""
        relevance_labels = relevance_labels.cpu().numpy().astype(int)

        if self.metric_to_optimize == 'ndcg':
            res = delta_NDCG(relevance_labels, scores.detach().cpu().numpy())
            return torch.from_numpy(res)
        
        indices = scores.detach().cpu().numpy().argsort()[::-1]
        orig_metric = self._calc_metric_from_indices(relevance_labels, indices)
        
        n = len(indices)
        args = []
        for i in range(n-1):
            for j in range(i+1, n):
                args.append((self.metric_fn, i, j, relevance_labels, indices))
        
        new_metrics = process_map(_calc_metric_after_swap, args, max_workers=4, chunksize=4, disable=True)
        
        # new_metrics = map(_calc_metric_after_swap, args)

        res = torch.zeros(size=(n,n))
        for arg, new_metric in zip(args, new_metrics):
            i, j = arg[1:3]
            delta = abs(new_metric - orig_metric)
            res[i, j] = delta
            res[j, i] = delta
        
        return res
    
    def _calc_metric_from_indices(self, relevance_labels, indices):
        return self.metric_fn({'0': relevance_labels}, {'0': indices}, as_indices=True)

def _calc_metric_after_swap(args):
    metric_fn, i, j, relevance_labels, indices = args

    relevance_labels = relevance_labels.copy()
    indices = indices.copy()

    for x in [relevance_labels, indices]:
        x[[i,j]] = x[[j,i]]
    
    res = metric_fn({'0': relevance_labels}, {'0': indices}, as_indices=True)

    return res
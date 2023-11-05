from torch import nn
import torch
import torch.nn.functional as F
from ..metrics import MAP, MRR, NDCG

class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size, temperature):
        """two layer MLP"""
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.temperature = temperature

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, batch, return_score=True, compute_lambdas=True, compute_metrics=True):
        """`batch` is a tuple of (`features`, `relevance_labels`), where
        - `features`: `(n, d)`
        - `relevance_labels`: `(n,)`"""
        
        res = []

        features, relevance_labels = [x.squeeze(0) for x in batch]
        scores = self.net(features).squeeze(1)
        if return_score:
            res.append(scores)
        
        if compute_lambdas:
            lambdas = self._calc_lambdas(relevance_labels, scores)
            res.append(lambdas)
        if compute_metrics:
            metrics = self._calc_metrics(relevance_labels, scores)
            res.append(metrics)

        return res
    
    def _calc_lambdas(self, relevance_labels, scores):
        is_less = relevance_labels[:, None] < relevance_labels[None, :]
        is_eq = relevance_labels[:, None] == relevance_labels[None, :]
        diff = scores[:, None] - scores[None, :]
        lambda_ij = -1 * F.sigmoid(diff / self.temperature) / self.temperature
        lambda_ij[is_less] *= -1
        lambda_ij[is_eq] *= 0
        return torch.sum(lambda_ij, dim=1)
    
    def _calc_metrics(self, relevance_labels, scores):
        y_true = relevance_labels.cpu().numpy().astype(int)
        y_score = scores.detach().cpu().numpy()
        inputs = ({'0': y_true}, {'0': y_score})
        return {
            'map': MAP(*inputs),
            'mrr': MRR(*inputs),
            'ndcg': NDCG(*inputs),
        }

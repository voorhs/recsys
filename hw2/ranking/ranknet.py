from torch import nn
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


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
    
    def forward(self, batch, compute_loss=True):
        """`batch` is a tuple of (`features`, `relevance_labels`), where
        - `features`: `(n, d)`
        - `relevance_labels`: `(n,)`"""
        
        # feed
        features, relevance_labels = batch
        features = features.squeeze(0)
        relevance_labels = relevance_labels.squeeze(0)
        scores = self.net(features).squeeze(1)

        if not compute_loss:
            return scores
        
        # find lambdas
        is_less = relevance_labels[:, None] < relevance_labels[None, :]
        is_eq = relevance_labels[:, None] == relevance_labels[None, :]
        diff = scores[:, None] - scores[None, :]
        lambda_ij = -1 * F.sigmoid(diff / self.temperature) / self.temperature
        lambda_ij[is_less] *= -1
        lambda_ij[is_eq] *= 0

        lambdas = torch.sum(lambda_ij, dim=1)

        # compute metric
        y_true = (relevance_labels > 0).cpu().numpy().astype(int)
        y_score = scores.detach().cpu().numpy()
        metric = average_precision_score(y_true, y_score)

        return scores, lambdas, metric

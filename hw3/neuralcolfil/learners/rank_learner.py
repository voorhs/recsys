from dataclasses import dataclass
from .base import BaseLearner, BaseLearnerConfig
from torchmetrics.retrieval import RetrievalHitRate as HR, RetrievalNormalizedDCG as NDCG
import torch
import torch.nn.functional as F


@dataclass
class RankLearnerConfig(BaseLearnerConfig):
    top_k: int = 10
    feed_size: int = 100
    temperature: float = 1.


class RankLearner(BaseLearner):
    def __init__(self, model, config: RankLearnerConfig):
        super().__init__()
        
        self.model = model
        self.config = config
        self.automatic_optimization = False

        self.ndcg = NDCG(empty_target_action='neg', top_k=config.top_k)
        self.hr = HR(empty_target_action='neg', top_k=config.top_k)

        self.scores = []
        self.lambdas = []

    def forward(self, batch, return_score, compute_lambdas, compute_metrics):
        """feed features for a single query: (n, d)"""
        features, relevance_labels = batch
        
        res = []
        scores = self.model(features)
        
        if return_score:
            res.append(scores)
        
        if compute_lambdas:
            lambdas = self.calc_lambdas(relevance_labels, scores)
            res.append(lambdas)
        
        if compute_metrics:
            metrics = self.calc_metrics(relevance_labels, scores)
            res.append(metrics)
        
        return res
        
    def training_step(self, batch, batch_idx):
        """some inspiration came from here: https://github.com/haowei01/pytorch-examples/blob/master/ranking/LambdaRank.py#L212"""
        
        scores, lambdas, metrics = self.forward(
            batch,
            return_score=True,
            compute_lambdas=True,
            compute_metrics=True
        )

        self.scores.append(scores)
        self.lambdas.append(lambdas)
        
        if (batch_idx + 1) % self.config.batch_size == 0:
            opt = self.optimizers()
            opt.zero_grad()
            for scores, lambdas in zip(self.scores, self.lambdas):
                scores.backward(gradient=lambdas)
            opt.step()
            self.lr_schedulers().step()
            self.scores.clear()
            self.lambdas.clear()

        self.log_dict(
            dictionary={f'train_{name}': val for name, val in metrics.items()},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
    
    def validation_step(self, batch, batch_idx):
        metrics, = self.forward(
            batch,
            return_score=False,
            compute_lambdas=False,
            compute_metrics=True
        
        )
        self.log_dict(
            dictionary={f'val_{name}': val for name, val in metrics.items()},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )

    def calc_lambdas(self, relevance_labels, scores):
        is_less = relevance_labels[:, None] < relevance_labels[None, :]
        is_eq = relevance_labels[:, None] == relevance_labels[None, :]
        diff = scores[:, None] - scores[None, :]
        lambda_ij = -1 * F.sigmoid(diff / self.config.temperature) / self.config.temperature
        lambda_ij[is_less] *= -1
        lambda_ij[is_eq] *= 0
        return torch.sum(lambda_ij, dim=1)
    
    def calc_metrics(self, targets, preds):
        indexes = torch.zeros_like(targets).long()
        res = {}
        res[f'ndcg@{self.config.top_k}'] = self.ndcg(preds, targets, indexes)
        res[f'hr@{self.config.top_k}'] = self.hr(preds, targets != 0, indexes)
        return res

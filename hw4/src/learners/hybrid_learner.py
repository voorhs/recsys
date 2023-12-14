from dataclasses import dataclass
from torch import nn
import torch
from torchmetrics.retrieval import RetrievalHitRate as HR, RetrievalNormalizedDCG as NDCG
from .colfil_learner import ColFilLearner



class HybridLearner(ColFilLearner):
    def forward(self, batch, return_ids):
        targets, user_ids, item_ids, user_features, item_features = batch

        scores = self.model(user_ids, item_ids, user_features, item_features)

        res = [scores, targets]
        if return_ids:
            res += [user_ids]
        
        return res

from dataclasses import dataclass
from torch import nn
import torch
from torchmetrics.retrieval import RetrievalHitRate as HR, RetrievalNormalizedDCG as NDCG
from .base import BaseLearner, BaseLearnerConfig


@dataclass
class ColFilLearnerConfig(BaseLearnerConfig):
    top_k: int = 10
    n_negatives_train: int = 3
    n_negatives_val: int = 100


class ColFilLearner(BaseLearner):
    def __init__(self, model, config: ColFilLearnerConfig):
        super().__init__()
        
        self.model = model
        self.config = config

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.ndcg = NDCG(empty_target_action='neg', top_k=config.top_k)
        self.hr = HR(empty_target_action='neg', top_k=config.top_k)

    def forward(self, batch, return_ids):
        user_ids = batch[:, 0]
        item_ids = batch[:, 1]
        pos_targets = torch.ones_like(user_ids)
        
        negative_item_ids = [batch[:, i] for i in range(2, batch.shape[1])]
        n_negatives = len(negative_item_ids)
        neg_targets = torch.zeros_like(user_ids).repeat(n_negatives)
        
        users = user_ids.repeat(1 + n_negatives)
        items = torch.cat([item_ids] + negative_item_ids, dim=0)
        targets = torch.cat([pos_targets, neg_targets], dim=0)

        scores = self.model(users, items)

        res = [scores, targets]
        if return_ids:
            res += [users]
        
        return res
   
    def training_step(self, batch, batch_idx):
        scores, targets = self.forward(batch, return_ids=False)
        
        loss = self.loss_fn(scores, targets.float())
        
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        # print('step')

        return loss
    
    def validation_step(self, batch, batch_idx):
        scores, targets, users = self.forward(batch, return_ids=True)
        
        metrics = self.metric_fn(scores, targets, users)

        self.log_dict(
            dictionary={f'val_{name}': val for name, val in metrics.items()},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
    
    def metric_fn(self, preds, targets, indexes):
        res = {}
        res[f'ndcg@{self.config.top_k}'] = self.ndcg(preds, targets, indexes)
        res[f'hr@{self.config.top_k}'] = self.hr(preds, targets, indexes)
        return res

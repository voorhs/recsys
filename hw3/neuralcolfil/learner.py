from dataclasses import dataclass, asdict
import lightning.pytorch as pl
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
import torch
import math
from torchmetrics.retrieval import RetrievalHitRate as HR, RetrievalNormalizedDCG as NDCG


@dataclass
class LearnerConfig:
    max_lr: float = 1e-3
    lr_div_factor: float = 1e3
    batch_size: int = 4
    warmup_pct: float = 0.1
    weight_decay = 1e-2
    betas : tuple = (0.9, 0.999)
    total_steps: int = None
    lr_decay: int = False

    top_k: int = 10
    n_negatives_train: int = 3
    n_negatives_val: int = 100


class Learner(pl.LightningModule):
    def __init__(self, model, config: LearnerConfig):
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

    def on_train_start(self):
        # hparams = self.model.get_hparams()
        hparams = asdict(self.config)
        self.logger.log_hyperparams(hparams)

    def get_optim_groups(self):
        """Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def configure_optimizers(self):
        optim_groups = self.get_optim_groups()
        optimizer = AdamW(optim_groups, lr=self.config.max_lr, betas=self.config.betas)

        if self.config.lr_decay:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                div_factor=self.config.lr_div_factor,
                final_div_factor=self.config.lr_div_factor,
                pct_start=self.config.warmup_pct,
                total_steps=self.config.total_steps
            )
        else:
            def cosine_warmup_no_decay(step):
                lr_max = self.config.max_lr
                lr_min = lr_max / self.config.lr_div_factor

                warmup_pct = self.config.warmup_pct
                total_steps = self.config.total_steps
                warmup_steps = math.floor(warmup_pct * total_steps)
                
                if step < warmup_steps:
                    lr = lr_max - 0.5 * (lr_max - lr_min) * (1 + math.cos(step / warmup_steps * math.pi))
                    return lr / lr_max
                
                return 1
            
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=cosine_warmup_no_decay
            )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}
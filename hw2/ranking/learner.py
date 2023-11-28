from dataclasses import dataclass, asdict
import lightning.pytorch as pl
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
from torch.cuda.amp import GradScaler
from torch import autocast
import torch
import math


@dataclass
class LearnerConfig:
    max_lr: float = 1e-3
    lr_div_factor: float = 1e3
    batch_size: int = 32
    warmup_pct: float = 0.1
    weight_decay = 1e-2
    betas = (0.9, 0.999)
    n_epochs: int = None
    steps_per_epoch: int = None
    lr_decay: int = False


class Learner(pl.LightningModule):
    def __init__(self, model, config: LearnerConfig):
        super().__init__()
        
        self.model = model
        self.config = config
        self.automatic_optimization = False

        self.scaler = GradScaler()

        self.scores = []
        self.lambdas = []

    def forward(self, batch, **kwargs):
        """`url_features` for a single query: (n, d)"""
        return self.model(batch, **kwargs)
        
    def training_step(self, batch, batch_idx):
        """some inspiration came from here: https://github.com/haowei01/pytorch-examples/blob/master/ranking/LambdaRank.py#L212"""
        
        with autocast(device_type='cuda', dtype=torch.float16):
            scores, lambdas, metrics = self(batch, return_score=True, compute_lambdas=True, compute_metrics=True)
        self.scores.append(scores)
        self.lambdas.append(lambdas)
        
        if (batch_idx + 1) % self.config.batch_size == 0:
            opt = self.optimizers()
            opt.zero_grad()
            for scores, lambdas in zip(self.scores, self.lambdas):
                self.scaler.scale(scores).backward(gradient=lambdas)
            self.scaler.step(opt)
            self.scaler.update()
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
        metrics, = self(batch, return_score=False, compute_lambdas=False, compute_metrics=True)
        self.log_dict(
            dictionary={f'val_{name}': val for name, val in metrics.items()},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )

    @staticmethod
    def get_default_config():
        raise NotImplementedError()

    def on_train_start(self):
        # hparams = self.model.get_hparams()
        hparams = asdict(self.config)
        self.logger.log_hyperparams(hparams)

    def get_optim_groups(self):
        """Taken from https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136"""
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d)
        blacklist_weight_modules = (nn.BatchNorm2d)
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
                epochs=self.config.n_epochs,
                steps_per_epoch=self.config.steps_per_epoch
            )
        else:
            def cosine_warmup_no_decay(step):
                lr_max = self.config.max_lr
                lr_min = lr_max / self.config.lr_div_factor

                warmup_pct = self.config.warmup_pct
                total_steps = self.config.n_epochs * self.config.steps_per_epoch
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

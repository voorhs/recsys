from dataclasses import dataclass,  asdict
import lightning.pytorch as pl
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
import math


@dataclass
class BaseLearnerConfig:
    max_lr: float = 1e-3
    lr_div_factor: float = 1e3
    batch_size: int = 4
    warmup_pct: float = 0.1
    weight_decay = 1e-2
    betas : tuple = (0.9, 0.999)
    total_steps: int = None
    lr_decay: bool = False


class BaseLearner(pl.LightningModule):
    config: BaseLearnerConfig
    
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

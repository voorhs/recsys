import torch
from torch import nn
from .gmf import GMF as GMF
from .mlp import MLP as MLP
from ..train_utils import HParamsPuller, LightningCkptLoadable


class NCF(nn.Module, HParamsPuller, LightningCkptLoadable):
    def __init__(self, gmf: GMF, mlp: MLP):
        super().__init__()

        self.gmf = gmf
        self.mlp = mlp

        self.projector = self._get_projector()
    
    def _get_projector(self):
        gmf_projector = self.gmf.projector
        mlp_projector = self.mlp.projector

        in_features = gmf_projector.in_features + mlp_projector.in_features
        
        res = nn.Linear(in_features, 1, bias=False)
        weight = torch.cat([gmf_projector.weight.data/2, mlp_projector.weight.data/2], dim=1)
        res.weight = nn.Parameter(weight)

        return res
    
    def forward(self, users, items):
        features = self.get_features(users, items)
        score = self.projector(features).squeeze(dim=1)

        return score
    
    def get_features(self, users, items):
        gmf_features = self.gmf.get_features(users, items)
        mlp_features = self.mlp.get_features(users, items)

        return torch.cat([gmf_features, mlp_features], dim=1)

import torch
from torch import nn
from .gmf import GMF as GMF
from .mlp import MLP as MLP
from ..train_utils import HParamsPuller, LightningCkptLoadable


class HybridRecommender(nn.Module, HParamsPuller, LightningCkptLoadable):
    def __init__(self, gmf: GMF, mlp: MLP, with_hidden_layers=False):
        super().__init__()

        self.gmf = gmf
        self.mlp = mlp
        self.with_hidden_layers = with_hidden_layers
        if with_hidden_layers:
            self.fc = nn.Linear(51, 51)

        self.projector = self._get_projector()
    
    def _get_projector(self):
        gmf_projector = self.gmf.projector
        mlp_projector = self.mlp.projector

        in_features = gmf_projector.in_features + mlp_projector.in_features + 51
        
        res = nn.Linear(in_features, 1, bias=False).to(gmf_projector.weight.data.device)
        weight = torch.cat([gmf_projector.weight.data/2, mlp_projector.weight.data/2, res.weight.data[:, -51:]], dim=1)
        res.weight = nn.Parameter(weight)

        return res
    
    def forward(self, *args):
        features = self.get_features(*args)
        score = self.projector(features).squeeze(dim=1)

        return score
    
    def get_features(self, user_ids, item_ids, user_features, item_features):
        gmf_features = self.gmf.get_features(user_ids, item_ids)
        mlp_features = self.mlp.get_features(user_ids, item_ids)
        other_features = torch.cat([user_features, item_features], dim=1).float()

        if self.with_hidden_layers:
            other_features = self.fc(other_features)

        return torch.cat([gmf_features, mlp_features, other_features], dim=1)

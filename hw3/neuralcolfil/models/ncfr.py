import torch
from torch import nn
from . import GMF, MLP, RankNet
from ..train_utils import HParamsPuller, LightningCkptLoadable


class NCFR(nn.Module, HParamsPuller, LightningCkptLoadable):
    def __init__(self, gmf: GMF, mlp: MLP, ranknet: RankNet):
        super().__init__()

        self.gmf = gmf
        self.mlp = mlp
        self.ranknet = ranknet

        self.projector = self._get_projector()
    
    def _get_projector(self):
        gmf_projector = self.gmf.projector
        mlp_projector = self.mlp.projector
        ranknet_projector = self.ranknet.projector

        in_features = gmf_projector.in_features + mlp_projector.in_features + ranknet_projector.in_features
        
        res = nn.Linear(in_features, 1, bias=False)
        weight = [
            gmf_projector.weight.data/3,
            mlp_projector.weight.data/3,
            ranknet_projector.weight.data/3
        ]
        res.weight = nn.Parameter(torch.cat(weight, dim=1))

        return res
    
    def forward(self, users, items):
        features = self.get_features(users, items)
        score = self.projector(features).squeeze(dim=1)

        return score
    
    def get_features(self, users, items):
        gmf_features = self.gmf.get_features(users, items)
        mlp_features = self.mlp.get_features(users, items)
        user_features = self.mlp.embed_user(users)
        item_features = self.mlp.embed_item(items)
        ranknet_features = self.ranknet.get_features(user_features * item_features)
        features = [
            gmf_features,
            mlp_features,
            ranknet_features
        ]
        return torch.cat(features, dim=1)

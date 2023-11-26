from torch import nn
from .gmf import GeneralizedMatrixFactorization as GMF
from .mlp import MLPCollaborativeFilterer as MLP


class NeuralCollaborativeFilterer(nn.Module):
    def __init__(self, gmf: GMF, mlp: MLP):
        super().__init__()

        self.gmf = gmf
        self.mlp = mlp
    
    def forward(self, users, items):
        gmf_score = self.gmf(users, items)
        mlp_score = self.mlp(users, items)

        return (gmf_score + mlp_score) / 2

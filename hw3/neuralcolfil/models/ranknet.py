from torch import nn
from ..train_utils import HParamsPuller, LightningCkptLoadable


class RankNet(nn.Module, HParamsPuller, LightningCkptLoadable):
    def __init__(self, input_size, hidden_size, n_hidden_layers, temperature):
        """two layer MLP"""
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.temperature = temperature

        hidden_layers = []
        for _ in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Sequential(*hidden_layers),
        )

        self.projector = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, features):
        """
        - `features`: `(n, d)` features for single query
        """
        
        scores = self.projector(self.get_features(features)).squeeze(dim=1)
        
        return scores
    
    def get_features(self, features):
        return self.net(features)

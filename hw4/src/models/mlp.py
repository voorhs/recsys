from torch import nn
import torch
from ..train_utils import HParamsPuller, LightningCkptLoadable


class MLP(nn.Module, HParamsPuller, LightningCkptLoadable):
    def __init__(self, n_users, n_items, embedding_dim, hidden_sizes):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes

        self.embed_user = nn.Embedding(n_users, embedding_dim)
        self.embed_item = nn.Embedding(n_items, embedding_dim)

        layers = []
        hidden_sizes = [2 * embedding_dim] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            in_features, out_features = hidden_sizes[i:i+2]
            layer = nn.Linear(in_features, out_features)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ReLU(inplace=True))
        
        self.projector = nn.Linear(out_features, 1, bias=False)
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, users, items):
        """
        - `users`: `(B,)` user ids
        - `items`: `(B,)` item ids
        """
        features = self.get_features(users, items)
        score = self.projector(features).squeeze(dim=1)

        return score
    
    def get_features(self, users, items):
        user_embeddings = self.embed_user(users)
        item_embeddings = self.embed_item(items)

        input_features = torch.cat([user_embeddings, item_embeddings], dim=1)
        return self.layers(input_features)

from torch import nn
from typing import Literal
from functools import partial
import torch


class MLPCollaborativeFilterer(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_units_list, act_fn: Literal['relu', 'sigmoid', 'gelu', 'elu-hard', 'elu-moderate']):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_units_list = n_units_list
        self.act_fn = act_fn

        if act_fn == 'relu':
            act_fn = partial(nn.ReLU, inplace=True)
        elif act_fn == 'sigmoid':
            act_fn = nn.Sigmoid
        elif act_fn == 'gelu':
            act_fn == nn.GELU
        elif act_fn == 'elu-hard':
            act_fn = partial(nn.ELU, inplace=True, alpha=1)
        elif act_fn == 'elu-moderate':
            act_fn = partial(nn.ELU, inplace=True, alpha=0.3)
        else:
            raise ValueError(f'unknown act_fn {act_fn}')

        self.embed_user = nn.Embedding(n_users, embedding_dim)
        self.embed_item = nn.Embedding(n_items, embedding_dim)

        layers = []
        n_units_list = [2 * embedding_dim] + n_units_list
        for i in range(len(n_units_list)-1):
            in_features, out_features = n_units_list[i:i+2]
            layer = nn.Linear(in_features, out_features)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(act_fn())
        layers.append(nn.Linear(out_features, 1, bias=False))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, users, items):
        """
        - `users`: `(B,)` user ids
        - `items`: `(B,)` item ids
        """
        user_embeddings = self.embed_user(users)
        item_embeddings = self.embed_user(items)

        input_features = torch.cat([user_embeddings, item_embeddings], dim=1)
        score = self.layers(input_features)

        return score

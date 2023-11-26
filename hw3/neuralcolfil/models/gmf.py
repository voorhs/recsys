from torch import nn
from ..train_utils import HParamsPuller, LightningCkptLoadable


class GeneralizedMatrixFactorization(nn.Module, HParamsPuller, LightningCkptLoadable):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()

        self.n_items = n_items
        self.n_users = n_users
        self.embedding_dim = embedding_dim

        self.embed_user = nn.Embedding(n_users, embedding_dim)
        self.embed_item = nn.Embedding(n_items, embedding_dim)

        self.projector = nn.Linear(embedding_dim, 1, bias=False)
    
    def forward(self, users, items):
        """
        - `users`: `(B,)` user ids
        - `items`: `(B,)` item ids
        """
        user_embeddings = self.embed_user(users)
        item_embeddings = self.embed_user(items)

        score = self.projector(user_embeddings * item_embeddings).squeeze(dim=1)

        return score

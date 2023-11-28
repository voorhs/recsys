import torch
import os
from typing import Literal
from .implicit import RekkoImplicit
import numpy as np
from ..models import MLP


class RekkoRanking(RekkoImplicit):
    def __init__(
            self,
            feed_size,
            split: Literal['train', 'val'],
            embedding_path='/home/ilya/repos/recsys/data/rekko/embeddings-v2',
            dataframe_path='/home/ilya/repos/recsys/data/rekko/my_splits'
        ):
        super().__init__(n_negatives=0, split=split, path=dataframe_path)
        
        self.feed_size = feed_size
        self.embedding_path = embedding_path
        self.dataframe_path = dataframe_path

        self.embed_user, self.embed_item = load_embeddings(embedding_path)

        self.all_users = self.df.user_id.unique().tolist()

    def __len__(self):
        return len(self.all_users)

    def __getitem__(self, i):
        user_id = self.all_users[i]
        mask = self.df.user_id == user_id
        batch = self.df[mask]

        rating = batch.rating.to_numpy()
        rating = (rating >= 7).astype('float') + (rating >= 9).astype('float')

        item_ids = batch.item_id.to_list()
        n_negatives = self.feed_size - len(rating)
        
        if n_negatives > 0:
            item_ids += self._negative_sampling(user_id, k=n_negatives)
            rating = np.concatenate([rating, np.zeros(shape=(n_negatives,))])
        
        user = self.embed_user(torch.LongTensor([user_id]))
        items = self.embed_item(torch.LongTensor(item_ids))

        features = items * user

        return features, rating

def load_embeddings(path):
    model = MLP(
        n_users=9939,
        n_items=6852,
        embedding_dim=64,
        hidden_sizes=[128, 128, 64, 64, 32]
    )
    user_path = os.path.join(path, 'user.pth')
    model.embed_user.load_state_dict(torch.load(user_path, map_location='cpu'))
    model.embed_user.requires_grad_(False)
    item_path = os.path.join(path, 'item.pth')
    model.embed_item.load_state_dict(torch.load(item_path, map_location='cpu'))
    model.embed_item.requires_grad_(False)
    return model.embed_user, model.embed_item

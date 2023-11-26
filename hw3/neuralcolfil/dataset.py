from torch.utils.data import Dataset
from typing import Literal
import os
import json
import pandas as pd
from random import sample


class RekkoDataset(Dataset):
    def __init__(
            self,
            n_negatives,
            split: Literal['train', 'val'],
            path='/home/ilya/repos/recsys/data/rekko/retr_splits',
        ):
        self.n_negatives = n_negatives
        self.split = split
        self.path = path
        
        df_path = os.path.join(path, f'{split}.csv')
        self.df = pd.read_csv(df_path)
        
        positive_sets_path = os.path.join(path, 'positive_sets.json')
        positive_sets = json.load(open(positive_sets_path, 'r'))
        self.positive_sets = {int(user_id): set(pos_items) for user_id, pos_items in positive_sets.items()}

        all_items_path = os.path.join(path, 'all_items.json')
        all_items = json.load(open(all_items_path, 'r'))
        self.all_items = set(all_items)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, i):
        user_id, item_id = [int(x) for x in self.df.loc[i, ['user_id', 'item_id']]]
        negative_item_ids = self.all_items.difference(self.positive_sets[user_id])
        negative_item_ids = sample(negative_item_ids, k=self.n_negatives)

        res = [user_id, item_id] + negative_item_ids
        return res
        
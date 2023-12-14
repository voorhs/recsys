from typing import Literal
import os
import json
import pandas as pd
from .implicit import RekkoImplicit
import numpy as np


class RekkoHybrid(RekkoImplicit):
    def __init__(
            self,
            n_negatives,
            split: Literal['train', 'val'],
            path='/home/ilya/repos/recsys/data/rekko',
        ):
        super().__init__(n_negatives, split, os.path.join(path, 'my_splits'))
        
        self.path_preprocessed = os.path.join(path, 'hw4')

        self.user_inner2orig = json.load(open(os.path.join(self.path_preprocessed, 'user_inner2orig.json'), 'r'))
        self.item_inner2orig = json.load(open(os.path.join(self.path_preprocessed, 'item_inner2orig.json'), 'r'))
        
        # item features
        self.catalogue = pd.read_csv(os.path.join(self.path_preprocessed, 'catalogue.csv'))

        # user features
        self.delta_ts = json.load(open(os.path.join(self.path_preprocessed, 'delta_ts.json'), 'r'))
        self.n_bookmarks = json.load(open(os.path.join(self.path_preprocessed, 'n_bookmarks.json'), 'r'))
        self.users_with_features = set(self.delta_ts.keys())

        # item-user pairs features
        # self.transactions = pd.read_csv(os.path.join(self.path_preprocessed, 'transactions.csv'))
    
    def __getitem__(self, i):
        columns_to_retrieve = ['user_id', 'item_id', 'user_uid', 'element_uid']
        user_id, item_id, user_uid, element_uid = [int(x) for x in self.df.loc[i, columns_to_retrieve]]
        negative_item_ids = self._negative_sampling(user_id, k=self.n_negatives)
        item_ids = np.array([item_id] + negative_item_ids)
        element_uids = [self.item_inner2orig[x] for x in item_ids]

        if user_uid in self.users_with_features:
            user_features = np.array([self.n_bookmarks[user_uid], self.delta_ts[user_uid]])
        else:
            user_features = np.zeros(2)
        
        # mask = (self.transactions['element_uid'].isin(element_uids)) & (self.transactions['user_uid'] == user_uid)
        # sub_transactions = self.transactions[mask]

        # pair_features = []
        # for element_uid in element_uids:
        #     mask = (sub_transactions['element_uid'] == element_uid) & (sub_transactions['user_uid'] == user_uid)
        #     tmp = sub_transactions[mask]
        #     if tmp.shape[0] == 0:
        #         features = np.array([1] + [0] * 14)
        #     else:
        #         columns_to_drop = ['element_uid', 'user_uid', 'consumption_mode', 'device_type', 'device_manufacturer']
        #         features = tmp.drop(columns=columns_to_drop).iloc[0].tolist()
        #         features = np.array([0] + features)
        #     pair_features.append(features)

        # pair_features = np.stack(pair_features, axis=0)

        mask = self.catalogue['element_uid'].isin(element_uids)
        item_features = self.catalogue[mask].drop(columns=['element_uid']).to_numpy()
        user_features = np.repeat(user_features[None, :], repeats=len(element_uids), axis=0)

        user_ids = np.array([user_id] * len(element_uids))
        targets = np.array([1] + [0] * self.n_negatives)

        return targets, user_ids, item_ids, user_features, item_features#, pair_features

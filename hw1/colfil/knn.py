from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import numpy as np
from typing import Literal
from collections import defaultdict
import pandas as pd


def knn_user_based(
        df_train,
        df_test,
        k,
        metric : Literal['cosine', 'msd'] = 'cosine',
        min_support=1,
        verbose=False,
        with_means=False
    ):
    if verbose: print('...preprocessing')
    df_train, df_test, matr_train, matr_test, test_ids = _preprocess(df_train, df_test)
    
    if verbose: print('...counting common items')
    n_common_items = _n_common_items(matr_train, matr_test, test_ids)
    
    if verbose: print('...computing similarities')
    sim = _similarities(matr_train, matr_test, metric, n_common_items, min_support)
    user2items = df_test.groupby('_user_id')['_item_id'].apply(lambda x: list(x)).to_dict()
    
    if verbose: print('...finding neighbors')
    neighbor_ids = _k_neighbors(matr_train, sim, k, n_common_items, min_support, user2items, test_ids)
    
    if verbose: print('...predicting')
    mu = df_train.rating.mean()
    preds = _predictions(matr_train, test_ids, user2items, neighbor_ids, sim, with_means, mu)
    
    df_test: pd.DataFrame = df_test.merge(pd.DataFrame(preds), on=['_user_id', '_item_id'], how='left')
    df_test = df_test.rename(columns={'rating_x': 'rating', 'rating_y': 'pred_rating'})
    
    return df_train, df_test

def _preprocess(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()

    # define raw and inner ids for users and items (inner ids are for indexing arrays)
    raw_user_ids = list(set(df_train.user_id.unique()).union(set(df_test.user_id.unique())))
    raw_item_ids = list(set(df_train.item_id.unique()).union(set(df_test.item_id.unique())))
    n_users = len(raw_user_ids)
    n_items = len(raw_item_ids)
    for df in [df_train, df_test]:
        df['_user_id'] = df.user_id.map(lambda i: raw_user_ids.index(i))
        df['_item_id'] = df.item_id.map(lambda i: raw_item_ids.index(i))

    matr_train = csr_matrix((df_train.rating, (df_train._user_id, df_train._item_id)), shape=(n_users, n_items))

    # extract users that require predictions
    test_ids = df_test._user_id.unique()
    matr_test = matr_train[test_ids]

    return df_train, df_test, matr_train, matr_test, test_ids.tolist()

def _n_common_items(matr_train, matr_test, test_ids):
    # compute number of common items for each pair (test_user, train_user)
    # n_common_items = np.zeros((matr_test.shape[0], matr_train.shape[0]))
    # for i, test_id in enumerate(test_ids):
    #     for train_id in range(matr_train.shape[0]):
    #         n_common_items[i, train_id] = len(matr_train[test_id].multiply(matr_train[train_id]).data)
    matr_train_ = matr_train.copy()
    matr_test_ = matr_test.copy()
    matr_train_[matr_train_.nonzero()] = 1
    matr_test_[matr_test_.nonzero()] = 1
    n_common_items = matr_test_.dot(matr_train_.T).todense()
    # zero attention to common items between duplicate objects
    n_common_items[np.arange(matr_test.shape[0]), test_ids] = 0
    return n_common_items

def _similarities(matr_train, matr_test, metric, n_common_items, min_support):
    # compute pairwise similarities
    if metric == 'cosine':
        sim = pairwise_distances(matr_test, matr_train, metric='cosine')
    elif metric == 'msd':
        eu_dists = pairwise_distances(matr_test, matr_train, metric='euclidean')
        sim = n_common_items / (1 + eu_dists)
    else:
        raise ValueError(f'unknown metric {metric}')

    # zero attention to distances between duplicate objects and unacceptable neighbors
    sim[n_common_items < min_support] = -np.inf

    return np.array(sim)

def _k_neighbors(matr_train, sim, k, n_common_items, min_support, user2items, test_ids):
    # find k closest users with acceptable number of common items
    neighbor_ids = []
    for test_id, train_ids in zip(test_ids, np.argsort(-sim, axis=1)):
        neighbor_ids_for_user = []
        for item_id in user2items[test_id]:
            neighbor_ids_for_item = []
            for train_id in train_ids:
                if len(neighbor_ids_for_item) >= k:
                    break
                if n_common_items[test_ids.index(test_id), train_id] < min_support:
                    continue
                if matr_train[train_id, item_id] == 0:
                    continue
                neighbor_ids_for_item.append(train_id)
            neighbor_ids_for_user.append(neighbor_ids_for_item)
        neighbor_ids.append(neighbor_ids_for_user)
    return neighbor_ids

def _predictions(matr_train, test_ids, user2items, neighbor_ids, sim, with_means, mu):
    if with_means:
        matr_train = matr_train.copy()
        userwise_sums = matr_train.sum(axis=1).A1
        counts = np.diff(matr_train.indptr)
        userwise_means = userwise_sums / counts
    res = defaultdict(list)
    for j, test_id in enumerate(test_ids):
        for i, item_id in enumerate(user2items[test_id]):
            train_ids = np.array(neighbor_ids[j][i])
            if len(train_ids) != 0:
                weights = sim[j, train_ids]
                denom = weights.sum()

                if with_means:
                    values = matr_train[train_ids].toarray()[:, item_id] - userwise_means[train_ids]
                    nom = (values * weights).sum(axis=0)
                    pred = nom / denom + userwise_means[test_id]
                else:
                    values = matr_train[train_ids].toarray()[:, item_id]
                    nom = (values * weights).sum(axis=0)
                    pred = nom / denom
                
                res['rating'].append(pred)
                res['impossible'].append(False)
            else:
                res['rating'].append(mu)
                res['impossible'].append(True)
            res['_user_id'].append(test_id)
            res['_item_id'].append(item_id)
    return res

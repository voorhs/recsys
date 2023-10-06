import numpy as np
import random
import math
from .svd import (
    _preprocess as _svd_preprocess,
    _init as _svd_init
)

def svdpp(
        df_train,
        df_test,
        n_factors=100,
        n_epochs=20,
        batch_size=256,
        init_mean=0,
        init_std_dev=.1,
        lr=.005,
        reg=.02,
        random_state=None,
        verbose=False,
        return_logs=False
    ):
    np.random.seed(random_state)

    _svd_preprocess(df_train, df_test)
    n_users = len(df_train.user_id.unique())
    n_items = len(df_train.item_id.unique())
    bu, bi, P, Q, yj = _init(n_users, n_items, n_factors, init_mean, init_std_dev)
    rmse = _optimize(bu, bi, P, Q, yj, df_train, lr, reg, n_epochs, batch_size, verbose)
    
    df_test['pred_rating'] = _predictions(df_train, df_test, bu, bi, P, Q)
    
    if return_logs:
        return rmse

def _predictions(df_train, df_test, bu, bi, P, Q):
    mu = df_train['rating'].mean()
    test_data = df_test.to_dict('list')
    bu_test = bu[test_data['_user_id']]
    bi_test = bi[test_data['_item_id']]
    pu_test = P[test_data['_user_id']]
    qi_test = Q[test_data['_item_id']]

    return mu + bu_test + bi_test + np.sum(pu_test * qi_test, axis=1)

def _init(n_users, n_items, n_factors, init_mean, init_std_dev):
    bu, bi, P, Q = _svd_init(n_users, n_items, n_factors, init_mean, init_std_dev)
    yj = np.random.normal(
        loc=init_mean,
        scale=init_std_dev,
        size=(n_items, n_factors)
    )

    return bu, bi, P, Q, yj

def _inversed_root(df):
    return df.groupby('_user_id')['_item_id'].apply(lambda x: list(x)).to_list()

def _optimize(bu, bi, P, Q, yj, df_train, lr, reg, n_epochs, batch_size, verbose, logging_times=4):
    user2items = _inversed_root(df_train)
    data = df_train.to_dict('list')
    n_data = df_train.shape[0]
    logging_interval = math.ceil(n_data / batch_size) // logging_times
    mu = df_train['rating'].mean()

    epoch_rmse = []
    for i_epoch in range(n_epochs):
        if verbose:
            print(f'=== {i_epoch=} ===')
        epoch_indices = list(range(n_data))
        random.shuffle(epoch_indices)

        batch_rmse = []

        for i_batch in range(n_data // batch_size):
            start = i_batch * batch_size
            end = (i_batch + 1) * batch_size
            
            user_ids = [data['_user_id'][i] for i in epoch_indices[start:end]]
            item_ids = [data['_item_id'][i] for i in epoch_indices[start:end]]
            ratings = [data['rating'][i] for i in epoch_indices[start:end]]
            
            item_lists = [user2items[u] for u in user_ids]
            inversed_roots = [len(lst) ** (-0.5) for lst in item_lists]
            implicit_feedback = np.stack([yj[items_rated].sum(axis=0) * denom for items_rated, denom in zip(item_lists, inversed_roots)], axis=0)

            hat_ratings = mu + bu[user_ids] + bi[item_ids] + np.sum(Q[item_ids] * (P[user_ids] + implicit_feedback), axis=1)
            errors = ratings - hat_ratings

            # gradients
            bu_anti_grads = errors - reg * bu[user_ids]
            bi_anti_grads = errors - reg * bi[item_ids]
            P_anti_grads = errors[:, None] * Q[item_ids] - reg * P[user_ids]
            Q_anti_grads = errors[:, None] * (P[user_ids] + implicit_feedback) - reg * Q[item_ids]
            yj_anti_grad = np.zeros_like(yj)
            for error, item_id, items_to_update, inv_root in zip(errors, item_ids, item_lists, inversed_roots):
                anti_grad = error * inv_root * Q[item_id] - reg * yj[item_id]
                yj_anti_grad[items_to_update] += anti_grad
            
            # optimization step
            np.add.at(bu, user_ids, lr * bu_anti_grads)
            np.add.at(bi, item_ids, lr * bi_anti_grads)
            np.add.at(P, user_ids, lr * P_anti_grads)
            np.add.at(Q, item_ids, lr * Q_anti_grads)
            yj += lr * yj_anti_grad

            rmse = np.mean(errors ** 2) ** 0.5
            batch_rmse.append(rmse)

            if verbose and i_batch % logging_interval == 0:
                print(f'{i_batch=}, {rmse=:.4f}')
        
        epoch_rmse.append(batch_rmse)
        if verbose:
            print()

    return epoch_rmse
import numpy as np
import random
import math
from .svd import _preprocess, _init

def svdals(
        df_train,
        df_test,
        n_factors=100,
        n_epochs=10,
        init_mean=0,
        init_std_dev=.1,
        reg=.02,
        random_state=None,
        verbose=False,
        return_logs=False
    ):
    np.random.seed(random_state)

    df_train, df_test, n_users, n_items = _preprocess(df_train, df_test)
    _, _, P, Q = _init(n_users, n_items, n_factors, init_mean, init_std_dev)
    rmse = _optimize(P, Q, df_train, reg, n_epochs, verbose)
    
    df_test['pred_rating'] = _predictions(df_test, P, Q)
    
    if return_logs:
        return rmse
    
    return df_train, df_test

def _predictions(df_test, P, Q):
    pu_test = P[df_test['_user_id']]
    qi_test = Q[df_test['_item_id']]

    return np.sum(pu_test * qi_test, axis=1)

def _optimize(P, Q, df_train, reg, n_epochs, verbose):
    epoch_rmse = []
    for i_epoch in range(n_epochs):
        if verbose:
            print(f'{i_epoch=}', end=' ')
        
        for user_id in df_train._user_id.unique():
            cur_df = df_train[df_train._user_id == user_id]
            qi = Q[cur_df._item_id]
            ru = cur_df.rating
            A = reg * np.eye(qi.shape[1]) + qi.T @ qi
            b = qi.T @ ru
            P[user_id] = np.linalg.solve(A, b)
        
        for item_id in df_train._item_id.unique():
            cur_df = df_train[df_train._item_id == item_id]
            pu = P[cur_df._user_id]
            ri = cur_df.rating
            A = reg * np.eye(pu.shape[1]) + pu.T @ pu
            b = pu.T @ ri
            Q[item_id] = np.linalg.solve(A, b)
        
        hat_R = P @ Q.T
        hat_ratings = hat_R[df_train._user_id, df_train._item_id]
        train_errors = hat_ratings - df_train.rating
        train_rmse = (train_errors ** 2).mean() ** 0.5

        epoch_rmse.append(train_rmse)
        if verbose:
            print(f'{train_rmse=}')

    return epoch_rmse

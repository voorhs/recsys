import numpy as np
import random
import math

def svd(
        df_train,
        df_test,
        n_factors=100,
        n_epochs=10,
        batch_size=256,
        init_mean=0,
        init_std_dev=.1,
        biased=True,
        lr=.005,
        reg=.02,
        random_state=None,
        verbose=False,
        return_logs=False
    ):
    np.random.seed(random_state)

    df_train, df_test, n_users, n_items = _preprocess(df_train, df_test)
    bu, bi, P, Q = _init(n_users, n_items, n_factors, init_mean, init_std_dev)
    rmse = _optimize(bu, bi, P, Q, df_train, lr, reg, n_epochs, batch_size, verbose, biased)
    
    df_test['pred_rating'] = _predictions(df_train, df_test, bu, bi, P, Q)
    
    if return_logs:
        return rmse
    
    return df_train, df_test

def _predictions(df_train, df_test, bu, bi, P, Q):
    mu = df_train['rating'].mean()
    test_data = df_test.to_dict('list')
    bu_test = bu[test_data['_user_id']]
    bi_test = bi[test_data['_item_id']]
    pu_test = P[test_data['_user_id']]
    qi_test = Q[test_data['_item_id']]

    return mu + bu_test + bi_test + np.sum(pu_test * qi_test, axis=1)

def _init(n_users, n_items, n_factors, init_mean, init_std_dev):
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    
    P = np.random.normal(
        loc=init_mean,
        scale=init_std_dev,
        size=(n_users, n_factors)
    )
    Q = np.random.normal(
        loc=init_mean,
        scale=init_std_dev,
        size=(n_items, n_factors)
    )
    return bu, bi, P, Q

def _preprocess(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()

    # define raw and inner ids for users and items (inner ids are for indexing arrays)
    raw_user_ids = list(set(df_train.user_id.unique()).union(set(df_test.user_id.unique())))
    raw_item_ids = list(set(df_train.item_id.unique()).union(set(df_test.item_id.unique())))

    user_ids_orig2inner = {x: i for i, x in enumerate(raw_user_ids)}
    item_ids_orig2inner = {x: i for i, x in enumerate(raw_item_ids)}

    for df in [df_train, df_test]:
        df['_user_id'] = df.user_id.map(user_ids_orig2inner)
        df['_item_id'] = df.item_id.map(item_ids_orig2inner)

    n_users = len(raw_user_ids)
    n_items = len(raw_item_ids)

    return df_train, df_test, n_users, n_items

def _optimize(bu, bi, P, Q, df_train, lr, reg, n_epochs, batch_size, verbose, biased, logging_times=4):
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
            
            hat_ratings = mu + bu[user_ids] + bi[item_ids] + np.sum(P[user_ids] * Q[item_ids], axis=1)
            error = ratings - hat_ratings

            # gradients
            if biased:
                bu_anti_grads = error - reg * bu[user_ids]
                bi_anti_grads = error - reg * bi[item_ids]
            P_anti_grads = error[:, None] * Q[item_ids] - reg * P[user_ids]
            Q_anti_grads = error[:, None] * P[user_ids] - reg * Q[item_ids]
            
            # optimization step
            if biased:
                np.add.at(bu, user_ids, lr * bu_anti_grads)
                np.add.at(bi, item_ids, lr * bi_anti_grads)
            np.add.at(P, user_ids, lr * P_anti_grads)
            np.add.at(Q, item_ids, lr * Q_anti_grads)

            rmse = np.mean(error ** 2) ** 0.5
            batch_rmse.append(rmse)

            if verbose and i_batch % logging_interval == 0:
                print(f'{i_batch=}, {rmse=:.4f}')
        
        epoch_rmse.append(batch_rmse)
        if verbose:
            print()

    return epoch_rmse

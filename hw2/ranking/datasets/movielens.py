from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np


class MovielensDataset(Dataset):
    def __init__(self, split, path='/home/ilya/repos/recsys/data/ml-100k', user_based=True):
        if split == 'train':
            name = 'ua.base'
        elif split == 'test':
            name = 'ua.test'
        else:
            raise ValueError(f'unknown split {split}')
        
        movies_path = os.path.join(path, 'u.item')
        ratings_path = os.path.join(path, name)
        users_path = os.path.join(path, 'u.user')

        df_movies = _get_movies(movies_path)
        df_ratings = _get_ratings(ratings_path, df_movies)
        df_users = _get_users(users_path)
        
        df_movies.drop(columns=['release date'], inplace=True)
        df = pd.merge(df_ratings, df_movies, how='left', on='item_id')
        df = pd.merge(df, df_users, how='left', on='user_id')

        if user_based:
            dct = (df.groupby('user_id')['rating'].sum() > 0).to_dict()
            self.user_ids = [user_id for user_id, exist_relevant in dct.items() if exist_relevant]
        else:
            dct = (df.groupby('item_id')['rating'].sum() > 0).to_dict()
            self.item_ids = [item_id for item_id, exist_relevant in dct.items() if exist_relevant]

        for name in ['days_after_release', 'age']:
            _normalize(df, name)
        
        self.dataset = df
        self.user_based = user_based
        self.n_features = self[0][0].shape[1]
    
    def __len__(self):
        if self.user_based:
            return len(self.user_ids)
        
        return len(self.item_ids)
    
    def __getitem__(self, i):
        if self.user_based:
            mask = self.dataset.user_id == self.user_ids[i]
        else:
            mask = self.dataset.item_id == self.item_ids[i]
        
        batch = self.dataset[mask]
        targets = batch['rating'].to_numpy()
        features = batch.drop(columns=['user_id', 'item_id', 'rating']).to_numpy().astype(float)
        
        return features, targets


def _get_movies(path):
    df_movies = pd.read_csv(
        path,
        sep='|', encoding_errors='replace',
        names=['movie id', 'movie title' , 'release date' , 'video release date' ,
                'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
                'Children' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
                'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
                'Thriller' , 'War' , 'Western'],
    ).drop(columns=['video release date', 'IMDb URL', 'unknown', 'movie title'])
    df_movies.rename(columns={'movie id': 'item_id'}, inplace=True)
    df_movies['release date'] = pd.to_datetime(df_movies['release date'])
    return df_movies


def _get_ratings(path, df_movies):
    def encode_cycle(df, name):
        max_val = df[name].max()
        df[f'{name}_sin'] = np.sin(2 * np.pi * df[name] / max_val)
        df[f'{name}_cos'] = np.cos(2 * np.pi * df[name] / max_val)
    df_ratings = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df_ratings['timestamp'] = pd.to_datetime(df_ratings.timestamp, unit='s')
    df_ratings['hour'] = df_ratings.timestamp.dt.hour
    encode_cycle(df_ratings, 'hour')
    df_ratings.drop(columns=['hour'], inplace=True)
    df_ratings['rating'] = df_ratings['rating'].apply(lambda x: (x>3)*(x-3))
    df_movies_dates = df_movies[['item_id', 'release date']]
    tmp = pd.merge(df_ratings, df_movies_dates, how='left', on='item_id')
    df_ratings['days_after_release'] = (tmp['timestamp'] - tmp['release date']).dt.days
    df_ratings.drop(columns=['timestamp'], inplace=True)
    df_ratings.ffill(inplace=True)
    return df_ratings


def _get_users(path):
    df_users = pd.read_csv(
        path,
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip code']
    ).drop(columns=['zip code'])
    one_hot_occupation = pd.get_dummies(df_users.occupation, drop_first=True)
    df_users = pd.concat([df_users, one_hot_occupation], axis=1).drop(columns=['occupation'])
    df_users['gender'] = df_users['gender'].apply(lambda x: int(x == 'M'))
    return df_users


def _normalize(df, name):
    col = df[name]
    df[name] = (col - col.mean()) / col.std()

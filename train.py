# Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time

from envs import OfflineEnv
from recommender import DRRAgent

import os
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')
STATE_SIZE = 10
MAX_EPISODE_NUM = 1000

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":

    print('Data loading...')

    # Loading datasets - whole dataset 
    # ratings_list = [i.strip().split("::") for i in open(
    #     os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
    # users_list = [i.strip().split("::") for i in open(
    #     os.path.join(DATA_DIR, 'users.dat'), 'r').readlines()]
    # movies_list = [i.strip().split("::") for i in open(
    #     os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]
    # ratings_df = pd.DataFrame(ratings_list, columns=[
    #                           'UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=np.uint32)
    # movies_df = pd.DataFrame(movies_list, columns=[
    #                          'MovieID', 'Title', 'Genres'])
    # movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    # Loading dataset v2 - interacted items only  
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ml_1m.inter'), sep=',', dtype=np.uint32)
    ratings_df.columns = ['UserID', 'MovieID', 'Rating']

    users_list = np.loadtxt(os.path.join(DATA_DIR, 'users.csv'), dtype=int)

    movies_df = pd.read_csv(os.path.join(DATA_DIR, 'items.csv'), dtype=int)
    movies_df.columns = ['MovieID']

    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로 - 안쓸거임
    movies_id_to_movies = {movie[0]: movie[0:] for movie in movies_df.values}
    ratings_df = ratings_df.applymap(int)

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load(DATA_DIR + '/user_dict_new.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load(DATA_DIR + '/users_histroy_len_new.npy')

    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1

    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k: users_dict.item().get(k)
                        for k in range(1, train_users_num+1)}
    train_users_history_lens = users_history_lens[:train_users_num]

    print('DONE!')
    time.sleep(2)

    env = OfflineEnv(train_users_dict, train_users_history_lens,
                     movies_id_to_movies, STATE_SIZE)
    recommender = DRRAgent(env, users_num, items_num,
                           STATE_SIZE, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(MAX_EPISODE_NUM, load_model=False)

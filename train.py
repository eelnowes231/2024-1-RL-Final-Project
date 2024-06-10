# Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time
import argparse

from data import load_interact_dataset, load_whole_dataset
from envs import OfflineEnv
from recommender import DRRAgent

import os
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')
STATE_SIZE = 5
MAX_EPISODE_NUM = 8000

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, help='Modality')
    parser.add_argument('--fusion', type=str, help='Fusion')
    parser.add_argument('--aggregation', type=str, help='Aggregation')
    args = parser.parse_args()
    if args.modality:
        args.modality = tuple(args.modality.split(','))
    # Loading datasets - whole dataset 
    # ratings_df, users_dict, users_history_lens, movies_id_to_movies = load_whole_dataset(DATA_DIR)

    # Loading dataset v2 - interacted items only  
    ratings_df, users_dict, users_history_lens, movies_id_to_movies = load_interact_dataset(DATA_DIR)

    # 6039명의 user, 2819개의 영화
    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1
    
    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k: users_dict.item().get(k) for k in range(train_users_num)}
    train_users_history_lens = users_history_lens[:train_users_num]

    print('DONE!')
    time.sleep(2)

    TOP_K = 5
    env = OfflineEnv(train_users_dict, train_users_history_lens,
                     movies_id_to_movies, STATE_SIZE)
    print(f"Available number of users: {len(env.available_users)}")
  
    recommender = DRRAgent(env, users_num, items_num,
                           STATE_SIZE, args, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(MAX_EPISODE_NUM, load_model=False, top_k=TOP_K)

import numpy as np
import time
import os
import tensorflow as tf
import pandas as pd
from envs import OfflineEnv
from recommender import DRRAgent
"""
[Evaluation 방식 - Offline Evaluation (Algorithm 2)]
- eval_user_list에서 한명씩 평가진행
- 각 time step마다, 학습된 policy로 action 취하고 item 추천 -> reward 관찰, state update되고 추천된 item은 추천가능 목록에서 제거
- 한 user 당 몇번의 추천을 진행할지는 결정해야 할 듯 (Jupyter notebook에서는 한번만 하는 것으로 보이는데 알고리즘 상에는 T번해서 평균 내는 듯)
"""

# Version from Shanu
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')
STATE_SIZE = 10

def evaluate(recommender, env, check_movies: bool=False, top_k: int=1, length: int=1):
    # episodic reward
    mean_precision = 0
    mean_ndcg = 0

    # episodic reward
    episode_reward = 0
    steps = 0

    # Environment    
    user_id, items_ids, done = env.reset()
    print(f"[STARTING RECOMMENDATION TO USER {user_id}]")
    if check_movies:
        print(f'user_id : {user_id}, rated_items_length:{len(env.user_items)}')
        # print('history items : \n', np.array(env.get_items_names(items_ids)))

    while not done:
        # Observe current state & Find action        
        # Embedding        
        user_eb = recommender.embedding_network.get_layer('user_embedding')(np.array(user_id))
        items_eb = recommender.embedding_network.get_layer('movie_embedding')(np.array(items_ids))

        # SRM state        
        state = recommender.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])

        # Action(ranking score)        
        action = recommender.actor.network(state)

        # Item recommendation 
        recommended_item = recommender.recommend_item(action, env.recommended_items, top_k=top_k)

        # reward 받기
        next_items_ids, reward, done, _ = env.step(recommended_item, top_k=top_k)

        if check_movies:
            print(f'\t[step: {steps+1}] recommended items ids : {recommended_item}, reward : {reward}')
            # print(f'recommened items : \n {np.array(env.get_items_names(recommended_item), dtype=object)}')

        correct_list = [1 if r > 0 else 0 for r in reward]

        # ndcg
        dcg, idcg = calculate_ndcg(correct_list, [1 for _ in range(len(reward))])        
        mean_ndcg += dcg/idcg

        # precision
        correct_num = len(reward)-correct_list.count(0)
        mean_precision += correct_num/len(reward)

        reward = np.sum(reward)
        items_ids = next_items_ids
        episode_reward += reward
        steps += 1

        # 각 user마다 length만큼 추천
        if done or steps >= length:
            break

    if check_movies:
        print(f"\tprecision@{top_k} : {mean_precision/steps}, ndcg@{top_k} : {mean_ndcg/steps}, episode_reward : {episode_reward/steps}\n")

    return mean_precision, mean_ndcg, episode_reward/steps


def calculate_ndcg(rel, irel):
    dcg = 0
    idcg = 0
    rel = [1 if r > 0 else 0 for r in rel]
    for i, (r, ir) in enumerate(zip(rel, irel)):
        dcg += (r)/np.log2(i+2)
        idcg += (ir)/np.log2(i+2)
    return dcg, idcg


if __name__ == "__main__":
    print('Data loading...')

    # Loading datasets
    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
    users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'users.dat'), 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]
    ratings_df = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=object)
    movies_df = pd.DataFrame(movies_list, columns=['MovieID', 'Title', 'Genres'])
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
    ratings_df = ratings_df.applymap(int)

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load(ROOT_DIR + '/data/user_dict.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load(ROOT_DIR + '/data/users_histroy_len.npy')

    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1

    # Validation setting
    eval_users_num = int(users_num * 0.2)
    eval_items_num = items_num
    eval_users_dict = {k:users_dict.item().get(k) for k in range(users_num-eval_users_num, users_num)}
    eval_users_history_lens = users_history_lens[-eval_users_num:]

    print("DONE!")
    time.sleep(2)

    #######################################################
    saved_actor = './save_model/trail-2024-06-07-16-59-23/actor_8000_fixed.h5'
    saved_critic = './save_model/trail-2024-06-07-16-59-23/critic_8000_fixed.h5'

    tf.keras.backend.set_floatx('float64')

    TOP_K = 5
    LENGTH = 100

    sum_precision, sum_ndcg = 0, 0

    end_evaluation = 10

    for i, user_id in enumerate(eval_users_dict.keys()):
        env = OfflineEnv(eval_users_dict, users_history_lens, movies_id_to_movies, STATE_SIZE, fix_user_id=user_id)
        recommender = DRRAgent(env, users_num, items_num, STATE_SIZE)
        recommender.actor.build_networks()
        recommender.critic.build_networks()
        recommender.load_model(saved_actor, saved_critic)
        precision, ndcg, _ = evaluate(recommender, env, check_movies=True, top_k=TOP_K, length=LENGTH)
        sum_precision += precision
        sum_ndcg += ndcg

        if i > end_evaluation:
            break

    print("\n[FINAL RESULT]")
    print(f'precision@{TOP_K} : {sum_precision/len(eval_users_dict)}, ndcg@{TOP_K} : {sum_ndcg/len(eval_users_dict)}')

# Dependencies
import os
import argparse

from data import load_dataset, load_dataset_session
from envs import OfflineEnv
from recommender import DRRAgent
"""
[Training 방식]
- 매 에피소드마다 user를 랜덤하게 선택 (OfflineEnv.user)
- user의 최근에 본 영화 10개를 이용해 state 생성
- user가 최근에 본 영화 10개를 제외한 나머지 영화들을 추천 (DRRAgent.recommend_item())
- 한 user당 trajectory 최대 길이는 약 3000, 그 전에 user가 본 영화 history 길이만큼 추천받으면 종료(OfflineEnv.step()) 
- Actor, Critic 파라미터 업데이트는 replay buffer에서 32개씩 batch로 묶어서 진행
"""

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')
STATE_SIZE = 10

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, help='Modality')
    parser.add_argument('--fusion', type=str, help='Fusion')
    parser.add_argument('--aggregation', type=str, help='Aggregation')
    parser.add_argument('--max_episode_num', type=int, default=8000, help='Number of episodes')
    parser.add_argument('--top_k', type=int, default=5, help='Top k items to recommend')
    args = parser.parse_args()
    if args.modality:
        args.modality = tuple(args.modality.split(','))
    # Loading dataset
    # total_users_num, total_items_num, train_users_dict, users_history_lens, movies_id_to_movies = load_dataset(DATA_DIR, 'train')
    total_users_num, total_items_num, train_users_dict, users_history_lens, movies_id_to_movies = load_dataset_session(DATA_DIR, 'train')

    env = OfflineEnv(train_users_dict, users_history_lens, movies_id_to_movies, STATE_SIZE)
    print(f"Available number of users: {len(env.available_users)}")
  
    recommender = DRRAgent(env, total_users_num, total_items_num, STATE_SIZE, args, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(args.max_episode_num, load_model=False, top_k=args.top_k)

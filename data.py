import os
import numpy as np
import pandas as pd

def load_dataset(DATA_DIR: str, mode: str="train"):
    assert mode in ['train', 'test'], "mode should be either 'train' or 'test'"

    print('Interact Data loading...')

    ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings_converted.csv'), sep=',', dtype=np.uint32)
    ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    movies_df = pd.read_csv(os.path.join(DATA_DIR, 'items.csv'), dtype=int, header=None)
    movies_df.columns = ['MovieID']
    
    print("Data loading complete!")
    print("Data preprocessing...")

    # 영화 id를 영화 제목으로
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_df.values}
    ratings_df = ratings_df.applymap(int)

    # 유저별로 본 영화들 순서대로 정리
    users_dict = np.load(DATA_DIR + 'final_user_dict.npy', allow_pickle=True)

    # 각 유저별 영화 히스토리 길이
    users_history_lens = np.load(DATA_DIR + 'final_users_history_len.npy')
    total_users_num = max(ratings_df["UserID"])+1
    total_items_num = max(ratings_df["MovieID"])+1
    print(f"total users_num : {total_users_num}, total items_num : {total_items_num}")

    train_users_num = int(total_users_num * 0.8)

    if mode == 'train':
        users_num = train_users_num
        users_dict = {k: users_dict.item().get(k) for k in range(users_num)}
        # users_history_lens = users_history_lens[:users_num]
        print(f"train_users_num : {users_num}")
    else:
        users_num = total_users_num - train_users_num 
        users_dict = {k: users_dict.item().get(k) for k in range(train_users_num, total_users_num)}
        # users_history_lens = users_history_lens[-users_num:]
        print(f"eval_users_num : {users_num}")

    print("Done")

    return users_num, total_items_num, users_dict, users_history_lens, movies_id_to_movies 

if __name__ == "__main__":
    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')
    eval_users_num, _, test_users_dict, test_users_history_lens, _ = load_dataset(DATA_DIR, 'test')

    print(len(test_users_history_lens))
    print(test_users_history_lens[:5])
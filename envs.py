import numpy as np

"""
[Training 방식]
- 매 에피소드마다 user를 랜덤하게 선택 (OfflineEnv.user)
- user의 최근에 본 영화 10개를 이용해 state 생성
- user가 최근에 본 영화 10개를 제외한 나머지 영화들을 추천 (DRRAgent.recommend_item())
- 한 user당 trajectory 최대 길이는 약 3000, 그 전에 user가 본 영화 history 길이만큼 추천받으면 종료(OfflineEnv.step()) 
- Actor, Critic 파라미터 업데이트는 replay buffer에서 32개씩 batch로 묶어서 진행

[수정해볼만한 것들]
1. 4832 user에서 중복허용해서 random으로 8000명 뽑는 것 같음 -> 중복 허용하지 않고 1번 혹은 2번 씩 학습
2. user가 최근에 본 영화 10개를 이용해 state를 만드는 것 같은데, 개수를 수정하거나 최근 것에 더 가중치를 두도록 수정
3. top_k로 수정
4. history가 10개 보다 적은 user도 같이 학습. state 만드는 부분에 대한 수정이 필요해보임
5. 언제나 item을 추천하기보다 아무것도 추천안하는 경우도 만들기? reward -0.5만. rating 1, 2 짜리 추천해주면 reward 더 떨어지니.
"""

class OfflineEnv(object):
    
    def __init__(self, users_dict, users_history_lens, movies_id_to_movies, state_size, fix_user_id=None):

        self.users_dict = users_dict                    # user_id : [(movie_id, rating), ...] -> 총 4832 user, id는 1부터 시작, 앞에 있는 movie가 최근에 본 movie
        self.users_history_lens = users_history_lens    # user_id 별 positive로 체크한 history 길이
        self.items_id_to_name = movies_id_to_movies     # movie_id : (title, genre)
        
        self.state_size = state_size                    # state size = 10 : 최근에 본 movie의 개수 기준 설정 
        self.available_users = self._generate_available_users() # 현재 추천가능한 user list 추리기

        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)      # 이번 에피소드에서 movie를 추천할 user
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}              # 해당 user가 본 {movie: rating}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]         # 해당 user의 history 중 최근 10개의 movie의 id
        self.done = False
        self.recommended_items = set(self.items)        # 추천된 movie들의 id set
        self.done_count = 3000                          # trajectory의 최대 길이
        
    def _generate_available_users(self):
        """
        self.state_size (10) 보다 긴 history를 가진 user 들만 available_users에 추가
        episode가 진행될 수록 available user가 감소하는건 아닌건가? 그러면 불필요하게 한 user에 대해서 여러번 훈련될 수도 있나? 
        """
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users
    
    def reset(self):
        """
        init 반복
        """
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)    # training: random 선택, validation : fixed 선택
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}                      # 해당 user가 본 {movie: rating}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]                 # 해당 user의 history 중 최근 10개의 movie의 id
        self.done = False
        self.recommended_items = set(self.items)                                                        # 추천된 movie들의 id set
        return self.user, self.items, self.done
        
    def step(self, action, top_k=False):
        
        reward = -0.5       # time step마다 -0.5 reward

        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                # 추천한 movie가 user가 봤던 history에 있고, 최근 본 10개 중에 없다면 올바른 추천으로 취급
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rating = self.user_items[act]
                    rewards.append((rating-3)/2)    # rating 1, 2, 3, 4, 5를 [-1, 1]로 정규화
                else:
                    # 추천한 movie가 user가 봤던 history에 없거나, 최근 본 10개 중에 있다면
                    rewards.append(-0.5)
                
                # 과거 추천받은 movie list에 현재 추천받은 movie 추가 
                self.recommended_items.add(act)
            
            # reward를 받았으면, 잘 추천된 movie들을 history에 추가
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended

            reward = rewards

        else:
            # 추천한 movie가 user가 봤던 history에 있고, 최근 본 10개 중에 없다면.
            if action in self.user_items.keys() and action not in self.recommended_items:
                reward = self.user_items[action] -3  # reward : rating이 1~5까지니까 4, 5인 경우에 + reward를 받게 됨
            
            # reward가 0 보다 크다면, 추천한 movie를 history에 추가 -> 이건 현재 episode에서만 반영됨 (user가 중복 학습 안된다고 하면 문제 없음)
            # self.items: 지금은 10개가 순서 무관하게 동일하게 취급되지만, 최근꺼를 더 반영하는 식으로 수정가능할 듯
            if reward > 0:
                self.items = self.items[1:] + [action]
            
            # 추천받은 movie에 action 추가
            self.recommended_items.add(action)
        
        # 추천받은 item개수가 done_count보다 크거나, 추천받은 item 개수가 user의 history 길이보다 크거나 같다면
        # self.user -1 인 이유는 user id가 1부터 시작해서
        # 추천받은 item 개수가 user의 history 길이보다 크거나 같은지는 왜 체크하지? -> 같아지면 더 이상 추천할 영화 없음
        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[self.user]:
            self.done = True
            
        # self.items : 이게 state가 만약 적절한 추천을 해줬다면 다음 self.items에 추가되었을 것이고 state만드는데 사용됨
        return self.items, reward, self.done, self.recommended_items

    # 여기서 쓰이는 곳은 없음 -> Evaluation에서 log 출력에 사용됨
    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names

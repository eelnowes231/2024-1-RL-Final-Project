import os
import argparse
import numpy as np
import tensorflow as tf

def get_user_posItem(users_dict):
    # Args:
    #   users_dict : {user_id: [(item_id, rating), ...]}
    # Returns:
    #   user_posItem_pairs : ndarray([(user_id, item_id), ...])
    #   user_posItems_dict : {user_id: [item_id, ...], ...}
    
    user_posItem_pairs = []
    user_posItems_dict = {u: [] for u in users_dict.keys()}
    
    for user_id, item_rating_list in users_dict.items():
        for item_id, rating in item_rating_list:
            # if rating >= 1:
            user_posItem_pairs.append((user_id, item_id))
            user_posItems_dict[user_id].append(item_id)
                
    user_posItem_pairs = np.array(user_posItem_pairs)           
    return user_posItem_pairs, user_posItems_dict


def get_dataloader(user_posItem_pairs, user_posItems_dict, batch_size, negative_ratio=0.5):
    # Args:
    #   user_posItem_pairs : ndarray([(user_id, item_id), ...])
    #   user_posItems_dict : {user_id: [item_id, ...], ...}
    #   negative_ratio : float
    # Returns:
    #   generator : (user_id_batch, item_id_batch, label_batch)

    batch = np.zeros((batch_size, 3))
    
    positive_batch_size = batch_size - int(batch_size*negative_ratio)
    negative_batch_size = batch_size - positive_batch_size
    
    num_user = 6039
    num_item = 2819
    
    while True:
        idx = np.random.choice(len(user_posItem_pairs), positive_batch_size)
        pos_sample = user_posItem_pairs[idx]
        batch[:positive_batch_size,:2] = pos_sample
        batch[:positive_batch_size, 2] = 1
        
        neg_user = np.random.randint(num_user, size=negative_batch_size)
        batch[positive_batch_size:, 0] = neg_user
        for i, user in enumerate(neg_user):
            neg_item = np.random.randint(num_item)
            while neg_item in user_posItems_dict[user]:
                neg_item = np.random.randint(num_item)
            batch[positive_batch_size+i, 1] = neg_item
            batch[positive_batch_size+i, 2] = 0
            
        np.random.shuffle(batch)
        yield batch[:,0].astype(int), batch[:,1].astype(int), batch[:,2]
        
        
class UserMovieEmbedding(tf.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim, modality=('video', 'audio', 'text'), fusion='early', aggregation='concat'):
        super(UserMovieEmbedding, self).__init__()
        self.modality = modality
        self.fusion = fusion
        self.aggregation = aggregation
        
        # input: (user, movie)
        self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        
        # user embedding
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
        
        # item embedding        
        if not modality:
            self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        else:
            # load multimodal features
            for mod in modality:
                mod_name = 'image' if mod == 'video' else mod # rename due to file name
                setattr(self, f'{mod}_feat', np.load(f'{DATA_DIR}/{mod_name}_feat.npy'))
                
            if fusion == 'early':
                self.mm_fc = tf.keras.layers.Dense(embedding_dim, name='mm_fc')
                
            elif fusion == 'late':
                if aggregation == 'concat':
                    def divide_integer(n, parts):
                        q, r = divmod(n, parts)
                        return [q+1]*(r) + [q]*(parts-r)
                    embedding_dims = divide_integer(embedding_dim, len(modality))
                elif aggregation == 'mean':
                    embedding_dims = [embedding_dim]*len(modality)
                    
                for i, mod in enumerate(modality):
                    setattr(self, f'{mod}_fc', tf.keras.layers.Dense(embedding_dims[i], name=f'{mod}_fc'))
        
        # dot product
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def get_embedding(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        
        if not self.modality:
            memb = self.m_embedding(x[1])
        else:
            mm_emb = []
            for mod in self.modality:
                mm_feat = getattr(self, f'{mod}_feat')
                mm_feat = tf.gather(mm_feat, x[1])
                
                if self.fusion == 'early':
                    mm_emb.append(mm_feat)
                elif self.fusion == 'late':
                    mm_emb.append(getattr(self, f'{mod}_fc')(mm_feat))
            
            if self.aggregation == 'concat':
                memb = tf.concat(mm_emb, axis=1)
            elif self.aggregation == 'mean':
                memb = tf.reduce_mean(tf.stack(mm_emb), axis=0)
                
            if self.fusion == 'early':
                memb = self.mm_fc(memb)
        return uemb, memb
        
    def call(self, x):
        uemb, memb = self.get_embedding(x)
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pretrain embedding')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--modality', type=str, help='Modality')
    parser.add_argument('--fusion', type=str, help='Fusion')
    parser.add_argument('--aggregation', type=str, help='Aggregation')
    args = parser.parse_args()
    if args.modality:
        args.modality = tuple(args.modality.split(','))
    
    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')
    
    users_dict = np.load(DATA_DIR + '/user_dict_new.npy', allow_pickle=True).item()
    
    user_ids = set(users_dict.keys())
    item_ids = set()
    for value in users_dict.values():
        for item_id, _ in value:
            item_ids.add(item_id)
    num_user, num_item = len(user_ids), len(item_ids)
    
    u_m_pairs, u_m_dict = get_user_posItem(users_dict)
    
    u_m_model = UserMovieEmbedding(num_user, num_item, args.embed_dim,
                                   modality=args.modality,
                                   fusion=args.fusion,
                                   aggregation=args.aggregation)
    
    optimizer = tf.keras.optimizers.Adam()
    bce = tf.keras.losses.BinaryCrossentropy()
    
    u_m_train_loss = tf.keras.metrics.Mean(name='train_loss')
    u_m_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    @tf.function
    def u_m_train_step(u_m_inputs, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = u_m_model(u_m_inputs, training=True)
            loss = bce(labels, predictions)
        gradients = tape.gradient(loss, u_m_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, u_m_model.trainable_variables))

        u_m_train_loss(loss)
        u_m_train_accuracy(labels, predictions)
        
    # u_m_losses = []
    for epoch in range(args.epoch):
        u_m_generator = get_dataloader(u_m_pairs, u_m_dict, args.batch_size, 0.5)
        for step in range(len(u_m_pairs)//args.batch_size):
            # embedding layer update
            u_batch, m_batch, u_m_label_batch = next(u_m_generator)
            u_m_train_step([u_batch, m_batch], u_m_label_batch)
            
            print(f'{epoch} epoch, Batch size : {args.batch_size}, {step} steps, Loss: {u_m_train_loss.result():0.4f}, Accuracy: {u_m_train_accuracy.result() * 100:0.1f}', end='\r')

        # u_m_losses.append(u_m_train_loss.result())

    if args.modality:
        mod_name = ''.join([mod[0] for mod in args.modality]).upper()
        weights_name = f'{mod_name}_{args.fusion}_{args.aggregation}'
    else:
        weights_name = f'ID'
    u_m_model.save_weights(f'{ROOT_DIR}/save_weights/u_m_model_{weights_name}.h5')
    
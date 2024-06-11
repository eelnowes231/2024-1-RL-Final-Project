import tensorflow as tf
import numpy as np
import os

class MovieGenreEmbedding(tf.keras.Model):
    def __init__(self, len_movies, len_genres, embedding_dim):
        super(MovieGenreEmbedding, self).__init__()
        self.m_g_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        self.g_embedding = tf.keras.layers.Embedding(name='genre_embedding', input_dim=len_genres, output_dim=embedding_dim)
        # dot product
        self.m_g_merge = tf.keras.layers.Dot(name='movie_genre_dot', normalize=True, axes=1)
        # output
        self.m_g_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        x = self.m_g_input(x)
        memb = self.m_embedding(x[0])
        gemb = self.g_embedding(x[1])
        m_g = self.m_g_merge([memb, gemb])
        return self.m_g_fc(m_g)

# class UserMovieEmbedding(tf.keras.Model):
#     def __init__(self, len_users, embedding_dim):
#         super(UserMovieEmbedding, self).__init__()
#         self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
#         # embedding
#         self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
#         # dot product
#         self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
#         # output
#         self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
#     def call(self, x):
#         x = self.m_u_input(x)
#         uemb = self.u_embedding(x[0])
#         m_u = self.m_u_merge([x[1], uemb])
#         return self.m_u_fc(m_u)


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
                ROOT_DIR = os.getcwd()
                DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')
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
    
# class UserMovieEmbedding(tf.keras.Model):
#     def __init__(self, len_users, len_movies, embedding_dim):
#         super(UserMovieEmbedding, self).__init__()
#         self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
#         # embedding
#         self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
#         self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
#         # dot product
#         self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
#         # output
#         self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
#     def call(self, x):
#         x = self.m_u_input(x)
#         uemb = self.u_embedding(x[0])
#         memb = self.m_embedding(x[1])
#         m_u = self.m_u_merge([memb, uemb])
#         return self.m_u_fc(m_u)


# class UserMovieEmbedding(tf.keras.Model):
#     def __init__(self, len_users, len_movies, embedding_dim):
#         super(UserMovieEmbedding, self).__init__()
#         self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
#         # embedding
#         self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
#         self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
#         # dot product
#         self.m_u_concat = tf.keras.layers.Concatenate(name='movie_user_concat', axis=1)
#         # output
#         self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
#     def call(self, x):
#         x = self.m_u_input(x)
#         uemb = self.u_embedding(x[0])
#         memb = self.m_embedding(x[1])
#         m_u = self.m_u_concat([memb, uemb])
#         return self.m_u_fc(m_u)
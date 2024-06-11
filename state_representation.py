import tensorflow as tf
import numpy as np

class DRRAveStateRepresentation(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):
        items_eb = tf.transpose(x[1], perm=(0, 2, 1)) 
        # items_eb shape is (1, 64, 10)

        # Define the weights manually
        weights = tf.constant([0.19, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.01], dtype=tf.float32)
        weights = tf.cast(weights, items_eb.dtype)  # Cast to the same dtype as items_eb

        # Expand dimensions of weights to match items_eb for multiplication
        weights = tf.reshape(weights, (1, 1, 10))  # Shape (1, 1, 10)
        
        # Perform weighted sum along the last axis
        weighted_items_eb = items_eb * weights  # Broadcasting weights to match items_eb
        wav = tf.reduce_sum(weighted_items_eb, axis=2)  # Sum along the last axis to get shape (1, 64)
        
        user_wav = tf.keras.layers.multiply([x[0], wav])
        concat = self.concat([x[0], user_wav, wav])
        return self.flatten(concat)
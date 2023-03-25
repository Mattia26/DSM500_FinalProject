# RecommenderNet from 

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class RecommenderNet(keras.Model):
        def __init__(self, num_users, num_items, embedding_size, dot=True, **kwargs):
            """
             Initialize the class. This is the method that is called by the constructor. You can override this method in your subclass if you want to do something other than setting the dot attribute to False.
             The inspiration for this model is taken from https://keras.io/examples/structured_data/collaborative_filtering_movielens/

             @param num_users - Number of users.
             @param num_items - Number of items
             @param embedding_size - Embedding size for the users and items.
             @param dot - Whether or not to use  dot product for prediction
            """
            super().__init__(**kwargs)
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_size = embedding_size
            self.dot = dot
            self.user_embedding = layers.Embedding(
                num_users,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.user_bias = layers.Embedding(num_users, 1)
            self.item_embedding = layers.Embedding(
                num_items,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.item_bias = layers.Embedding(num_items, 1)
            
            # Predictor piece of the neural model
            self.concatenator1 = layers.Concatenate(axis =1)
            self.concatenator2 = layers.Dense(128, activation='relu')
            self.concatenator3 = layers.Dense(1, activation='relu')
        
        def call(self, inputs):
            """
             Applies sigmoid to embeddings. It is assumed that the user has been preprocessed beforehand.
             
             @param inputs - A 2 - D tensor of shape [ batch_size num_items ].
             
             @return A 2 - D tensor of shape [ batch_size num_items ] with the first dimension indexing items and second indexing user
            """
            user_vector = self.user_embedding(inputs[:, 0])
            user_bias = self.user_bias(inputs[:, 0])
            item_vector = self.item_embedding(inputs[:, 1])
            item_bias = self.item_bias(inputs[:, 1])

            #simple dot product
            if self.dot:
                user_item = tf.tensordot(user_vector, item_vector, 2)
                x = user_item + user_bias + item_bias
                return tf.nn.sigmoid(x)
            #neural network
            else: 
                 user_item = self.concatenator1([user_vector,item_vector])
                 user_item = self.concatenator2(user_item)
                 user_item = self.concatenator3(user_item)
                 x = user_item + user_bias + item_bias
                 return tf.nn.sigmoid(x)
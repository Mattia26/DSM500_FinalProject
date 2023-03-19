# RecommenderNet from https://keras.io/examples/structured_data/collaborative_filtering_movielens/

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class RecommenderNet(keras.Model):
        def __init__(self, num_users, num_movies, embedding_size, y_train, dot=True, **kwargs):
            super().__init__(**kwargs)
            self.y_train = y_train
            self.num_users = num_users
            self.num_movies = num_movies
            self.embedding_size = embedding_size
            self.dot = dot
            self.user_embedding = layers.Embedding(
                num_users,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.user_bias = layers.Embedding(num_users, 1)
            self.movie_embedding = layers.Embedding(
                num_movies,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.movie_bias = layers.Embedding(num_movies, 1)
            self.concatenator1 = layers.Concatenate(axis =1)
            self.concatenator2 = layers.Dense(128, activation='relu')
            self.concatenator3 = layers.Dense(1, activation='relu')
        
        def call(self, inputs):
            user_vector = self.user_embedding(inputs[:, 0])
            user_bias = self.user_bias(inputs[:, 0])
            movie_vector = self.movie_embedding(inputs[:, 1])
            movie_bias = self.movie_bias(inputs[:, 1])
            if self.dot:
                user_movie = tf.tensordot(user_vector, movie_vector, 2)
                x = user_movie + user_bias + movie_bias
                return tf.nn.sigmoid(x)
            else: 
                 user_movie = self.concatenator1([user_vector,movie_vector])
                 #shape = user_movie.shape
                 #user_movie = layers.Reshape((512,12))(user_movie)
                 user_movie = self.concatenator2(user_movie)
                 user_movie = self.concatenator3(user_movie)
                 x = user_movie + user_bias + movie_bias
                 return tf.nn.sigmoid(x)
            
            #output = self.output
            # Add all the components (including bias)
            
            #user_np = user_vector.numpy()
            #movie_np = movie_vector.numpy()
            #concat_values = list(zip(user_np, movie_np))
            #concat_values = np.asarray(concat_values)
            #num_samples, nx, ny = concat_values.shape
            #concat_values = concat_values.reshape(num_samples, nx*ny)
            #predictor = RandomForestRegressor(max_depth=2, random_state = 42)
            #predictor.fit(concat_values,self.y_train)
            #x = predictor.predict(concat_values)
            # The sigmoid activation forces the rating to between 0 and 1
            
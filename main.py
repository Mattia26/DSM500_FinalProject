import numpy as np
from zipfile import ZipFile
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from embedder.RecommenderNet import RecommenderNet
from preprocessor.MovieLensProcessor import MovieLensProcessor
from evaluator.Evaluator import Evaluator
from sklearn.ensemble import RandomForestRegressor

EMBEDDING_SIZE = 6



if __name__ == "__main__":

    processor = MovieLensProcessor("Dataset/movielens_100k/ratings.csv", 0.9)
    
    x_train, x_val, x_hold, y_train, y_val, y_hold, num_movies, num_users = processor.getData()
    #y = RecommenderNet(30000,30000,5, [1,2,3,4,5,6,7,8,9,10])(x_val[:10])

    if True :
        model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE, y_train)
        
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            run_eagerly=True
        ) 

        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=512,
            epochs=1,
            verbose=1,
            validation_data=(x_val, y_val)
        )

        model.save("embedder/saved")
    if False:
        model = keras.models.load_model('embedder/saved')
    

    eval = Evaluator(x_hold, y_hold, model, 2, processor.userencoded2user)
    print(eval.evaluate())
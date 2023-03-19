from tensorflow import keras
import tensorflow as tf
import numpy as np
from embedder.RecommenderNet import RecommenderNet
from preprocessor.MovieLensProcessor import MovieLensProcessor
from evaluator.Evaluator import Evaluator
from predictor.Predictor import Predictor
from predictor.KNeighborsPredictor import KNeighborsPredictor
import datetime
import time
EMBEDDING_SIZE = 50



if __name__ == "__main__":

    processor = MovieLensProcessor("Dataset/amazon_books_100k/amazon_books.csv", 0.9)
    
    x_train, x_val, x_hold, y_train, y_val, y_hold, num_movies, num_users = processor.getData()
    #y = RecommenderNet(30000,30000,5, [1,2,3,4,5,6,7,8,9,10])(x_val[:10])

    #h_val = np.random.rand(len(y_val))
    #eval = Evaluator(x_hold, h_val, y_hold, 20)
    #print(eval.evaluate())
    if False :
        model_dot = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE, y_train, dot=True)
        
        model_dot.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            run_eagerly=True
        ) 
        history = model_dot.fit(
            x=x_train,
            y=y_train,
            batch_size=128,
            epochs=5,
            verbose=1,
            validation_data=(x_val, y_val)
        )

        model_dot.save("embedder/saved_dot")

        model_nn = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE, y_train, dot=False)
        
        model_nn.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            run_eagerly=True
        ) 
        history = model_nn.fit(
            x=x_train,
            y=y_train,
            batch_size=128,
            epochs=5,
            verbose=1,
            validation_data=(x_val, y_val)
        )

        model_nn.save("embedder/saved_nn")
    if True:
        model_dot = keras.models.load_model('embedder/saved_dot')
        model_nn = keras.models.load_model('embedder/saved_nn')
    def rand_pred(x):
        return np.random.rand(len(x))
    with open('results', 'a') as r:
        r.write('\n\n')
        r.write(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H:%M:%S - '))
        r.write('\n')

        predict = rand_pred
        eval = Evaluator(x_hold, y_hold, 20, predict)
        r.write('--- random ---')
        r.write(str(eval.evaluate()))
        r.write('\n')
        print(str(eval.evaluate()))

        predict = model_dot.predict
        eval = Evaluator(x_hold, y_hold, 20, predict)
        r.write('--- dot product ---')
        r.write(str(eval.evaluate()))
        r.write('\n')

        predict = model_nn.predict
        eval = Evaluator(x_hold, y_hold, 20, predict)
        r.write('--- nn ---')
        r.write(str(eval.evaluate()))
        r.write('\n')

        predictor = Predictor(model_dot, 'RANDOM_FOREST')
        predictor.fit(x_train, y_train)
        predict = predictor.predict

        eval = Evaluator(x_hold, y_hold, 20, predict)
        r.write('--- random forest - dot emb ---')
        r.write(str(eval.evaluate()))
        r.write('\n')

        predictor = Predictor(model_dot, 'KNEIGH')
        predictor.fit(x_train, y_train)
        predict = predictor.predict

        eval = Evaluator(x_hold, y_hold, 20, predict)
        r.write('--- kneigh - dot emb ---')
        r.write(str(eval.evaluate()))
        r.write('\n')

        predictor = Predictor(model_nn, 'RANDOM_FOREST')
        predictor.fit(x_train, y_train)
        predict = predictor.predict

        eval = Evaluator(x_hold, y_hold, 20, predict)
        r.write('--- random forest - net emb ---')
        r.write(str(eval.evaluate()))
        r.write('\n')

        predictor = Predictor(model_nn, 'KNEIGH')
        predictor.fit(x_train, y_train)
        predict = predictor.predict

        eval = Evaluator(x_hold, y_hold, 20, predict)
        r.write('--- kneigh - net emb ---')
        r.write(str(eval.evaluate()))
        r.write('\n')


 

    

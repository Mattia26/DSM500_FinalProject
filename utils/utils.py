import tensorflow as tf
from tensorflow import keras
from embedder.RecommenderNet import RecommenderNet
import numpy as np

def rand_pred(x):
    """
     Generate a random prediction. This is a function to be used in conjunction with : func : ` ~chainer. Variable. predict `
     
     @param x - The input variable to predict.
     
     @return A random prediction of the input variable ` x `. See Also : func : ` ~chainer. Variable. predict
    """
    return np.random.rand(len(x))

def array2csv(values):
    """
     Converts an array of values to a comma separated string. This is useful for CSV files that are read from disk
     
     @param values - The array of values to convert
     
     @return A comma separated string of the values in the same order as the array passed in with newlines between each
    """
    result = ""
    # Returns a comma separated list of values
    for v in values:
        result += str(v) + ","
    
    result += '\n'

    return result

def get_trained_models(num_users, num_items, x_train, y_train, x_val, y_val, embedding_size, train, dataset):
    """
     Creates and trains neural networks. This is a helper function to get the training and validation data for recommender.
     
     @param num_users - Number of users in the recommender.
     @param num_items - Number of items in the recommender.
     @param x_train - Training data for user training. It is a 2D tensor of shape [ num_users num_items ] where each row is a user
     @param y_train - Training data for user training. It is a 2D tensor of shape [ num_users num_items ] where each row is a user
     @param x_val
     @param y_val
     @param embedding_size
     @param train
     @param dataset
    """
    
    # If needed train models from scratch
    if train:
        model_dot = RecommenderNet(num_users, num_items, embedding_size, dot=True)
        model_nn = RecommenderNet(num_users, num_items, embedding_size, dot=False)

        model_dot.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            run_eagerly=True
        ) 

        model_nn.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            run_eagerly=True
        )
        print("Fitting neural networks")
        model_nn.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=128,
                    epochs=5,
                    verbose=1,
                    validation_data=(x_val, y_val)
                )
        
        model_dot.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=128,
                    epochs=5,
                    verbose=1,
                    validation_data=(x_val, y_val)
                )
        
        
        model_dot.save('embedder/saved/' + dataset + '/model_dot/')
        model_nn.save('embedder/saved/' + dataset + '/model_nn/')
        return model_dot,model_nn
    else:
        model_dot = keras.models.load_model('embedder/saved/' + dataset + '/model_dot/')
        model_nn = keras.models.load_model('embedder/saved/' + dataset + '/model_nn/')

        return model_dot, model_nn
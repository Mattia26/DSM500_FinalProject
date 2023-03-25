import pandas as pd
import numpy as np

class Processor():
    def __init__(self, path, training_split):
        """
         Initialize the class. This is called by __init__ and should not be called directly. The base class does nothing.
         
         @param path - Path to the data file. It is assumed that this is a file that contains the dataset
         @param training_split - The training split
        """
        self.path = path
        self.training_split = training_split
    
    def getData(self):
        """
         Reads data from csv file and converts it to numpy arrays. This is a helper to get data from file
         Code is a modified version of the pre-processing explained in https://keras.io/examples/structured_data/collaborative_filtering_movielens/
         
         @return training, test and holdout sets
        """
        df = pd.read_csv(self.path)

        #encoding the values to work with integers only
        user_ids = df[df.columns[0]].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        self.userencoded2user = {i: x for i, x in enumerate(user_ids)}
        movie_ids = df[df.columns[1]].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
        df[df.columns[0]] = df[df.columns[0]].map(user2user_encoded)
        df[df.columns[1]] = df[df.columns[1]].map(movie2movie_encoded)

        num_users = len(user2user_encoded)
        num_movies = len(movie_encoded2movie)
        df[df.columns[2]] = df[df.columns[2]].values.astype(np.float32)
        
        # min and max ratings will be used to normalize the ratings later
        min_rating = min(df[df.columns[2]])
        max_rating = max(df[df.columns[2]])

        print(
            "Number of users: {}, Number of Items: {}, Min rating: {}, Max rating: {}".format(
                num_users, num_movies, min_rating, max_rating
            )
        )

        df = df.sample(frac=1, random_state=42)
        x = df[[df.columns[0], df.columns[1]]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = df[df.columns[2]].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
        # test set is split in half: one half for the test data for training and half
        # for the holdout set for the final validation.
        train_indices = int((self.training_split-0.1) * df.shape[0])
        holdout_indices = int(0.9*df.shape[0])

        return (
            x[:train_indices],
            x[train_indices:holdout_indices],
            x[holdout_indices:],
            y[:train_indices],
            y[train_indices:holdout_indices],
            y[holdout_indices:],
            num_movies,
            num_users
        )
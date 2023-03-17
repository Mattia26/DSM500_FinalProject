import pandas as pd
import numpy as np

class MovieLensProcessor():
    def __init__(self, path, training_split):
        self.path = path
        self.training_split = training_split
    
    def getData(self):
        df = pd.read_csv(self.path)
        user_ids = df["userId"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        self.userencoded2user = {i: x for i, x in enumerate(user_ids)}
        movie_ids = df["movieId"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
        df["user"] = df["userId"].map(user2user_encoded)
        df["movie"] = df["movieId"].map(movie2movie_encoded)

        num_users = len(user2user_encoded)
        num_movies = len(movie_encoded2movie)
        df["rating"] = df["rating"].values.astype(np.float32)
        # min and max ratings will be used to normalize the ratings later
        min_rating = min(df["rating"])
        max_rating = max(df["rating"])

        print(
            "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
                num_users, num_movies, min_rating, max_rating
            )
        )

        df = df.sample(frac=1, random_state=42)
        x = df[["user", "movie"]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
        # Assuming training on 90% of the data and validating on 10%.
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
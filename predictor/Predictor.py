from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

class Predictor():
    def __init__(self, embedder, predictor):
        self.embedder = embedder
        match predictor:
            case 'RANDOM_FOREST':
                self.predictor = RandomForestRegressor(max_depth=2, random_state=0)
            case 'KNEIGH':
                self.predictor = KNeighborsRegressor(n_neighbors=5)
    
    def fit(self, x_train, y_train):
        x_train_emb = self.get_embedded(x_train)
        self.predictor.fit(x_train_emb, y_train)       

    def predict(self, x_test):
        x_test_emb = self.get_embedded(x_test)
        return self.predictor.predict(x_test_emb)
    
    def get_embedded(self, values):
        f1_emb = self.embedder.user_embedding(values[:,0])
        f2_emb = self.embedder.movie_embedding(values[:,1])
        concat_values = list(zip(f1_emb, f2_emb))
        concat_values = np.asarray(concat_values)
        num_samples, nx, ny = concat_values.shape
        concat_values = concat_values.reshape(num_samples, nx*ny)

        return concat_values
    
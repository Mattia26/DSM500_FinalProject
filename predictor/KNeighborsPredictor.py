from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class KNeighborsPredictor():
    def __init__(self, embedder, x_train, y_train, x_test):
        self.embedder = embedder 
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.predictor = KNeighborsRegressor(n_neighbors=5)
    
    def fit_predict(self):

        x_train_emb = self.get_embedded(self.x_train)
        x_test_emb = self.get_embedded(self.x_test)
        self.predictor.fit(x_train_emb, self.y_train)

        return self.predictor.predict(x_test_emb)

    
    def get_embedded(self, values):
        f1_emb = self.embedder.user_embedding(values[:,0])
        f2_emb = self.embedder.movie_embedding(values[:,1])
        concat_values = list(zip(f1_emb, f2_emb))
        concat_values = np.asarray(concat_values)
        num_samples, nx, ny = concat_values.shape
        concat_values = concat_values.reshape(num_samples, nx*ny)

        return concat_values
    
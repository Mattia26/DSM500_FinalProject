from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

class Predictor():
    def __init__(self, embedder, predictor):
        """
         Initialize the embedder. This is called by __init__ and should not be called directly. You should call this method instead
         
         @param embedder - The class that embeds the data
         @param predictor - The predictor to use
        """
        self.embedder = embedder
        match predictor:
            case 'RANDOM_FOREST':
                self.predictor = RandomForestRegressor(max_depth=2, random_state=0)
            case 'KNEIGH':
                self.predictor = KNeighborsRegressor(n_neighbors=5)
    
    def fit(self, x_train, y_train):
        """
         Fit the model to the training data. This is the first step in training the model. It calls the predictor's fit method
         
         @param x_train - The data used to train the model
         @param y_train - The target of the training data ( y
        """
        print("Fitting {} model".format(self.predictor))
        x_train_emb = self.get_embedded(x_train)
        self.predictor.fit(x_train_emb, y_train)       

    def predict(self, x_test):
        """
         Predict with the predictor. This is a wrapper around the : py : meth : ` ~gensim. models. EMBInference. predict ` method
         
         @param x_test - The data to use for prediction
         
         @return The prediction of the data in the form of a numpy array of shape ( n_samples n_features
        """
        print("Predicting with {}".format(self.predictor))
        x_test_emb = self.get_embedded(x_test)
        return self.predictor.predict(x_test_emb)
    
    def get_embedded(self, values):
        """
         Get embeddings for user and movie. This is a helper function for : meth : ` get_data `.
         
         @param values - Numpy array with shape ( num_samples nx ny )
         
         @return Numpy array with shape ( num_samples nx * ny
        """
        f1_emb = self.embedder.user_embedding(values[:,0])
        f2_emb = self.embedder.item_embedding(values[:,1])
        concat_values = list(zip(f1_emb, f2_emb))
        concat_values = np.asarray(concat_values)
        num_samples, nx, ny = concat_values.shape
        concat_values = concat_values.reshape(num_samples, nx*ny)

        return concat_values
    
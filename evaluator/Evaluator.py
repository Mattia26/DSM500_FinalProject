from math import log
import numpy as np
from collections import Counter
from tqdm import tqdm

class Evaluator():

    def __init__(self, x_val, y_val, k, predict):
        """
         Initialize the class. This is called by __init__ and should not be called directly. The arguments are for convenience
         
         @param x_val - The x values for the predictors
         @param y_val - The y values for the predictors
         @param k - K for the accuracy@k and ndcg@k
         @param predict - the predictor to evaluate
        """
        self.y_val = y_val
        self.x_val = x_val
        self.k = k
        self.predict = predict

    def evaluate(self):
        """
         Evaluate items and return recall NDCG and Recall. This is a method to use when you want to perform evaluation on a list of users.
         
         
         @return a list of recall and NDCG values. Each value is a numpy array with shape [ num_users num_recall ]
        """
        user_list = set([x[0] for x in self.x_val])
        top_100_items = [x[0] for x in Counter(self.x_val[:,1]).most_common(100)]

        x_all = np.array([])
        # calculate the values to predict, for performance reasons only the top 50 items
        for u in tqdm(user_list):
            for m in top_100_items[0:100]:
                if len(x_all) == 0:
                    x_all = [u,m]
                else:
                    x_all = np.vstack((x_all, [u,m]))
        
        #predict the values
        h_val = self.predict(x_all)
        h_val = [x[0] for x in h_val] if len(h_val.shape) > 1 else h_val
       
        #initialize metrics arrays
        recall_list = []
        ndcg_list = []

        for u in tqdm(user_list):
            #Sorted top k recommendations for ground truth and predictions for user u
            u_a_indices = [i for i,x in enumerate(x_all) if x[0] == u]
            u_x_indices = [i for i,x in enumerate(self.x_val) if x[0] == u]
            u_h_val  = [h_val[i] for i in u_a_indices]
            u_y_val = [self.y_val[i] for i in u_x_indices]
            top_k_u_h_val = sorted(range(len(u_h_val)), key=lambda i:u_h_val[i], reverse=True)[:self.k]
            top_k_u_y_val = sorted(range(len(u_y_val)), key=lambda i:u_y_val[i], reverse = True)[:self.k]
            
            #calculate recall@k for user u
            recommended_items = [x_all[i][1] for i in top_k_u_h_val]
            truth_movies = [self.x_val[i][1] for i in top_k_u_y_val]
            u_recall_k = len(set(recommended_items) & set(truth_movies) )/ self.k
            
            #calculate ndcg@k for user u
            u_i_dcg = []
            for i,x in enumerate (top_k_u_y_val):
                try:
                    m = self.x_val[x][1]
                    i_x = np.where(x_all[:,1] == m)[0][0]
                    u_i_dcg.append((u_h_val[i_x])/log(i+2,2))
                except IndexError:
                    u_i_dcg.append(0)
            u_dcg = sum(u_i_dcg)
            u_idcg = sum([(u_y_val[x])/log(i+2,2) for i,x in enumerate(top_k_u_y_val)])
            u_ndcg_k = u_dcg / u_idcg if u_idcg != 0 else 0
            #add to user metrics list
            recall_list.append(u_recall_k)
            ndcg_list.append(u_ndcg_k)
        
        
        #return the average metrics of users
        return [(sum(recall_list) / len(recall_list)),(sum(ndcg_list) / len(ndcg_list))]


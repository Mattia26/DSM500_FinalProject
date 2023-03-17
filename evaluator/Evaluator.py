from math import log

class Evaluator():

    def __init__(self, x_val, y_val, model, k, userencoded2user):
        self.x_val = x_val
        self.y_val = y_val
        self.model = model
        self.k = k
        self.userencoded2user = userencoded2user

    def evaluate(self):
        #predicted values
        h_val = self.model.predict(self.x_val)
        h_val = [x[0] for x in h_val]
        #calculate metrics
        user_list = set([x[0] for x in self.x_val])
        recall_list = []
        ndcg_list = []

        for u in user_list:
            u_indices = [i for i,x in enumerate(self.x_val) if x[0] == u]
            u_h_val  = [h_val[i] for i in u_indices]
            u_y_val = [self.y_val[i] for i in u_indices]
            top_k_u_h_val = sorted(range(len(u_h_val)), key=lambda i:u_h_val[i], reverse=True)[:self.k]
            top_k_u_y_val = sorted(range(len(u_y_val)), key=lambda i:u_y_val[i], reverse = True)[:self.k]
            u_recall_k = len(set(top_k_u_h_val) & set(top_k_u_y_val) )/ self.k
            u_dcg = sum([(u_h_val[x])/log(i+2,2) for i,x in enumerate(top_k_u_y_val)])
            u_idcg = sum([(u_y_val[x])/log(i+2,2) for i,x in enumerate(top_k_u_y_val)])
            u_ndcg_k = u_dcg / u_idcg if u_idcg != 0 else 0
            recall_list.append(u_recall_k)
            ndcg_list.append(u_ndcg_k)
        
        

        return [(sum(recall_list) / len(recall_list)),(sum(ndcg_list) / len(ndcg_list))]


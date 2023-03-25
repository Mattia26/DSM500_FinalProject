from preprocessor.Processor import Processor
from evaluator.Evaluator import Evaluator
from predictor.Predictor import Predictor
from utils.utils import get_trained_models, array2csv, rand_pred
import datetime
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#embedding size for users and items
EMBEDDING_SIZE = 50
#k evaluation
K = 50
#"yelp":"Dataset/yelp.csv"
#list of dataset for experiments
DATASETS = {"Amazon":"Dataset/amazon_books.csv", "Movielens":"Dataset/movielens.csv", "yelp":"Dataset/yelp.csv"}
#set to true if you want to retrain the models
TRAIN = False


if __name__ == "__main__":

    for dataset in list(DATASETS.keys()):
        print("Evaluation on {} dataset".format(dataset) )
        processor = Processor(DATASETS.get(dataset), 0.9)
        
        #split current dataset
        x_train, x_val, x_hold, y_train, y_val, y_hold, num_items, num_users = processor.getData()
        
        #retrieve the nerual networks
        model_dot, model_nn = get_trained_models(num_users, num_items, x_train, y_train, x_val, y_val, EMBEDDING_SIZE, TRAIN, dataset)

        #Run experiments
        with open('results.csv', 'a') as r:
            if os.stat('results.csv').st_size == 0:
                r.write('timestamp,dataset,model,k,accuracy,ndcg\n')
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H:%M:%S - ')

            predict = rand_pred
            eval = Evaluator(x_hold, y_hold, K, predict)
            results = eval.evaluate()
            r.write(array2csv([timestamp, dataset, 'random', K, results[0], results[1]]))

            predict = model_dot.predict
            eval = Evaluator(x_hold, y_hold, K, predict)
            results = eval.evaluate()
            r.write(array2csv([timestamp, dataset, 'emb_dot', K, results[0], results[1]]))

            predict = model_nn.predict
            eval = Evaluator(x_hold, y_hold, K, predict)
            results = eval.evaluate()
            r.write(array2csv([timestamp, dataset, 'emb_nn', K, results[0], results[1]]))

            predictor = Predictor(model_dot, 'RANDOM_FOREST')
            predictor.fit(x_train, y_train)
            predict = predictor.predict
            eval = Evaluator(x_hold, y_hold, K, predict)
            results = eval.evaluate()
            r.write(array2csv([timestamp, dataset, 'emb_dot_random_forest', K, results[0], results[1]]))

            predictor = Predictor(model_dot, 'KNEIGH')
            predictor.fit(x_train, y_train)
            predict = predictor.predict

            eval = Evaluator(x_hold, y_hold, K, predict)
            results = eval.evaluate()
            r.write(array2csv([timestamp, dataset, 'emb_dot_kneighbors', K, results[0], results[1]]))

            predictor = Predictor(model_nn, 'RANDOM_FOREST')
            predictor.fit(x_train, y_train)
            predict = predictor.predict

            eval = Evaluator(x_hold, y_hold, K, predict)
            results = eval.evaluate()
            r.write(array2csv([timestamp, dataset, 'emb_nn_random_forest', K, results[0], results[1]]))

            predictor = Predictor(model_nn, 'KNEIGH')
            predictor.fit(x_train, y_train)
            predict = predictor.predict

            eval = Evaluator(x_hold, y_hold, K, predict)
            results = eval.evaluate()
            r.write(array2csv([timestamp, dataset, 'emb_nn_kneighbors', K, results[0], results[1]]))





    

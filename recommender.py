import numpy as np
import pandas as pd
import tensorflow as tf


import dfFunctions
import tf_models

class SVDmodel(object):
    """
    Class to creat SVD models. This class does not deal with tensorflow. It
    separate the dataframe in three parts: train, test and validation; with
    that it comunicates with the class tf_models.SVD to creat a training 
    session and to create a prediction. 

    We use the params users, items and ratings to get the names
    from the columns of df.


    :type df: dataframe  
    :type users: string  
    :type items: string
    :type ratings: string
    """
    def __init__(self,df,users, items, ratings):
        self.df = df
        self.users = users
        self.items = items
        self.ratings = ratings
        self.size = len(df)
        self.num_of_users = max(self.df[self.users]) + 1
        self.num_of_items = max(self.df[self.items]) + 1
        self.train,self.test,self.valid = self.data_separation()
        
    def data_separation(self):
    """
    Fuction that randomizes the dataframe df and separate it
    in tree parts: 80% in traing, 10% in test and 10% in validation.  
 
    :rtype: triple of dataframes
    """

        rows = len(self.df)
        random_df = self.df.iloc[np.random.permutation(rows)].reset_index(drop=True)
        split_index = int(rows * 0.8)
        new_split = split_index + int((rows - split_index) *0.5)
        df_train = random_df[0:split_index]
        df_test = random_df[split_index: new_split].reset_index(drop=True)
        df_validation = random_df[new_split:].reset_index(drop=True)
        return df_train, df_test,df_validation


    def training(self,hp_dim,hp_reg,learning_rate,batch_size,num_steps):
        self.train_batches = dfFunctions.BatchGenerator(self.train,batch_size,self.users,self.items,self.ratings)
        self.test_batches = dfFunctions.BatchGenerator(self.test,batch_size,self.users,self.items,self.ratings)
        self.valid_batches = dfFunctions.BatchGenerator(self.valid,len(self.valid),self.users,self.items,self.ratings)
        self.tf_counterpart = tf_models.SVD(self.num_of_users,self.num_of_items,self.train_batches,self.test_batches,self.valid_batches)
        self.tf_counterpart.training(hp_dim,hp_reg,learning_rate,batch_size,num_steps)
        self.tf_counterpart.print_stats()

    def valid_prediction(self):
        return self.tf_counterpart.prediction(show_valid=True)
       
    def prediction(self,list_of_users,list_of_items):
        return self.tf_counterpart.prediction(list_of_users,list_of_items)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension",type=int, default=15, help="embedding vector size (default=15)")
    parser.add_argument("-r", "--reg",     type=float, default=0.05, help="regularizer constant for the loss function  (default=0.05)")
    parser.add_argument("-l", "--learning", type=float,   default=0.001,   help="learning rate (default=0.001)")
    parser.add_argument("-b", "--batch",type=int, default=1000, help="batch size (default=1000)")
    parser.add_argument("-s", "--steps",type=int, default=5000, help="number of training (default=5000)")
    args = parser.parse_args()


    path = "/var/tmp/Felsal_Projects/Recommender/movielens/ml-1m/ratings.dat"
    df = dfFunctions.get_data(path, sep="::")
    model = SVDmodel(df,'user', 'item','rate')

    dimension = args.dimension
    regularizer_constant = args.reg
    learning_rate = args.learning
    batch_size = args.batch
    num_steps = args.steps

    model.training(dimension,regularizer_constant,learning_rate,batch_size,num_steps)
    model.valid_prediction()
    print(model.prediction(np.array([0,0,0]),np.array([1192,660,913])))


        

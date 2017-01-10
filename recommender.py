import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
from datetime import datetime, timedelta


import dfFunctions
import tf_models

class SVDmodel(object):
    
    def __init__(self,dataframe,users, items, ratings):
        self.dataframe = dataframe
        self.users = users
        self.items = items
        self.ratings = ratings
        self.size = len(dataframe)
        self.num_of_users = max(self.dataframe[self.users]) + 1
        self.num_of_items = max(self.dataframe[self.items]) + 1
        self.train,self.test,self.valid = self.data_separation()
        
    def data_separation(self):
        rows = len(self.dataframe)
        df = self.dataframe.iloc[np.randomd.permutation(rows)].reset_index(drop=True)
        split_index = int(rows * 0.9)
        new_split = split_index + int((rows - split_index) *0.5)
        df_train = df[0:split_index]
        df_test = df[split_index: new_split].reset_index(drop=True)
        df_validation = df[new_split:].reset_index(drop=True)
        return df_train, df_test,df_validation


    def training(self,hp_dim,hp_reg,learning_rate,batch_size,num_steps):
        self.train_batches = dfFunctions.BatchGenerator(self.train,batch_size,self.users,self.items,self.ratings)
        self.test_batches = dfFunctions.BatchGenerator(self.test,batch_size,self.users,self.items,self.ratings)
        self.valid_batches = dfFunctions.BatchGenerator(self.valid,len(self.valid),self.users,self.items,self.ratings)
        self.tf_counterpart = tf_models.SVD(self.num_of_users,self.num_of_items,self.train_batches,self.test_batches,self.valid_batches)
        self.tf_counterpart.training(hp_dim,hp_reg,learning_rate,batch_size,num_steps)
        self.tf_counterpart.print_stats()

    def valid_prediction(self):
        self.tf_counterpart.prediction(show_valid=True)
       
    def prediction(self,list_of_users,list_of_items):
        return self.tf_counterpart.prediction(list_of_users,list_of_items)
        

if __name__ == '__main__':
    path = "/var/tmp/Felsal_Projects/Recommender/movielens/ml-1m/ratings.dat"
    df = dfFunctions.get_data(path, sep="::")
    model = SVDmodel(df,'user', 'item','rate')

    dimension = 15
    regularizer_constant = 0.05
    learning_rate = 0.001
    batch_size = 1000
    num_steps = 4000

    model.training(dimension,regularizer_constant,learning_rate,batch_size,num_steps)
    model.valid_prediction()
    print(model.prediction(np.array([0,0,0]),np.array([1192,660,913])))


        

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
        self.general_duration = 0 
        self.num_steps = 0
        self.dimension = None
        self.regularizer = None
        #assuming that the dataframe has n items started with 0
        self.num_of_users = max(self.dataframe[self.users]) + 1
        self.num_of_items = max(self.dataframe[self.items]) + 1
        self.best_acc_test = float('inf')
        self.train,self.test,self.valid = self.data_separation()

        
    def data_separation(self):
        rows = len(self.dataframe)
        df = self.dataframe.iloc[np.random.permutation(rows)].reset_index(drop=True)
        split_index = int(rows * 0.9)
        new_split = split_index + int((rows - split_index) *0.5)
        df_train = df[0:split_index]
        df_test = df[split_index: new_split].reset_index(drop=True)
        df_validation = df[new_split:].reset_index(drop=True)
        return df_train, df_test,df_validation

    def accuracy(self,predictions, ratings):
        return np.sqrt(np.mean(np.power(predictions - ratings, 2)))

    def get_graph(self,hp_dim,hp_reg,learning_rate):

        "Defining Tensorflow Graph"
        
        self.dimension = hp_dim
        self.regularizer = hp_reg
        self.learning_rate = learning_rate
        self.graph = tf.Graph() 
        with self.graph.as_default():

            #Placeholders
            self.tf_user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
            self.tf_item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
            self.tf_rate_batch = tf.placeholder(tf.float32, shape=[None],name="actual_ratings")

            #Applying the model
            tf_svd_model = tf_models.inference_svd(self.tf_user_batch, self.tf_item_batch, user_num=self.num_of_users, item_num=self.num_of_items, dim=hp_dim)
            self.infer, regularizer = tf_svd_model['infer'], tf_svd_model['regularizer'] 

            global_step = tf.contrib.framework.get_or_create_global_step()

            with tf.name_scope('loss'):
                self.tf_cost = tf_models.loss_function(self.infer, regularizer,self.tf_rate_batch,reg=hp_reg)

            #Optimizer
            with tf.name_scope('training'):
                global_step = tf.contrib.framework.assert_or_get_global_step()
                assert global_step is not None
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.tf_cost, global_step=global_step)

            #Saver
            self.saver = tf.train.Saver()
            save_dir = 'checkpoints/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_path = os.path.join(save_dir, 'best_validation')

            #Minibatch accuracy
            with tf.name_scope('accuracy'):
                self.acc_op =  tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(self.infer,self.tf_rate_batch),2)))

    
    def training(self,hp_dim,hp_reg,learning_rate,batch_size,num_steps):
        self.get_graph(hp_dim,hp_reg,learning_rate)
        self.num_steps = num_steps
        train_batches = dfFunctions.BatchGenerator(self.train,batch_size,self.users,self.items,self.ratings)
        test_batches = dfFunctions.BatchGenerator(self.test,batch_size,self.users,self.items,self.ratings)
        marker = ''

        with tf.Session(graph=self.graph) as sess:
            tf.initialize_all_variables().run()
            print("{} {} {} {} {}".format("step", "batch_error", "test_error", "cost" ,"elapsed_time"))
            start = time.time()
            initial_time = start
            for step in range(num_steps):
                users, items, rates = train_batches.get_batch()
                feed_dict = {self.tf_user_batch: users, self.tf_item_batch: items, self.tf_rate_batch: rates}         
                _, pred_batch,cost,train_error = sess.run([self.train_op, self.infer, self.tf_cost,self.acc_op], feed_dict=feed_dict)
                if (step % 1000)  == 0:
                    users, items, rates = test_batches.get_batch() 
                    feed_dict = {self.tf_user_batch: users, self.tf_item_batch: items, self.tf_rate_batch: rates}              
                    pred_batch = sess.run(self.infer, feed_dict=feed_dict)
                    test_error = self.accuracy(pred_batch,rates)
                    if test_error < self.best_acc_test:
                        self.best_acc_test = test_error
                        marker = "*"
                        self.saver.save(sess=sess, save_path=self.save_path)

                    end = time.time()
                    print("{:3d} {:f} {:f}{:s} {:f} {:f}(s)".format(step, train_error, test_error,marker,cost,
                                                           end - start))
                    marker = ''
                    start = end
        #duration of the whole process  
        self.general_duration = time.time() - initial_time

    def print_stats(self):
        sec = timedelta(seconds=int(self.general_duration))
        d_time = datetime(1,1,1) + sec
        print(' ')
        print('The duration of the whole training with % s steps is %.2f seconds,'\
          % (self.num_steps,self.general_duration))
        print("which is equal to:  %d:%d:%d:%d" % (d_time.day-1, d_time.hour, d_time.minute, d_time.second), end='')
        print(" (DAYS:HOURS:MIN:SEC)")

    def prediction(self,list_of_users=None,list_of_items=None,show_valid=False):
        if self.dimension == None and self.regularizer == None:
            print("You can not have a prediction without training!!!!")
        else:
            valid_batches = dfFunctions.BatchGenerator(self.valid,len(self.valid),self.users,self.items,self.ratings)
            self.get_graph(self.dimension,self.regularizer,self.learning_rate)
            with tf.Session(graph=self.graph) as sess:
                self.saver.restore(sess=sess, save_path=self.save_path)
                users, items, rates = valid_batches.get_batch()
                if show_valid:
                    feed_dict = {self.tf_user_batch: users, self.tf_item_batch: items, self.tf_rate_batch: rates}
                    valid_error = sess.run(self.acc_op, feed_dict=feed_dict)
                    print("Avarege error of the whole valid dataset: ", valid_error)         
                else:
                    feed_dict = {self.tf_user_batch: list_of_users, self.tf_item_batch: list_of_items, self.tf_rate_batch: rates}
                    prediction = sess.run(self.infer, feed_dict=feed_dict)
                    return prediction    





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
    model.print_stats()
    model.prediction(show_valid=True)
    print(model.prediction(np.array([0,0,0]),np.array([1192,660,913])))


        

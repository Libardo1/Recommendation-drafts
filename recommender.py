from __future__ import absolute_import, division, print_function
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
        self.valid_dataset = None
        self.num_steps = 0
        #assuming that the dataframe has n items started with 0
        self.num_of_users = max(self.dataframe[self.users]) + 1
        self.num_of_items = max(self.dataframe[self.items]) + 1
        
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

    def tf_training(self,hp_dim,hp_reg,learning_rate,batch_size,num_steps):
        #creating tf graph
        graph = tf.Graph() 
        with graph.as_default():

            #Placeholders
            tf_user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
            tf_item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
            tf_rate_batch = tf.placeholder(tf.float32, shape=[None],name="actual_ratings")

            #Applying the model
            tf_svd_model = tf_models.inference_svd(tf_user_batch, tf_item_batch, user_num=self.num_of_users, item_num=self.num_of_items, dim=hp_dim)
            infer, regularizer = tf_svd_model['infer'], tf_svd_model['regularizer'] 

            global_step = tf.contrib.framework.get_or_create_global_step()

            with tf.name_scope('loss'):
                tf_cost = tf_models.loss_function(infer, regularizer,tf_rate_batch,reg=hp_reg)

            #Optimizer.
            with tf.name_scope('training'):
                global_step = tf.contrib.framework.assert_or_get_global_step()
                assert global_step is not None
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_cost, global_step=global_step)

            #Minibatch accuracy
            with tf.name_scope('accuracy'):
                acc_op =  tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(infer,tf_rate_batch),2)))

     
        train, test, valid = self.data_separation()
        self.valid_dataset = valid
        self.num_steps = num_steps
        train_batches = dfFunctions.BatchGenerator(train,batch_size,self.users,self.items,self.ratings)
        test_batches = dfFunctions.BatchGenerator(test,batch_size,self.users,self.items,self.ratings)


        with tf.Session(graph=graph) as sess:
            tf.initialize_all_variables().run()
            print("{} {} {} {} {}".format("step", "batch_error", "test_error", "cost" ,"elapsed_time"))
            start = time.time()
            initial_time = start
            for step in range(num_steps):
                users, items, rates = train_batches.get_batch()      
                _, pred_batch,cost,train_error = sess.run([train_op, infer, tf_cost,acc_op], feed_dict={tf_user_batch: users,
                                                                       tf_item_batch: items,
                                                                       tf_rate_batch: rates})
                if (step % 1000)  == 0:
                    users, items, rates = test_batches.get_batch()            
                    pred_batch = sess.run(infer, feed_dict={tf_user_batch: users,
                                                                       tf_item_batch: items,
                                                                       tf_rate_batch: rates})
                    test_error = self.accuracy(pred_batch,rates)
                    end = time.time()
                    print("{:3d} {:f} {:f} {:f} {:f}(s)".format(step, train_error, test_error,cost,
                                                           end - start))
                    start = end

        #printing the duration of the whole process  
        self.general_duration = time.time() - initial_time

    def print_stats(self):
        sec = timedelta(seconds=int(self.general_duration))
        d_time = datetime(1,1,1) + sec
        print(' ')
        print('The duration of the whole training with % s steps is %.2f seconds,'\
          % (self.num_steps,self.general_duration))
        print("which is equal to:  %d:%d:%d:%d" % (d_time.day-1, d_time.hour, d_time.minute, d_time.second), end='')
        print(" (DAYS:HOURS:MIN:SEC)")


if __name__ == '__main__':
    df = dfFunctions.get_data_raw()
    model = SVDmodel(df,'user', 'item','rate')

    dimension = 15
    regularizer_constant = 0.05
    learning_rate = 0.001
    batch_size = 1000
    num_steps = 90000

    model.tf_training(dimension,regularizer_constant,learning_rate,batch_size,num_steps)
    model.print_stats()


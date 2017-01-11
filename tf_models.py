import tensorflow as tf
from utils import accuracy
import os
import time
from datetime import datetime,timedelta


def inference_svd(user_batch, item_batch, user_num, item_num, dim=5):
    """
    This function creates one tensor of shape=[dim] for every user
    and every item. We select the indices for users from the tensor user_batch
    and select the indices for items from the tensor item_batch. After that we
    calculate the infered score as the inner product between the user vector and
    the item vector (we also sum the global bias, the bias from that user and 
    the bias from that item). infer is the tensor with the result of this
    caculation.

    We calculate also a regularizer to use in the loss function. This function
    returns a dictionary with the tensors infer, regularizer, w_user (tensor with
    all the user vectors) and w_items (tensor with all the item vectors).
    
    :type item_batch: tensor of int32     
    :type user_batch: tensor of int32    
    :type user_num: int
    :type item_num: int
    :type dim: int
    :rtype: dictionary   

    """
    with tf.name_scope('Declaring_variables'):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    with tf.name_scope('Rating_prediction'):
        infer = tf.reduce_sum(tf.mul(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
    dic_of_values = {'infer': infer, 'regularizer': regularizer, 'w_user': w_user, 'w_item': w_item}    
    return dic_of_values


def loss_function(infer, regularizer, rate_batch,reg):
    cost_l2 = tf.nn.l2_loss(tf.sub(infer, rate_batch))
    penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
    cost = tf.add(cost_l2, tf.mul(regularizer, penalty))
    return cost


class SVD(object):

    def __init__(self,num_of_users,num_of_items,train_batch_generator,test_batch_generator,valid_batch_generator):
        self.num_of_users = num_of_users
        self.num_of_items = num_of_items
        self.train_batch_generator = train_batch_generator
        self.test_batch_generator = test_batch_generator
        self.valid_batch_generator = valid_batch_generator
        self.general_duration = 0 
        self.num_steps = 0
        self.dimension = None
        self.regularizer = None
        self.best_acc_test = float('inf')


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
            tf_svd_model = inference_svd(self.tf_user_batch, self.tf_item_batch, user_num=self.num_of_users, item_num=self.num_of_items, dim=hp_dim)
            self.infer, regularizer = tf_svd_model['infer'], tf_svd_model['regularizer'] 

            global_step = tf.contrib.framework.get_or_create_global_step()

            with tf.name_scope('loss'):
                self.tf_cost = loss_function(self.infer, regularizer,self.tf_rate_batch,reg=hp_reg)

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
        marker = ''

        with tf.Session(graph=self.graph) as sess:
            tf.initialize_all_variables().run()
            print("{} {} {} {}".format("step", "batch_error", "test_error","elapsed_time"))
            start = time.time()
            initial_time = start
            for step in range(num_steps):
                users, items, rates = self.train_batch_generator.get_batch()
                feed_dict = {self.tf_user_batch: users, self.tf_item_batch: items, self.tf_rate_batch: rates}         
                _, pred_batch,cost,train_error = sess.run([self.train_op, self.infer, self.tf_cost,self.acc_op], feed_dict=feed_dict)
                if (step % 1000)  == 0:
                    users, items, rates = self.test_batch_generator.get_batch() 
                    feed_dict = {self.tf_user_batch: users, self.tf_item_batch: items, self.tf_rate_batch: rates}              
                    pred_batch = sess.run(self.infer, feed_dict=feed_dict)
                    test_error = accuracy(pred_batch,rates)
                    if test_error < self.best_acc_test:
                        self.best_acc_test = test_error
                        marker = "*"
                        self.saver.save(sess=sess, save_path=self.save_path)

                    end = time.time()
                    print("{:3d} {:f} {:f}{:s} {:f}(s)".format(step, train_error, test_error,marker,
                                                           end - start))
                    marker = ''
                    start = end 
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
            self.get_graph(self.dimension,self.regularizer,self.learning_rate)
            with tf.Session(graph=self.graph) as sess:
                self.saver.restore(sess=sess, save_path=self.save_path)
                users, items, rates = self.valid_batch_generator.get_batch()
                if show_valid:
                    feed_dict = {self.tf_user_batch: users, self.tf_item_batch: items, self.tf_rate_batch: rates}
                    valid_error = sess.run(self.acc_op, feed_dict=feed_dict)
                    #print("Avarege error of the whole valid dataset: ", valid_error)
                    return valid_error         
                else:
                    feed_dict = {self.tf_user_batch: list_of_users, self.tf_item_batch: list_of_items, self.tf_rate_batch: rates}
                    prediction = sess.run(self.infer, feed_dict=feed_dict)
                    return prediction    

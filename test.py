import unittest
import numpy as np
import pandas as pd
import tensorflow as tf


import dfFunctions
import tf_models
import recommender as re

class TestRecomendation(unittest.TestCase):
    """
    asasasa
    """
    def test_upperbound(self):
        path = "/var/tmp/Felsal_Projects/Recommender/movielens/ml-1m/ratings.dat"
        df = dfFunctions.get_data(path, sep="::")
        model = re.SVDmodel(df,'user', 'item','rate')

        dimension = 15
        regularizer_constant = 0.05
        learning_rate = 0.001
        batch_size = 1000
        num_steps = 5000

        model.training(dimension,regularizer_constant,learning_rate,batch_size,num_steps)
        prediction = model.valid_prediction()
        self.assertTrue(prediction <=1.0001, \
                            "\n with num_steps = {0} \n, the mean square error of the valid dataset should be less than 1 and not {1}"\
                            .format(num_steps,prediction))



def run_test():
    """ 
    sssss
    """
    print("Running some tests...")
    suite = unittest.TestSuite()
    for method in dir(TestRecomendation):
       if method.startswith("test"):
          suite.addTest(TestRecomendation(method))
    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    run_test()

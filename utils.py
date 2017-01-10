import numpy as np

def accuracy(predictions, ratings):
    return np.sqrt(np.mean(np.power(predictions - ratings, 2)))

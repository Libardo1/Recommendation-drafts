import numpy as np
import pandas as pd


def get_data(filname, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df



class BatchGenerator(object):
    
    def __init__(self,dataframe,batch_size,users,items,ratings):
        self.batch_size = batch_size
        self.users = np.array(dataframe[users])
        self.items = np.array(dataframe[items])
        self.ratings = np.array(dataframe[ratings])
        self.num_cols = len(dataframe.columns)
        self.size = len(dataframe)
        
    def get_batch(self):
        random_indices = np.random.randint(0,self.size,self.batch_size)
        users = self.users[random_indices]
        items = self.items[random_indices]
        ratings = self.ratings[random_indices]         
        return users, items, ratings

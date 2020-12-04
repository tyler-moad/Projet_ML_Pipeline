import pandas as pd
import numpy as np


class DataLoader:
    """@Author Mouad Jallouli"""
    def __init__(self, data_path,test_size:float=0.2,train_size:float=None,header :bool =True ):
        self.test_size = test_size
        self.train_size = train_size
        if header:
            self.data = pd.read_csv(data_path)
        else:
            self.data = pd.read_csv(data_path,header=None)
        self.target = None
        self.features = []
        self.X, self.X_train, self.X_test = None, None, None
        self.y, self.y_train, self.y_test = None,None, None
        
    """@Author Reda Bensaid"""
    @staticmethod
    def train_test_split(X, y, test_size):

        permutation = np.random.permutation(len(X))
        X = X.iloc[permutation]
        y = y.iloc[permutation]

        X_train = X.iloc[:int((1 - test_size) * len(X))]
        y_train = y.iloc[:int((1 - test_size) * len(y))]
        X_test = X.iloc[int((1 - test_size) * len(X)):]
        y_test = y.iloc[int((1 - test_size) * len(y)):]

        return X_train, X_test, y_train, y_test
    """@Author Mouad Jallouli"""
    def load(self,target_column,columns_to_remove=[]):
        print(columns_to_remove)
        self.target = target_column
        for col in self.data.columns:
            if col != self.target and col not in columns_to_remove :
                print(col)
                self.features.append(col)
        self.X, self.y = self.data[self.features], self.data[self.target]
        self.X_train, self.X_test,self.y_train, self.y_test = DataLoader.train_test_split(self.X, self.y,
                                                            test_size=self.test_size)
    
    



    
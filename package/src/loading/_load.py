import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
class DataLoader:
    def __init__(self, data_path,test_size:float=0.2,train_size:float=None):
        self.test_size = test_size
        self.train_size = train_size
        self.data = pd.read_csv(data_path) 
        self.target = None
        self.features = []
        self.X, self.X_train, self.X_test = None, None, None
        self.y, self.y_train, self.y_test = None,None, None

    def train_test_split(X, y, test_size):

        permutation = np.random.permutation(len(X))
        X = X.iloc[permutation]
        y = y.iloc[permutation]

        X_train = X.iloc[:int((1 - test_size) * len(X))]
        y_train = y.iloc[:int((1 - test_size) * len(y))]
        X_test = X.iloc[int((1 - test_size) * len(X)):]
        y_test = y.iloc[int((1 - test_size) * len(y)):]

        return X_train, X_test, y_train, y_test

    def load(self,target_column):
        self.target = target_column
        for col in self.data.columns:
            if col != self.target:
                self.features.append(col)
        self.X, self.y = self.data[self.features], self.data[self.target]
        self.X_train, self.X_test,self.y_train, self.y_test = DataLoader.train_test_split(self.X, self.y,
                                                            test_size=self.test_size)
    
    



    
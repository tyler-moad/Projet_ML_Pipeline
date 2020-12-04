import pandas as pd
import numpy as np


class DataLoader:
    """@Author Mouad Jallouli"""
=======
    """
    This class implements a dataloader that is responsible for loading the datasets in diffenrent formats and split it to train test data
    """
    def __init__(self, data_path,test_size:float=0.2,train_size:float=None,header :bool =True ):
        """
        :param data_path: the absolute path to the dataset file
        :param test_size: the portion size of the test dataset
        :param train_size: the portion size of the train dataset
        :param header: specify whether the dataset contains any header representing column names
        """
        self.test_size = test_size
        self.train_size = train_size
        #load data according the the presence or not of headers
        if header:
            self.data = pd.read_csv(data_path)
        else:
            self.data = pd.read_csv(data_path,header=None)
        # initialize the data containers
        self.target = None
        self.features = []
        self.X, self.X_train, self.X_test = None, None, None
        self.y, self.y_train, self.y_test = None,None, None
        
    """@Author Reda Bensaid"""
    @staticmethod
    def train_test_split(X, y, test_size):
        """
        this method splits the dataset into a train set and a test test
        :param X: the design matrix dataframe
        :param y: the labels dataframe
        :param test_size: the portion size of the test set
        :return:X_train,X_test,y_train,y_test representing the data splitted
        """
        #We do a random shuffling of the dataset
        permutation = np.random.permutation(len(X))
        X = X.iloc[permutation]
        y = y.iloc[permutation]
        #We split the data to a train part and a test part
        X_train = X.iloc[:int((1 - test_size) * len(X))]
        y_train = y.iloc[:int((1 - test_size) * len(y))]
        X_test = X.iloc[int((1 - test_size) * len(X)):]
        y_test = y.iloc[int((1 - test_size) * len(y)):]

        return X_train, X_test, y_train, y_test
    """@Author Mouad Jallouli"""
    def load(self,target_column,columns_to_remove=[]):
        """
        this method is the final method for loading a dataset from pandas conmpatible file (".csv,.txt,...)
        :param target_column: a string representing which column represents the labels
        :param columns_to_remove: a list of strings representing which columns the user wants to drop
        :return:
        """
        self.target = target_column
        for col in self.data.columns:
            if col != self.target and col not in columns_to_remove :
                self.features.append(col)
        self.X, self.y = self.data[self.features], self.data[self.target]
        self.X_train, self.X_test,self.y_train, self.y_test = DataLoader.train_test_split(self.X, self.y,
                                                            test_size=self.test_size)
        print("Dataset has been loaded in split into training and test data")
    
    



    
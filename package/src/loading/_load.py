import pandas as pd 
from sklearn.model_selection import train_test_split
class DataLoader:
    def __init__(self, data_path,test_size:float=0.1,train_size:float=None):
        self.test_size = test_size
        self.train_size = train_size
        self.data = pd.read_csv(data_path) 
        self.target = None
        self.features = []
        self.X, self.X_train, self.X_test = None, None, None
        self.y, self.y_train, self.y_test = None,None, None

    def load(self,target_column):
        self.target = target_column
        for col in self.data.columns:
            if col != self.target:
                self.features.append(col)
        self.X, self.y = self.data[self.features], self.data[self.target]
        self.X_train, self.X_test,self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                            test_size=self.test_size, stratify = self.y)
    
    



    
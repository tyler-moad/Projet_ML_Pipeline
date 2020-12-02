import numpy as np
from sklearn.feature_selection import SelectKBest,mutual_info_classif
class FeatureSelection:
    def __init__(self):
        self.fs = SelectKBest(score_func=mutual_info_classif, k=5)
        
    def fit(X,y,n_features=8):
        
        if n_features:
            self.fs.set_params(k=n_features);
	    self.fs.fit(X, y)

    def transform(X):
        return self.fs.transform(X)
      

    


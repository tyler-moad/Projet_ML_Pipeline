from sklearn.base import BaseEstimator
import numpy as np
class Model(BaseEstimator):
    def __init__(self,model,params=None):
        self.model = model
        self.params = params
    
    def f1_score(y_true,y_predict):
        p=0
        r=0
        n_classes = y_true.nunique()
        print(n_classes)
        for i in range(n_classes):
            tp= sum(((y_true==i) & (y_predict==i)))
            fp= sum(((y_true!=i) & (y_predict==i)))
            fn= sum(((y_true==i) & (y_predict!=i)))
            p += tp/(tp+fp)
            r += tp/(tp+fn)
        
        return 2*p*r/(p+r)
    def gridsearchCV(self,X,y,param_grid):
        from sklearn.model_selection import ParameterGrid
        scores = {}
        for params in ParameterGrid(param_grid):
            model = self.model.set_params(**params)
            score = self.cross_validation_score(X,y,model)
            scores[score] = model
        best_model = scores[max(scores.keys())]
        score = np.mean(scores.keys())
        return best_model, score
            

    

    def cross_validation_score(self,X_train:np.array,y_train:np.array,model,k = 10) : 
    
        evaluation = []
        for i in range(k):
            X_test_fold = X_train[int((i/k)*len(X_train)):int(((i+1)/k)*len(X_train))]
            y_test_fold = y_train[int((i/k)*len(y_train)):int(((i+1)/k)*len(y_train))]
            X_train_fold = np.concatenate((X_train[:int((i/k)*len(X_train))],X_train[int(((i+1)/k)*len(X_train)):]))
            y_train_fold = np.concatenate((y_train[:int((i/k)*len(y_train))],y_train[int(((i+1)/k)*len(y_train)):]))
            model.fit(X_train_fold,y_train_fold)
            y_pred = model.predict(X_test_fold)
            evaluation.append(f1_score(y_test_fold, y_pred))
        return np.mean(evaluation)


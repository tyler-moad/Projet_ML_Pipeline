from sklearn.base import BaseEstimator
import numpy as np
class Model(BaseEstimator):
    def __init__(self,model,params=None):
        self.model = model
        self.params = params
    """ def set_params():
        self.model.set_params(**self.params)
    def set_params_grid(param_grid:dict):
        self.param_grid = param_grid
    def fit(self,X:np.array,y:np.array):
        self.model.fit(X,y)
    def predict(X:np.array):
        return self.model.predict(X) """

    def f1_score(self,y_true,y_predict):
        p=[]
        r=[]
        n_classes = y_true.nunique()
        for i in range(n_classes):
            tp= sum(((y_true==i) & (y_predict==i)))
            fp= sum(((y_true!=i) & (y_predict==i)))
            fn= sum(((y_true==i) & (y_predict!=i)))
            p.append(tp/(tp+fp))
            r.append(tp/(tp+fn))

        return 2*p*r/(p+r)

    def eval(self,X:np.array,y:np.array,metric:str,cv:int=10,avg=True):
        """ from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model,X,y,cv=cv)
        if avg:
            return np.mean(scores)
        return scores """
        y_pred = self.model.predict(X)
        return self.f1_score(y,y_pred)

    def gridsearchCV(self,X,y,param_grid):
        scores = {}
        for c in param_grid['C']:
            params={'C':c}
            model = self.model.set_params(**params)
            score = self.cross_validation(X,y,model)
            scores[score] = model
        best_model = scores[max(scores.keys())]
        return best_model
            

    def find_best_model(self,X,y,param_grid,scoring:str):
        """Set model to the model with the best parameters
        Parameters
        ----------
        X : array-like
            Input data
        y : array-like
            Target for Input data
        scoring : str
            used metric
            
        """
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(self.model,param_grid=param_grid,scoring=scoring,cv=10)
        self.model = cv.best_estimator_
        return cv.scores




    def cross_validation_score (self,X_train:np.array,y_train:np.array,k = 10) : 
    
        evaluation = []
        for i in range(k):
            X_test_fold = X_train[int((i/k)*len(X_train)):int(((i+1)/k)*len(X_train))]
            y_test_fold = y_train[int((i/k)*len(y_train)):int(((i+1)/k)*len(y_train))]
            X_train_fold = np.concatenate((X_train[:int((i/k)*len(X_train))],X_train[int(((i+1)/k)*len(X_train)):]))
            y_train_fold = np.concatenate((y_train[:int((i/k)*len(y_train))],y_train[int(((i+1)/k)*len(y_train)):]))
            self.model.fit(X_train_fold,y_train_fold)
            y_pred = self.model.predict(X_test_fold)
            evaluation.append(self.f1_score(y_test_fold, y_pred))
        return np.mean(evaluation)


    def eval(self,X:np.array,y:np.array,cv:int=10,avg=True):
        scores = self.cross_validation_score(self,X_train,y_train,cv)
        if avg:
            return np.mean(scores)
        return scores
        y_pred = self.model.predict(X)
        return self.f1_score(y,y_pred)
        

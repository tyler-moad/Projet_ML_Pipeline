import typing
class ModelPipeline:
    def __init__(self,model,params=None):
        self.model = model
        self.params = params
    def set_params():
        self.model.set_params(**self.params)
    def set_params_grid(param_grid:dict):
        self.param_grid = param_grid
    def fit(self,X:np.array,y:np.array):
        self.model.fit(X,y)
    def predict(X:np.array):
        return self.model.predict(X)
    def f1_score(y_true,y_predict):
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

    def find_best_model(X,y,param_grid,scoring:str):
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




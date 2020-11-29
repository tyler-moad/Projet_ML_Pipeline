import typing
class ModelPipeline:
    def __init__(self,model,params):
        self.model = model
        self.params = params
        self.params_grid = None
    def set_params():
        self.model.set_params(**self.params)
    def set_params_grid(param_grid:dict):
        self.param_grid = param_grid
    def fit(self,X:np.array,y:np.array):
        self.model.fit(X,y)
    def predict(X:np.array):
        return self.model.predict(X)
    def eval(self,X:np.array,y:np.array,metric:str,cv:int=10,avg=False):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model,X,y,cv=cv)
        if avg:
            return np.mean(scores)
        return scores
        
    def best_model(X,y,scoring:str):
        from sklearn.model_selection import GridSearchCV
        cv = GridSearchCV(self.model,param_grid=self.param_grid,scoring=scoring)
        return cv.best_estimator_

        
    def cross_val(X,y,scoring:str):



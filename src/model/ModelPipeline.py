import typing
class ModelPipeline:
    def __init__(self,model,params):
        self.model = model
        self.params = params
    def transform(self):
        pass
    def set_params():
        self.model.set_params(**self.params)
    def fit(self,X:np.array,y:np.array):
        self.model.fit(X,y)
    def predict(X:np.array):
        return self.model.predict(X)
    def eval(self,X:np.array,y:np.array,metric:List[str]):
        pass
    def precision_recall(y_true,y_predict,n_classes):
        p=[]
        r=[]
        for i in range(n_classes):
            tp= sum(((y_true==i) & (y_predict==i)))
            fp= sum(((y_true!=i) & (y_predict==i)))
            fn= sum(((y_true==i) & (y_predict!=i)))
            p.append(tp/(tp+fp))
            r.append(tp/(tp+fn))
        
        return p,r
    def grid_search()
    def cross_val(X,y,scoring:str):



from sklearn.base import BaseEstimator
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
    def gridsearchCV(X,y,param_grid):
        scores = {}
        for c in param_grid['C']:
            model = self.model.set_params(**{'C'=c})
            score = self.cross_validation(X,y,model)
            scores[score] = model
        best_model = scores[max(scores.keys())]
        return best_model
            

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


    def train_test_split(X,Y,test_size = 0.1, validation_size = 0):
    
        if len(X) != len(Y) : 
            print("error")
        permutation = np.random.permutation(len(X))
        X = X[permutation]
        Y = Y[permutation]
        if test_size + validation_size >= 1 : 
            print("error")
        else : 
            X_train = X[:int((1-test_size-validation_size)*len(X))]
            Y_train = Y[:int((1-test_size-validation_size)*len(Y))]
            X_validation = X[int((1-test_size-validation_size)*len(X)):int((1-test_size)*len(X))]
            Y_validation = Y[int((1-test_size-validation_size)*len(Y)):int((1-test_size)*len(Y))]
            X_test = X[int((1-test_size)*len(X)):]
            Y_test = Y[int((1-test_size)*len(Y)):]


        return X_train, Y_train, X_validation, Y_validation, X_test, Y_test


    def cross_validation (X_train,Y_train, model = "dip_leurning", k = 10) : 
    
        evaluation = []
        if len(X) != len(Y) : 
            print("error")
        for i in range(k):
            X_test_fold = X_train[int((i/k)*len(X)):int(((i+1)/k)*len(X))]
            Y_test_fold = Y_train[int((i/k)*len(Y)):int(((i+1)/k)*len(Y))]
            X_train_fold = np.concatenate((X_train[:int((i/k)*len(X))],X_train[int(((i+1)/k)*len(X)):]))
            Y_train_fold = np.concatenate((Y_train[:int((i/k)*len(Y))],Y_train[int(((i+1)/k)*len(Y)):]))
            train(X_train_fold,Y_train_fold, model)
            evaluation.append(test(X_test_fold,Y_test_fold, model))
        return evaluation



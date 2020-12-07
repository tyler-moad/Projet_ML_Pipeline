from itertools import product

class Model():

    def __init__(self, model):
        self.model = model
        self.metric = metric
        self.best_score_ = None
        self.best_params_ = None

    """@Author Reda Bensaid"""
    def cross_validation_score(self, X_train: np.array, y_train: np.array,metric:str = "accuracy",  k=10):
        """
             this method calculate the performance of the model trained and tested
             over all the folds and then returns the mean
             :param X_train: the training samples
             :param y_train: the training labels
             :param metric : the metric to use to calculate the performance of the model (accuracy of f1_score)
             :param k : number of folds
             :return: the mean of the metrics over the training and test of all folds
          """
        evaluation = []
        for i in range(k):
            X_test_fold = X_train[int((i / k) * len(X_train)):int(((i + 1) / k) * len(X_train))]
            y_test_fold = y_train[int((i / k) * len(y_train)):int(((i + 1) / k) * len(y_train))]
            
            X_train_fold = np.concatenate(
                (X_train[:int((i / k) * len(X_train))], X_train[int(((i + 1) / k) * len(X_train)):]))
            y_train_fold = np.concatenate(
                (y_train[:int((i / k) * len(y_train))], y_train[int(((i + 1) / k) * len(y_train)):]))
            self.model.fit(X_train_fold, y_train_fold)
            y_pred = self.model.predict(X_test_fold)
            if metric == "accuracy" : 
                evaluation.append(self.accuracy(y_test_fold, y_pred))
            else : 
                evaluation.append(self.f1_score(y_test_fold, y_pred))
        return np.mean(evaluation)

    """@Author Moad Taoufik"""
    def generateGrid(self,params):
        keys, values = zip(*params.items())
        grid = []
        for v in product(*values):
            d = dict(zip(keys, v))
            grid.append(d)
        return grid
        
    """@Author Mouad Jallouli"""
    def gridsearchCV(self,X,y,params):
        params_grid = self.generateGrid(params)
        
        best_param = None
        best_score = 0
        for param in params_grid:
            print(param)
            self.model.set_params(**param)
            score = self.cross_validation_score(X,y)
            print(score)
            if score > best_score:
                best_score = score
                best_param = param
        self.best_score_ = best_score
        self.best_params_ = best_param

    """@Author Mouad Jallouli"""
    @staticmethod
    def f1_score(y_true, y_predict):
        p = 0
        r = 0
        f = 0
        n_classes = y_true.nunique()
        for i in range(n_classes):
            tp = sum(((y_true == i) & (y_predict == i)))
            fp = sum(((y_true != i) & (y_predict == i)))
            fn = sum(((y_true == i) & (y_predict != i)))
            if fp == 0: # edge case
                p = 1
            else:
                p = tp / (tp + fp)
            if fn == 0: # edge case
                r = 1
            else:
                r = tp / (tp + fn)
            f += 2 * p * r / (p + r)
    
        return f/n_classes 
    
    """@Author Reda Bensaid"""
    @staticmethod    
    def accuracy (y_true, y_predict):
        """
           this method calculate the accuracy of the prediction
           :param y_true: the labels
           :param y_predict: the predictions
           :return: accuracy of the prediction
        """
        good_predictions = np.sum((y_true == y_predict)*1)
        return good_predictions /len(y_true)

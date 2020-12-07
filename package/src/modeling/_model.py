from itertools import product
class Model()
    def __init__(self, model):
        self.model = model

        self.params = params
    
    @staticmethod
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
        p = p / n_classes
        r = r / n_classes
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
            


    def fit(self,X,Y,params_dic):
        self.model.set_params(**params_dic)
        self.model.fit(X,Y)
    def predict(self,X):
        return self.model.predict(X)


    def cross_validation_score(self, X_train: np.array, y_train: np.array, k=10):
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
            evaluation.append(f1_score(y_test_fold, y_pred))
        return np.mean(evaluation)

    def generateGrid(self,params):
        keys, values = zip(*params.items())
        grid = []
        for v in product(*values):
            d = dict(zip(keys, v))
            grid.append(d)

    def gridsearchCV(self,X,y,params):
        param_grid = self.generateGrid(params)
        scores_dict = {}
        best_param = None
        best_score = 0
        for param in params_grid:
            self.model.set_params(**param)
            score = self.cross_validation_score(X,y)
            scores_dict[param] = score
            if score > best_score:
                best_score = score
                best_param = param
        self.best_score = best_score
        self.best_params = best_param

    def f1_score(y_true, y_predict):
        p = 0
        r = 0
        n_classes = y_true.nunique()
        print(n_classes)
        for i in range(n_classes):
            tp = sum(((y_true == i) & (y_predict == i)))
            fp = sum(((y_true != i) & (y_predict == i)))
            fn = sum(((y_true == i) & (y_predict != i)))
            p += tp / (tp + fp)
            r += tp / (tp + fn)
        return 2 * p * r / (p + r)

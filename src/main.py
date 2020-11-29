from preprocessing import DataLoader
from model import ModelPipeline
from sklearn.svm import SVC

#Data loading and preprocessing
dl = DataLoader()
# Instantiate the MOdel 
pipe = ModelPipeline(SVC())

# Hyperparameter tuning
X, y = dl.data_tain
param_grid = {'C':[.1,1,10]}
pipe.best_model(X,y,param_grid,scoring='f1')

# Testing using f1_score
X_test,y_test = dl.data_test
score = pipe.eval(X_test,y_test)

import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import time
from itertools import product

class DataLoader:
    """@Author Mouad Jallouli"""
    """
    This class implements a dataloader that is responsible for loading the datasets in different formats and split it to train test data
    """

    def __init__(self, data_path, test_size: float = 0.2, train_size: float = None, header: bool = True):
        """
        :param data_path: the absolute path to the dataset file
        :param test_size: the portion size of the test dataset
        :param train_size: the portion size of the train dataset
        :param header: specify whether the dataset contains any header representing columns names
        """
        self.test_size = test_size
        self.train_size = train_size
        # load data according the the presence or not of headers
        if header:
            self.data = pd.read_csv(data_path)
        else:
            self.data = pd.read_csv(data_path, header=None)
        # initialize the data containers
        self.target = None
        self.features = []
        self.X, self.X_train, self.X_test = None, None, None
        self.y, self.y_train, self.y_test = None, None, None


    """@Author Reda Bensaid"""


    @staticmethod
    def train_test_split(X, y, test_size):
        """
        this method splits the dataset into a train set and a test test
        :param X: the design matrix dataframe
        :param y: the labels dataframe
        :param test_size: the portion size of the test set
        :return:X_train,X_test,y_train,y_test representing the data splitted
        """
        # We do a random shuffling of the dataset
        permutation = np.random.permutation(len(X))
        X = X.iloc[permutation]
        y = y.iloc[permutation]
        # We split the data to a train part and a test part
        X_train = X.iloc[:int((1 - test_size) * len(X))]
        y_train = y.iloc[:int((1 - test_size) * len(y))]
        X_test = X.iloc[int((1 - test_size) * len(X)):]
        y_test = y.iloc[int((1 - test_size) * len(y)):]

        return X_train, X_test, y_train, y_test


    """@Author Mouad Jallouli"""


    def load(self, target_column, columns_to_remove=[]):
        """
        this method is the final method for loading a dataset from pandas conmpatible file (".csv,.txt,...)
        :param target_column: a string representing which column represents the labels
        :param columns_to_remove: a list of strings representing which columns the user wants to drop
        :return:
        """
        self.target = target_column
        for col in self.data.columns:
            if col != self.target and col not in columns_to_remove:
                self.features.append(col)
        self.X, self.y = self.data[self.features], self.data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = DataLoader.train_test_split(self.X, self.y,
                                                                                           test_size=self.test_size)
        print("Dataset has been loaded in split into training and test data")


class DataPreprocessor:
    """
    This class implements a data preprocessor that is responsible for cleaning the dataset by input NaN values ,
    correcting typos in the dataset and encoding it in a suitable categorical format for machine learning algorithms
    """

    def __init__(self, missing_data_strategy: str = "mean", categorical_encoding: str = "label", categorical_threshold=15):
        """
        The init for the data preprocessor
        :param missing_data_strategy: the strategy that should be used to input missing data : we can either select "mean"
        that will replace every missing value in a column by the mean of that column or select "median" tha will replace
        it by the median value of the column or even choose "radical" to give on the whole column
        if we are dealing with categorical column "mean" or "median" will set the values to the most present category in the column
        :param categorical_encoding: the encoding scheme that is chosen coul be "label" for label encoding or "onehot" for onehot encoding
        :param categorical_threshold: an integer that represents the maximum number of different values a categorical column can have
        """
        #Sanity checks for input strings
        if missing_data_strategy in ["mean", "median", "radical"]:
            self.missing_stategy = missing_data_strategy
        else:
            raise ValueError("Missing data strategy not in allowed methods ")
        if categorical_encoding in ["label", "onehot"]:
            self.categorical_encoding_stategy = categorical_encoding
        else:
            raise ValueError("catagorical encoding strategy not in allowed methods ")

        self.categorical_threshold = categorical_threshold
        self.columns_data=None
        self.typos_data=None
        self.columns=None


    def find_continuous_columns(self):
        """
        This method is responsible of finding all the continuous valued columns as there may be some datasets where continous variables
        are inputed as string objects.
        If such column is found it is converted to the numeric format
        """
        #search over all columns
        for col_name in self.data.columns:
            #select columns that have object type
            if self.data[col_name].dtype == 'O':
                #If they more different values than our threshold then they are surely continous
                if len(set(self.data[col_name])) > self.categorical_threshold:
                    #cast the column to numerical format
                    self.data[col_name] = pd.to_numeric(self.data[col_name], downcast='float', errors='coerce')

    def correct_typos(self,fit=True):
        """
        This method allow the preprocessor to correct typos that may exist in the dataset as some categorical values
        could be repeited with an error ex: "no problem" and "no problem \t"
        :param fit: this parameter makes the distinction between fitting phase as we transform dataset and where we should store
        information and the transformation of the test dataset where we should use the stored infos
        :return:
        """
        if fit==True:

            for col_name in self.data.columns:
                self.typos_data[col_name]=[]
                if self.data[col_name].dtype == 'O':
                    # for each column if the column is categorical we check the different values
                    values_taken = list(set(self.data[col_name]))
                    values_taken=list(filter(None, values_taken))
                    #We compute Levenstein distance between each string corresponding to different values
                    #typos should have a typical distance of 1
                    distance_matrix = [[self.LD(values_taken[i], values_taken[j]) for i in range(len(values_taken))] for j in
                                       range(len(values_taken))]
                    #We localize the typos by looking at the distances
                    typos = np.where((np.array(distance_matrix) > 0) & (np.array(distance_matrix) < 2))
                    typos = [point for point in zip(typos[0], typos[1]) if point[0] < point[1]]
                    for typo in typos:
                        values_to_exchange = values_taken[typo[0]], values_taken[typo[1]]
                        #We store the typo information for the test dataset
                        self.typos_data[col_name].append(values_to_exchange)
                        #we replace each occurence of the right handside of the typo by the other one
                        self.data[col_name][self.data[col_name] == values_to_exchange[0]] = values_to_exchange[1]
                        print("Typo corrected")
        else:
            #for the test dataset we use the stored typo pairs for the correction
            for col_name in self.data.columns:
                if self.data[col_name].dtype == 'O':
                    #we retrieve the typo pairs in the typos_data dictionary
                    for values_to_exchange in self.typos_data[col_name]:
                        #We do the correction
                        self.data[col_name][self.data[col_name] == values_to_exchange[0]] = values_to_exchange[1]


    def encode_categorical_columns(self):
        """
        This method is responsible for turning the categorical columns to a suitable encoding for machine learning
        algorithms
        :return:
        """
        #We search for all the cateogrical columns in the dataframe
        categorical_columns = self.find_categorical_columns()
        #We encode in the desired encoding
        if self.categorical_encoding_stategy == "label":
            for col_name in categorical_columns:
                self.data[col_name] = self.data[col_name].cat.codes
        elif self.categorical_encoding_stategy == "onehot":
            self.data = pd.get_dummies(self.data, columns=categorical_columns, prefix=categorical_columns)

    def find_categorical_columns(self):
        """
        This method find all the categorical columns in the dataframe
        :return:
        """
        categorical_columns = []
        for col_name in self.data.columns:
            if self.data[col_name].dtype == 'O':
                self.data[col_name] = self.data[col_name].astype('category')
                categorical_columns.append(col_name)
        return categorical_columns

    def find_incomplete_columns(self):
        """
        This method finds all columns where there are missing values
        :return:
        """
        incomplete_columns = []
        for col_name in self.data.columns:
            if np.sum(self.data[col_name].isna()) > 0:
                incomplete_columns.append(col_name)
        return incomplete_columns

    def complete_column(self, col_name, missing_strategy="mean",fit=True):
        """
        this method allows to inpute the missing values in a column given as input
        :param col_name:The name of the column where we should input the data
        :param missing_strategy: the strategy that is used to input missing values see __init__() doc for more details
        :param fit:this parameter makes the distinction between fitting phase as we transform dataset and where we should store
        information and the transformation of the test dataset where we should use the stored infos
        :return:
        """
        if fit==True:
            #We initialize the data to be stored during the fitting phase and to be used for the test dataset
            column_data=None
            #We input values in the continous columns first
            if self.data[col_name].dtype != "O":
                #We compute the desired statistic and replace all Nans with it
                if missing_strategy == "mean":
                    #Even if the column is not missing any value we compute and store the statistic if the test column is missing any value
                    mean = np.mean(self.data[col_name].iloc[np.where(self.data[col_name].isna() == False)[0]])
                    column_data=mean
                    #if a value is effectively missing we replace it
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data[col_name].iloc[np.where(self.data[col_name].isna() == True)[0]] = mean
                elif missing_strategy == "median":
                    #Even if the column is not missing any value we compute and store the statistic if the test column is missing any value
                    median = np.median(self.data[col_name].iloc[np.where(self.data[col_name].isna() == False)[0]])
                    column_data=median
                    # if a value is effectively missing we replace it
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data[col_name].iloc[np.where(self.data[col_name].isna() == True)[0]] = median
                elif missing_strategy == "radical":
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data = self.data.drop(columns=[col_name])
            #We then take care of the categorical columns
            else:
                if missing_strategy == "mean" or missing_strategy == "median":
                    #Even if the column is not missing any value we compute and store the statistic if the test column is missing any value
                    available_data = list(self.data[col_name].iloc[np.where(self.data[col_name].isna() == False)[0]].values)
                    most_found = max(set(available_data), key=available_data.count)
                    column_data=most_found
                    # if a value is effectively missing we replace it
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data[col_name].iloc[np.where(self.data[col_name].isna() == True)[0]] = most_found
                elif missing_strategy == "radical":
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data = self.data.drop(columns=[col_name])
            #We store the information in the columns_data dictionary
            self.columns_data[col_name]=column_data
        else:
            # for the test dataset we look for missing values and replace it according to the stored information in columns_data
            if self.data[col_name].dtype != "O":
                if missing_strategy == "mean":
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data[col_name].iloc[np.where(self.data[col_name].isna() == True)[0]] = self.columns_data[col_name]
                elif missing_strategy == "median":
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data[col_name].iloc[np.where(self.data[col_name].isna() == True)[0]] = self.columns_data[col_name]
                elif missing_strategy == "radical":
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data = self.data.drop(columns=[col_name])
            else:
                if missing_strategy == "mean" or missing_strategy == "median":
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data[col_name].iloc[np.where(self.data[col_name].isna() == True)[0]] = self.columns_data[col_name]
                elif missing_strategy == "radical":
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data = self.data.drop(columns=[col_name])


    def infer_missing_data(self,fit=True):
        """
        this method i used to input missing data in all the columns of the dataset
        :param fit:this parameter makes the distinction between fitting phase as we transform dataset and where we should store
        information and the transformation of the test dataset where we should use the stored infos
        :return:
        """
        incomplete_columns = self.find_incomplete_columns()
        for col_name in incomplete_columns:
            self.complete_column(col_name=col_name, missing_strategy=self.missing_stategy,fit=fit)

    """@Author Marouane Maachou"""

    def LD(self,s, t):
        """
        This method computes the Levenstein distance between two strings given as input
        :return: an integer representing the Levenstein distanc between s and t
        """
        if s == "":
            return len(t)
        if t == "":
            return len(s)
        if s[-1] == t[-1]:
            cost = 0
        else:
            cost = 1

        res = min([self.LD(s[:-1], t) + 1,
                   self.LD(s, t[:-1]) + 1,
                   self.LD(s[:-1], t[:-1]) + cost])

        return res

    def transform(self,X,y=None):
        """
        This method transforms a dataframe using the current preprocessor
        :param X: the design matrix dataframe
        :param y: the labels dataframe
        :return:
        """
        #We concatenate the data and the labels to process them at once
        data=pd.concat([X, y], axis=1, sort=False)
        self.data = data
        #We find continuous columns
        self.find_continuous_columns()
        print("Data Preprocessing for the test data is starting")
        #We infer missing values and correct typos using the information that was stored during the fit
        print("We infer missing data")
        self.infer_missing_data(fit=False)
        print("Inference of missing data is Done")
        print("We correct typos in the data")
        self.correct_typos(fit=False)
        print("Typos correction is Done")
        #We encode the categorical columns
        print("We encode the categorical columns")
        self.encode_categorical_columns()
        print("Columns encoding is Done")
        print("Test Data preprocessing is done ")
        return self.data[self.columns[:-1]],self.data[self.columns[-1]]
    def fit_transform(self,X,y):
        """
        This method is used to fit a preprocessor to the train dataset
        :param X: the design matrix dataframe
        :param y: the labels dataframe
        :return:
        """
        #We concatenate the data and the labels to process them at once
        data=pd.concat([X, y], axis=1, sort=False)
        #We construct the dataframe and initiliaze the information storing dictionaries
        self.data = data
        self.columns=self.data.columns
        self.columns_data={key: None for key in self.columns}
        self.typos_data={key: None for key in self.columns}
        #We find continous columns in the dataframe
        self.find_continuous_columns()
        #We infer missing data and storing the information in self.columns_data
        print("Data Preprocessing for the train data is starting")
        print("We infer missing data")
        self.infer_missing_data(fit=True)
        print("Inference of missing data is Done")
        #We correct typos in the dataframe and store information in self.typos_data
        print("We correct typos in the data")
        self.correct_typos(fit=True)
        print("Typos correction is Done")
        #We encode the categorical columns
        print("We encode the categorical columns")
        self.encode_categorical_columns()
        if self.categorical_encoding_stategy=="onehot":
            self.columns=self.data.columns
        print("Columns encoding is Done")
        print("Train Data preprocessing is done ")
        return self.data[self.columns[:-1]], self.data[self.columns[-1]]

class DataNormaliser:
    """
    This class implements a data normaliser that is responsible for normalising the dataset
    """
    def __init__(self, norm_strategy: str = "Mean"):
        """
        :param norm_strategy:  the strategy that is used to normalize the dataset we can either use MeanScaling or MinMaxScaling
        """
        self.offset_parameter = None
        self.scale_parameter = None
        self.norm_strategy = norm_strategy

    def fit(self, X):
        # print("Fitting the Normalizer")
        if self.norm_strategy not in ["MinMax", "Mean"]:
            raise Exception("Choose norm_strategy between MinMax or Mean")
        if self.norm_strategy == "Mean":
            self.offset_parameter = X.mean()
            self.scale_parameter = X.std()
        if self.norm_strategy == "MinMax":
            self.offset_parameter = X.min()
            self.scale_parameter = (X.max() - X.min())

    def transform(self, X):
        # print("We normalize the data")
        return (X - self.offset_parameter) / self.scale_parameter


class FeatureSelector:
    #Class used to select features
    def __init__(self, output_dimension: int = None, variance_treshhold: float = 0.95):
        if variance_treshhold > 1:
            raise Exception("variance_treshhold must be less than 1")
        self.output_dimension = output_dimension
        self.variance_treshhold = variance_treshhold #variance_explained kept in percent
        self.eigenvectors = None
        self.eigenvalues = None

    def fit(self, X_df):
        #Takes pandas X dataframe as input and prepares for PCA transformation
        X = X_df.to_numpy()
        cov = np.cov(X, rowvar=False) #covariance matrix of our samples
        u, s, _ = np.linalg.svd(cov) #apply singular value decomposition to get eigenvalues and vectors of covariance matrix
        self.eigenvalues = s #store results
        self.eigenvectors = u
        print(len(s), "componenents (PCA)")


    def transform(self, X_df):
        X = X_df.to_numpy()
        if self.output_dimension == None: #if user does not provide output dimension use variance_treshhold as a criterion to apply pca
            explained_variance = 0
            total = sum(self.eigenvalues)
            compteur = 0
            while explained_variance < self.variance_treshhold:
                explained_variance = explained_variance + self.eigenvalues[compteur] / total
                compteur = compteur + 1
            U_k = self.eigenvectors[:, :compteur]  # (n x k)
            X_reduced = np.dot(X, U_k)
            print("data reduced to %d dimensions" % (compteur))
            return X_reduced
        else:
            if np.shape(X)[1] < self.output_dimension:
                raise Exception("Output Dimension > Input Dimension")
            U_k = self.eigenvectors[:, :self.output_dimension]  # (n x k)
            X_reduced = np.dot(X, U_k)
            print("data reduced to %d dimensions" % (self.output_dimension))
            return X_reduced


class Model:

    def __init__(self, model, metric: str = "accuracy"):
        self.model = model
        self.metric = metric
        self.best_score_ = None
        self.best_params_ = None

    """@Author Reda Bensaid"""

    def cross_validation_score(self, X_train: np.array, y_train: np.array, k=10):
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
            if self.metric == "accuracy":
                evaluation.append(self._accuracy(y_test_fold, y_pred))
            elif self.metric =="f1_score":
                evaluation.append(self._f1_score(y_test_fold, y_pred))
            else:
                raise Exception("Choose metric between accuracy and f1_score")
        return np.mean(evaluation)

    """@Author Moad Taoufik"""

    def _generateGrid(self, params):
        keys, values = zip(*params.items())
        grid = []
        for v in product(*values):
            d = dict(zip(keys, v))
            grid.append(d)
        return grid

    """@Author Mouad Jallouli"""

    def gridsearchCV(self, X, y, params):
        """
          Implements a grid search cross validation on the training dataset
          The best found parameters and the best score are saved as a class attributes.

        """
        params_grid = self._generateGrid(params)
        best_param = None
        best_score = 0
        for param in params_grid:
            self.model.set_params(**param)
            score = self.cross_validation_score(X, y)
            if best_score == 1:
                break
            if score > best_score:
                best_score = score
                best_param = param
        self.best_score_ = best_score
        self.best_params_ = best_param

    """@Author Mouad Jallouli"""

    @staticmethod
    def _f1_score(y_true, y_predict):
        p = 0
        r = 0
        f = 0
        n_classes = y_true.nunique()
        for i in range(n_classes):
            tp = sum(((y_true == i) & (y_predict == i)))
            fp = sum(((y_true != i) & (y_predict == i)))
            fn = sum(((y_true == i) & (y_predict != i)))
            if fp == 0:  # edge case
                p = 1
            else:
                p = tp / (tp + fp)
            if fn == 0:  # edge case
                r = 1
            else:
                r = tp / (tp + fn)
            if p ==0 and r ==0:
                f += 0
            else:
                f += 2 * p * r / (p + r)

        return f / n_classes

    """@Author Mouad Jallouli"""

    """@Author Mouad Jallouli"""

    @staticmethod
    def score_matrix(y_true, y_predict):
        """
           Returns pd.DataFrame with precision, recall and f1 score for each class
          :params:
                y_true: array-like, true target
                y_predict: array-like, predicted target

        """
        p, r, f, a = 0, 0, 0, 0
        D = {"Accuracy": {}, "Precision": {}, "Recall": {}, "F1_score": {}}
        n_classes = y_true.nunique()
        for i in range(n_classes):
            tp = sum(((y_true == i) & (y_predict == i)))
            fp = sum(((y_true != i) & (y_predict == i)))
            fn = sum(((y_true == i) & (y_predict != i)))

            a = np.mean((y_true == y_predict))
            if fp == 0:  # edge case
                p = 1
            else:
                p = tp / (tp + fp)
            if fn == 0:  # edge case
                r = 1
            else:
                r = tp / (tp + fn)
            D["Precision"]["class " + str(i)] = p
            D["Recall"]["class " + str(i)] = r
            D["F1_score"]["class " + str(i)] = 2 * p * r / (p + r)
            D["Accuracy"]["class " + str(i)] = a

        return pd.DataFrame(D)

    """@Author Reda Bensaid"""

    @staticmethod
    def _accuracy(y_true, y_predict):
        """
           this method calculate the accuracy of the prediction
           :param y_true: the labels
           :param y_predict: the predictions
           :return: accuracy of the prediction
        """
        good_predictions = np.sum((y_true == y_predict) * 1)
        return good_predictions / len(y_true)


if __name__ == "__main__":

    import sys
    arguments = sys.argv[1:]
    if len(arguments) < 9:
        for i in range(6):
            arguments = arguments + [""]

    """
    arguments[0] :str data(csv)  path
    arguments[1] :str target column
    arguments[2]  :str  True or False wether dataframe has header with column names or not
    arguments[3] :str test size < 1
    arguments[4] :str missing data filling strategy ["","mean", "median","radical"]
    arguments[5] :str categorical features encoding ["","label","onehot"]
    arguments[6] :str normalisation_strategy ["","Mean", "MinMax"]
    arguments[7] :str pca parameters ["variance treshhold (=< 1)", "output_dimension(>1)"]
    arguments[8] :str evaluation metric ["","accuracy","f1_score"]
        
    """
    tic = time.time()
    dict_args = {}

    data_path = arguments[0]
    try:
        target_column = int(arguments[1])
    except ValueError:
        target_column = arguments[1]

    if len(arguments[2]) > 0:
        dict_args["header"] = arguments[2] == "True"
    if len(arguments[3]) > 0:
        dict_args["test_size"] = float(arguments[3])

    data_loader = DataLoader(data_path = data_path, **dict_args)
    data_loader.load(target_column = target_column)
    X_train, y_train, X_test, y_test = data_loader.X_train, data_loader.y_train,data_loader.X_test, data_loader.y_test

    dict_args = {}

    if len(arguments[4]) > 0:
        dict_args["missing_data_strategy"] = arguments[4]

    if len(arguments[5]) > 0:
        dict_args["categorical_encoding"] = arguments[5]

    data_preprocessor = DataPreprocessor(**dict_args)

    X_train, y_train = data_preprocessor.fit_transform(X_train, y_train)
    X_test, y_test = data_preprocessor.transform(X_test, y_test)

    dict_args = {}
    if len(arguments[6]) > 0:
        dict_args["norm_strategy"] = arguments[6]

    data_normaliser = DataNormaliser(**dict_args)
    data_normaliser.fit(X_train)
    X_train = data_normaliser.transform(X_train)
    X_test = data_normaliser.transform(X_test)

    dict_args = {}
    if len(arguments[7]) > 0:
        if float(arguments[7]) > 1:
            dict_args["output_dimension"] = int(arguments[7])
        elif float(arguments[7]) <= 1:
            dict_args["variance_treshhold"] = float(arguments[7])

    feature_selector = FeatureSelector(**dict_args)
    feature_selector.fit(X_train)
    X_train = feature_selector.transform(X_train)
    X_test = feature_selector.transform(X_test)





    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier



    dict_models = {}
    dict_models[DecisionTreeClassifier()] = {'min_samples_split': range(10, 500, 20), 'max_depth': range(1, 20, 2), 'criterion': ['gini', 'entropy']}
    dict_models[KNeighborsClassifier()] = {'n_neighbors': [4, 5, 6, 7], 'leaf_size': [1, 3, 5], 'weights': ['uniform', 'distance']}
    dict_models[SVC()] = {'C': [0.1, 1, 10], 'gamma': ['auto','scale'], 'kernel': ['rbf']}
    dict_models[RandomForestClassifier()] = {'n_estimators': [200, 500], 'max_features': ['auto', 'log2'], 'max_depth': [4, 5, 6, 7, 8],
                        'criterion': ['gini', 'entropy']}
    dict_models[LogisticRegression()] = {'C': [0.001, .009, 0.01, .09, 1, 5, 10, 25]}


    dict_args = {}
    if len(arguments[8]) > 0:
        dict_args["metric"] = arguments[8]

    dict_score = {}
    for model, grid in dict_models.items():
        m = Model(model,**dict_args)
        m.gridsearchCV(X_train,y_train,grid)
        best_param = m.best_params_
        best_score = m.best_score_
        dict_score[model] = [best_param, best_score]

    best_score = 0
    for key,value in dict_score.items():
        print("\n")
        print(type(key).__name__)
        print("Chosen parameters:", value[0])
        print("CrossValidation Score %(score)f" %{"score": value[1]})
        print("\n")
        #We search for best value
        if value[1] > best_score:
            best_score = value[1]
            chosen_model = key

    chosen_params = dict_score[chosen_model][0]
    chosen_model.set_params(**chosen_params)
    chosen_model.fit(X_train, y_train)
    y_predict = chosen_model.predict(X_test)
    accuracy = m._accuracy(y_test, y_predict)
    f1_score = m._f1_score(y_test, y_predict)
    score_df = m.score_matrix(y_test,y_predict)
    model_name = type(chosen_model).__name__
    print("Chosen model is " + model_name)
    print(score_df)
    print("\n")
    print("accuracy = %(accuracy)f \n f1_score = %(f1_score)f" %{"accuracy": accuracy, "f1_score": f1_score})
    print(" \n SK LEARN score \n")
    print(chosen_model.score(X_test,y_test))
    toc  = time.time()
    delay = toc - tic
    print("run in : ",  delay, "seconds")

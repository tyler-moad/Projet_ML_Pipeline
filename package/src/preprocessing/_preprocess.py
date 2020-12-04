

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

"""@Author Marouane Maachou"""
def LD(s, t):
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

    res = min([LD(s[:-1], t) + 1,
               LD(s, t[:-1]) + 1,
               LD(s[:-1], t[:-1]) + cost])

    return res

"""@Author Moad Taoufik"""
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
        print("Fitting the Normalizer")
        if self.norm_strategy not in ["MinMax", "Mean"]:
            raise Exception("Choose norm_strategy between MinMax or Mean")
        if self.norm_strategy == "Mean":
            self.offset_parameter = X.mean()
            self.scale_parameter = X.std()
        if self.norm_strategy == "MinMax":
            self.offset_parameter = X.min()
            self.scale_parameter = (X.max() - X.min())

    def transform(self, X):
        print("We normalize the data")
        return (X - self.offset_parameter) / self.scale_parameter









"""@Author Marouane Maachou"""
class DataPreprocessor:
    """
    This class implements a data preprocessor that is responsible for cleaning the dataset by input NaN values ,
    correcting typos in the dataset and encoding it in a suitable categorical format for machine learning algorithms
    """
>>>>>>> 617a12355c7da5c8be950fc00dc57b7c768b2d3d

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
                    distance_matrix = [[LD(values_taken[i], values_taken[j]) for i in range(len(values_taken))] for j in
                                       range(len(values_taken))]
                    #We localize the typos by looking at the distances
                    typos = np.where((np.array(distance_matrix) > 0) & (np.array(distance_matrix) < 2))
                    typos = [point for point in zip(typos[0], typos[1]) if point[0] < point[1]]
                    for typo in typos:
                        values_to_exchange = values_taken[typo[0]], values_taken[typo[1]]
                        #We store the typo information for the test dataset
                        self.typos_data[col_name].append(values_to_exchange)
                        print("Typo detected on column :",col_name,"betwen ",values_to_exchange[0],"and ",values_to_exchange[1])
                        #we replace each occurence of the right handside of the typo by the other one
                        self.data[col_name][self.data[col_name] == values_to_exchange[0]] = values_to_exchange[1]
                        print("Typo corrected")
        else:
            #for the test dataset we use the stored typo pairs for the correction
            for col_name in self.data.columns:
                if self.data[col_name].dtype == 'O':
                    print("Correction typos for test dataset for ",col_name)
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




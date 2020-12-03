import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:41:18 2020

@author: TheMatrix
"""
import numpy as np
import pandas as pd


class DataNormaliser:
    def __init__(self, norm_strategy: str = "Mean"):
        self.offset_parameter = None
        self.scale_parameter = None
        self.norm_strategy = norm_strategy

    def fit(self, X):
        if self.norm_strategy not in ["MinMax", "Mean"]:
            raise Exception("Choose norm_strategy between MinMax or Mean")
        if self.norm_strategy == "Mean":
            self.offset_parameter = X.mean()
            self.scale_parameter = X.std()
        if self.norm_strategy == "MinMax":
            self.offset_parameter = X.min()
            self.scale_parameter = (X.max() - X.min())

    def transform(self, X):
        return (X - self.offset_parameter) / self.scale_parameter




class DataPreprocessor(TransformerMixin):

    def __init__(self, data_path=None, missing_data_strategy: str = "mean", categorical_encoding: str = "label",
                 test_size: float = 0.1, validation_size=0.1, categorical_threshold=15):

        if missing_data_strategy in ["mean", "median", "radical"]:
            self.missing_stategy = missing_data_strategy
        else:
            raise ValueError("Missing data strategy not in allowed methods ")

        if categorical_encoding in ["label", "onehot"]:
            self.categorical_encoding_stategy = categorical_encoding
        else:
            raise ValueError("catagorical encoding strategy not in allowed methods ")

        self.test_size = test_size
        self.validation_size = validation_size
        # self.data = pd.read_csv(data_path)
        self.categorical_threshold = categorical_threshold

    def find_continuous_columns(self):
        for col_name in self.data.columns:
            if self.data[col_name].dtype == 'O':
                print(col_name)
                if len(set(self.data[col_name])) > self.categorical_threshold:
                    # self.data[col_name].to_numeric(convert_numeric=True)
                    self.data[col_name] = pd.to_numeric(self.data[col_name], downcast='float', errors='coerce')

    def correct_typos(self):
        for col_name in self.data.columns:
            print(col_name)
            if self.data[col_name].dtype == 'O':
                values_taken = list(set(self.data[col_name]))
                distance_matrix = [[LD(values_taken[i], values_taken[j]) for i in range(len(values_taken))] for j in
                                   range(len(values_taken))]
                typos = np.where((np.array(distance_matrix) > 0) & (np.array(distance_matrix) < 2))
                typos = [point for point in zip(typos[0], typos[1]) if point[0] < point[1]]
                for typo in typos:
                    values_to_exchange = values_taken[typo[0]], values_taken[typo[1]]
                    print(values_to_exchange)
                    self.data[col_name][self.data[col_name] == values_to_exchange[0]] = values_to_exchange[1]

    def encode_categorical_columns(self):
        categorical_columns = self.find_categorical_columns()
        if self.categorical_encoding_stategy == "label":
            for col_name in categorical_columns:
                self.data[col_name] = self.data[col_name].cat.codes
        elif self.categorical_encoding_stategy == "onehot":
            self.data = pd.get_dummies(self.data, columns=categorical_columns, prefix=categorical_columns)

    def find_categorical_columns(self):
        categorical_columns = []
        for col_name in self.data.columns:
            if self.data[col_name].dtype == 'O':
                self.data[col_name] = self.data[col_name].astype('category')
                categorical_columns.append(col_name)
        return categorical_columns

    def find_incomplete_columns(self):
        incomplete_columns = []
        for col_name in self.data.columns:
            if np.sum(self.data[col_name].isna()) > 0:
                incomplete_columns.append(col_name)
        return incomplete_columns

    def complete_column(self, col_name, missing_strategy="mean"):
        if np.sum(self.data[col_name].isna()) > 0:
            if self.data[col_name].dtype != "O":
                if missing_strategy == "mean":
                    mean = np.mean(self.data[col_name][np.where(self.data[col_name].isna() == False)[0]])
                    self.data[col_name][np.where(self.data[col_name].isna() == True)[0]] = mean
                elif missing_strategy == "median":
                    median = np.median(self.data[col_name][np.where(self.data[col_name].isna() == False)[0]])
                    self.data[col_name][np.where(self.data[col_name].isna() == True)[0]] = median
                elif missing_strategy == "radical":
                    self.data = self.data.drop(columns=[col_name])
            else:
                if missing_strategy == "mean" or missing_strategy == "median":
                    available_data = list(self.data[col_name][np.where(self.data[col_name].isna() == False)[0]].values)
                    most_found = max(set(available_data), key=available_data.count)
                    self.data[col_name][np.where(self.data[col_name].isna() == True)[0]] = most_found
                elif missing_strategy == "radical":
                    self.data = self.data.drop(columns=[col_name])

    def infer_missing_data(self):
        incomplete_columns = self.find_incomplete_columns()
        for col_name in incomplete_columns:
            print(col_name)
            self.complete_column(col_name=col_name, missing_strategy=self.missing_stategy)

    def transform(self,X,y=None):
        self.data = X
        self.find_continuous_columns()
        # self.infer_missing_data()
        self.correct_typos()
        self.encode_categorical_columns()
        return self.data
    def fit(self,X,y):
        # Save used setup
        self.transform(X,y)
        return self




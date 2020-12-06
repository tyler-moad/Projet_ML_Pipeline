

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:41:18 2020

@author: TheMatrix
"""
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

def LD(s, t):
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

    def __init__(self, missing_data_strategy: str = "mean", categorical_encoding: str = "label", categorical_threshold=15):

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
        for col_name in self.data.columns:
            if self.data[col_name].dtype == 'O':
                print(col_name)
                if len(set(self.data[col_name])) > self.categorical_threshold:
                    self.data[col_name] = pd.to_numeric(self.data[col_name], downcast='float', errors='coerce')

    def correct_typos(self,fit=True):
        if fit==True:
            self.typos_data = {}
            for col_name in self.data.columns:
                self.typos_data[col_name]=[]
                print(col_name)
                if self.data[col_name].dtype == 'O':
                    values_taken = list(set(self.data[col_name]))
                    values_taken=list(filter(None, values_taken))
                    print("values_taken",values_taken)
                    distance_matrix = [[LD(values_taken[i], values_taken[j]) for i in range(len(values_taken))] for j in
                                       range(len(values_taken))]
                    typos = np.where((np.array(distance_matrix) > 0) & (np.array(distance_matrix) < 2))
                    typos = [point for point in zip(typos[0], typos[1]) if point[0] < point[1]]
                    for typo in typos:
                        values_to_exchange = values_taken[typo[0]], values_taken[typo[1]]
                        self.typos_data[col_name].append(values_to_exchange)
                        print(values_to_exchange)
                        self.data[col_name][self.data[col_name] == values_to_exchange[0]] = values_to_exchange[1]
        else:
            for col_name in self.data.columns:
                if self.data[col_name].dtype == 'O':
                    print("correcting",col_name)
                    for values_to_exchange in self.typos_data[col_name]:
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

    def complete_column(self, col_name, missing_strategy="mean",fit=True):
        if fit==True:
            column_data=None
            if self.data[col_name].dtype != "O":
                if missing_strategy == "mean":
                    mean = np.mean(self.data[col_name].iloc[np.where(self.data[col_name].isna() == False)[0]])
                    column_data=mean
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data[col_name].iloc[np.where(self.data[col_name].isna() == True)[0]] = mean
                elif missing_strategy == "median":
                    median = np.median(self.data[col_name].iloc[np.where(self.data[col_name].isna() == False)[0]])
                    column_data=median
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data[col_name].iloc[np.where(self.data[col_name].isna() == True)[0]] = median
                elif missing_strategy == "radical":
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data = self.data.drop(columns=[col_name])
            else:
                if missing_strategy == "mean" or missing_strategy == "median":
                    available_data = list(self.data[col_name].iloc[np.where(self.data[col_name].isna() == False)[0]].values)
                    most_found = max(set(available_data), key=available_data.count)
                    column_data=most_found
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data[col_name].iloc[np.where(self.data[col_name].isna() == True)[0]] = most_found
                elif missing_strategy == "radical":
                    if np.sum(self.data[col_name].isna()) > 0:
                        self.data = self.data.drop(columns=[col_name])
            self.columns_data[col_name]=column_data
        else:
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
        incomplete_columns = self.find_incomplete_columns()
        for col_name in incomplete_columns:
            print(col_name)
            self.complete_column(col_name=col_name, missing_strategy=self.missing_stategy,fit=fit)

    def transform(self,X,y=None):
        data=pd.concat([X, y], axis=1, sort=False)
        self.data = data
        self.find_continuous_columns()
        self.infer_missing_data(fit=False)
        self.correct_typos(fit=False)
        self.encode_categorical_columns()
        return self.data[self.columns[:-1]],self.data[self.columns[-1]]
    def fit_transform(self,X,y):
        # Save used setup
        data=pd.concat([X, y], axis=1, sort=False)
        self.data = data
        self.columns=self.data.columns
        self.columns_data={key: None for key in self.columns}
        self.find_continuous_columns()
        self.infer_missing_data(fit=True)
        self.correct_typos(fit=True)
        self.encode_categorical_columns()
        if self.categorical_encoding_stategy=="onehot":
            self.columns=self.data.columns
        return self.data[self.columns[:-1]], self.data[self.columns[-1]]




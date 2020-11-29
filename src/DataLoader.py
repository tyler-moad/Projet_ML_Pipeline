import numpy as np
import pandas as pd


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

class DataLoader:

    def __init__(self, data_path, missing_data_strategy: str = "mean", categorical_encoding: str = "label",
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
        self.data = pd.read_csv(data_path)
        self.categorical_threshold = categorical_threshold

    def find_continuous_columns(self):
        for col_name in self.data.columns:
            if self.data[col_name].dtype == 'O':
                print(col_name)
                if len(set(self.data[col_name])) > self.categorical_threshold:
                    self.data[col_name] = pd.to_numeric(self.data[col_name], downcast='float', errors='coerce')

    def correct_typos(self):
        for col_name in self.data.columns:
            if data[col_name].dtype == 'O':
                values_taken = list(set(self.data[col_name]))

                distance_matrix = [[LD(values_taken[i], values_taken[j]) for i in range(len(values_taken))] for j in
                                   range(len(values_taken))]
                typos = np.where((np.array(matrix) > 0) & (np.array(matrix) < 3))
                typos = [point for point in zip(typos[0], typos[1]) if point[0] < point[1]]
                for typo in typos:
                    values_to_exchange = values_taken[typo[0]], values_taken[typo[1]]
                    print(values_to_exchange)
                    data["classification"][self.data["classification"] == values_to_exchange[0]] = values_to_exchange[1]

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
        if self.data[col_name].dtype != "O":
            if missing_strategy == "mean":
                mean = np.mean(self.data[col_name][np.where(self.data[col_name].isna() == False)[0]])
                self.data[col_name][np.where(self.data[col_name].isna() == True)[0]] = mean
        else:
            if missing_strategy == "mean":
                available_data = list(self.data[col_name][np.where(self.data[col_name].isna() == False)[0]].values)
                most_found = max(set(available_data), key=available_data.count)
                self.data[col_name][np.where(self.data[col_name].isna() == True)[0]] = most_found

    def infer_missing_data(self):
        incomplete_columns = self.find_incomplete_columns()
        for col_name in incomplete_columns:
            print(col_name)
            self.complete_column(col_name=col_name, missing_strategy=self.missing_stategy)
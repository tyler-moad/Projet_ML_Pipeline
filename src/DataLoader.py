import numpy as np
import pandas as pd


class DataLoader:

    def __init__(self, data_path, missing_data_strategy: str = "mean", categorical_encoding: str = "label",
                 test_size: float = 0.1, validation_size=0.1):

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

    def encode_categorical_columns(self):
        categorical_columns = self.find_categorical_columns()
        if self.categorical_encoding_stategy == "label":
            for col_name in categorical_columns:
                self.data[col_name] = self.data[col_name].cat.codes
        elif self.categorical_encoding_stategy == "onehot":
            pd.get_dummies(self.data, columns=categorical_columns, prefix=categorical_columns).head()

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

        if missing_strategy == "mean":
            mean = np.mean(self.data[col_name][np.where(self.data[col_name].isna() == False)[0]])
            self.data[col_name][np.where(self.data[col_name].isna() == True)[0]] = mean

    def infer_missing_data(self):
        incomplete_columns = self.find_incomplete_columns()
        for col_name in incomplete_columns:
            print(col_name)
            self.complete_column(col_name=col_name, missing_strategy=self.missing_stategy)

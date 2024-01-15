import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import logging

class DataPreprocessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        logging.info("DataPreprocessor inicializado.")

    def encode_gender(self):
        logging.info("Codificando la columna de género.")
        le = LabelEncoder()
        self.dataframe['Gender'] = le.fit_transform(self.dataframe['Gender'])

    def one_hot_encode_columns(self, columns):
        logging.info(f"Aplicando One-Hot Encoding a las columnas: {columns}")
        for column in columns:
            ohe = OneHotEncoder()
            transformed_data = ohe.fit_transform(self.dataframe[[column]]).toarray()
            columns_names = [f"{column}_{category}" for category in ohe.categories_[0]]
            new_df = pd.DataFrame(transformed_data, columns=columns_names)
            self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
            self.dataframe.drop(column, axis=1, inplace=True)

    def scale_columns(self, columns):
        logging.info(f"Escalando las columnas: {columns}")
        scaler = StandardScaler()
        for column in columns:
            self.dataframe[column] = scaler.fit_transform(self.dataframe[[column]])

    def categorize_age(self):
        logging.info("Categorizando la columna de edad.")
        bins = [0, 30, 60, 100]
        labels = ['joven', 'adulto', 'mayor']
        self.dataframe['Age'] = pd.cut(self.dataframe['Age'], bins=bins, labels=labels, include_lowest=True)

    def booleanize_tenure(self):
        logging.info("Booleanizando la columna Tenure.")
        self.dataframe['Tenure'] = self.dataframe['Tenure'].apply(lambda x: False if x == 0 else True)

    def booleanize_num_of_products(self):
        logging.info("Booleanizando la columna NumOfProducts.")
        self.dataframe['NumOfProducts'] = self.dataframe['NumOfProducts'].apply(lambda x: False if x == 0 else True)

    def one_hot_encode_age(self):
        logging.info("Aplicando One-Hot Encoding a la columna de edad.")
        ohe = OneHotEncoder()
        transformed_data = ohe.fit_transform(self.dataframe[['Age']]).toarray()
        columns_names = [f"Age_{category}" for category in ohe.categories_[0]]
        new_df = pd.DataFrame(transformed_data, columns=columns_names)
        self.dataframe = pd.concat([self.dataframe, new_df], axis=1)
        self.dataframe.drop('Age', axis=1, inplace=True)

    def preprocess_data(self):
        logging.info("Iniciando el preprocesamiento de datos.")
        self.encode_gender()
        self.one_hot_encode_columns(['Geography', 'IsActiveMember', 'HasCrCard'])
        self.scale_columns(['CreditScore', 'Balance', 'EstimatedSalary'])
        self.categorize_age()
        self.booleanize_tenure()
        self.booleanize_num_of_products()
        self.one_hot_encode_age()

logging.info("Preprocesamiento de datos completado.")

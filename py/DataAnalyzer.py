import pandas as pd
import numpy as np
import logging

class DataAnalyzer:
    def __init__(self, dataframe):
        """
        Inicializa la clase DataAnalyzer con un DataFrame.

        :param dataframe: DataFrame de Pandas con los datos.
        """
        self.dataframe = dataframe
        logging.info("DataAnalyzer inicializado.")

    def get_info(self):
        """
        Imprime la información general del DataFrame, incluyendo tipos de datos y valores no nulos.

        :return: None
        """
        logging.info("Obteniendo información del DataFrame.")
        print(self.dataframe.info())

    def get_description(self):
        """
        Imprime las estadísticas descriptivas del DataFrame.

        :return: None
        """
        logging.info("Obteniendo descripción estadística del DataFrame.")
        print(self.dataframe.describe())

    def count_nulls(self):
        """
        Cuenta y devuelve el número de valores nulos en cada columna.

        :return: Serie de Pandas con el recuento de nulos por columna.
        """
        logging.info("Contando valores nulos por columna.")
        return self.dataframe.isnull().sum()

    def analyze_target_balance(self, target_column):
        """
        Analiza el balance de la columna objetivo.

        :param target_column: Nombre de la columna objetivo.
        :return: None
        """
        logging.info(f"Analizando el balance de la columna objetivo: {target_column}")
        value_counts = self.dataframe[target_column].value_counts(normalize=True) * 100
        print(f"Distribución de la columna objetivo '{target_column}':\n{value_counts}")

# Configuración básica del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



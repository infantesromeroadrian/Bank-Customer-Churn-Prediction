from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, dataframe, target_column, exclude_columns=None, test_size=0.3, val_size=0.5):
        """
        Inicializa la clase DataSplitter.

        :param dataframe: DataFrame de Pandas con los datos.
        :param target_column: Nombre de la columna objetivo.
        :param exclude_columns: Lista de columnas para excluir del entrenamiento y prueba.
        :param test_size: Porcentaje del conjunto de datos que será para la prueba.
        :param val_size: Proporción del conjunto de prueba que será para la validación.
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.exclude_columns = exclude_columns
        self.test_size = test_size
        self.val_size = val_size

    def split_data(self):
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba.

        :return: Cuatro DataFrames: X_train, X_val, X_test, y_train, y_val, y_test.
        """
        # Eliminar columnas excluidas si existen
        df = self.dataframe.drop(columns=self.exclude_columns, errors='ignore') if self.exclude_columns else self.dataframe

        # Dividir primero en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(self.target_column, axis=1),
            df[self.target_column],
            test_size=self.test_size,
            random_state=42
        )

        # Dividir el conjunto de prueba en prueba y validación
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test,
            test_size=self.val_size,
            random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
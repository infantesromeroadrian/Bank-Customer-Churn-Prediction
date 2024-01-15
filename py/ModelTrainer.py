import logging
import mlflow
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, X_train, X_val, y_train, y_val, experiment_name):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.best_model = None

        # Configurar MLflow
        mlflow.set_experiment(experiment_name)

    def train_model(self, model, param_distributions, n_iter=10):
        logging.info(f"Iniciando el entrenamiento del modelo: {model.__class__.__name__}")
        try:
            with mlflow.start_run():
                random_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=3, n_jobs=-1, verbose=1, random_state=42)
                random_search.fit(self.X_train, self.y_train)

                logging.info("Modelo entrenado. Registrando resultados...")
                self._log_results(random_search)
                self.best_model = random_search.best_estimator_

        except Exception as e:
            logging.error(f"Error durante el entrenamiento del modelo: {e}")
            raise

    def _log_results(self, random_search):
        # Mejores parámetros y puntuación
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        logging.info(f"Mejores Parámetros: {best_params}")
        logging.info(f"Mejor Puntuación: {best_score}")

        # Evaluación en el conjunto de validación
        y_pred = random_search.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        logging.info(f"Precisión en Validación: {accuracy}")

        # Registrar métricas y parámetros en MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)


from DataLoader import DataLoader
from DataAnalyzer import DataAnalyzer
from DataPreprocessor import DataPreprocessor
from DataSplitter import DataSplitter
from ModelTrainer import ModelTrainer
from ModelVisualizer import ModelVisualizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Cambia la ruta del archivo según tu configuración
file_path = '/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/Kaggle/BankChurn/data/train.csv'

# Cargar y visualizar los datos
data_loader = DataLoader(file_path)
df = data_loader.load_data()
data_loader.preview_data(5)

# Analizar los datos
analyzer = DataAnalyzer(df)
analyzer.get_info()
analyzer.get_description()
analyzer.count_nulls()
analyzer.analyze_target_balance('Exited')

# Preprocesamiento de datos
preprocessor = DataPreprocessor(df)
preprocessor.preprocess_data()

# Dividir los datos
splitter = DataSplitter(preprocessor.dataframe, 'Exited', exclude_columns=['id', 'CustomerId', 'Surname'])
X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data()

# Entrenar el modelo
trainer = ModelTrainer(X_train, X_val, y_train, y_val, "Experimento_BankChurn")

# Definir parámetros para Random Forest y SVM
params_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
params_svc = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

# Entrenar modelos y seleccionar el mejor
trainer.train_model(RandomForestClassifier(), params_rf)
trainer.train_model(SVC(), params_svc)

# Visualizar resultados del mejor modelo
best_model = trainer.best_model
visualizer = ModelVisualizer(y_val, best_model.predict(X_val), "bankchurn_model")
visualizer.plot_confusion_matrix()
visualizer.plot_roc_curve()
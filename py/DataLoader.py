import pandas as pd
import logging

# Configuración básica del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        logging.info(f"DataLoader inicializado con el archivo: {filepath}")

    def load_data(self):
        try:
            data = pd.read_csv(self.filepath)
            logging.info("Datos cargados correctamente.")
            return data
        except FileNotFoundError:
            logging.error("El archivo no fue encontrado.", exc_info=True)
            return None

    def preview_data(self, n=5):
        data = self.load_data()
        if data is not None:
            logging.info(f"Vista previa de los primeros {n} registros:")
            print(data.head(n))
        else:
            logging.warning("No se pudo cargar los datos para la vista previa.")


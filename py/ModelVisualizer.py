import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc


class ModelVisualizer:
    def __init__(self, y_true, y_pred, model_name):
        """
        Inicializa la clase ModelVisualizer.

        :param y_true: Valores verdaderos.
        :param y_pred: Valores predichos por el modelo.
        :param model_name: Nombre del modelo (para etiquetas).
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name

    def plot_confusion_matrix(self):
        try:
            # Calcula la matriz de confusión
            cm = confusion_matrix(self.y_true, self.y_pred)
            cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'],
                                 columns=['Predicted Negative', 'Predicted Positive'])

            # Crea el gráfico
            fig = px.imshow(cm_df, text_auto=True, aspect="auto",
                            labels=dict(x="Predicted Value", y="Actual Value", color="Count"),
                            title=f"Confusion Matrix of {self.model_name}")
            fig.show()
        except Exception as e:
            logging.error(f"Error al generar la matriz de confusión: {e}")
            raise

    def plot_roc_curve(self):
        try:
            # Calcula la curva ROC y el área bajo la curva
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred)
            auc_score = auc(fpr, tpr)

            # Crea el gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

            fig.update_layout(title=f"ROC Curve of {self.model_name} (AUC = {auc_score:.2f})",
                              xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            fig.show()
        except Exception as e:
            logging.error(f"Error al generar la curva ROC: {e}")
            raise
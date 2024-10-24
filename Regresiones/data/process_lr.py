# Regresion Logistica
# 1) Hacer un split data con los datos normalizados de diabetesnorm.csv
# - El 70% de columnas (6) tienen que ser aleatorias asi como el otro 30% (5 Columnas)
# - Dataframe llamado test y train, dividir entre 70 y 30, el 30 es con la ultima columna.
# 2) Generar el modelo de regresion logistica: logistic_regression, con train_data y train_labels
# - Columnas: 1-10 con train_data, 11 para train_labels
# 3) Hacer la evaluacion como: y_pred = model.predict(test)
# - Calcular el accuracy: accuracy_score(y, y_pred)
# 4) Hacer una funcion para cada uno de los pasos.
# 5) Hacer logistic_regression con train_data (1-10 columnas), train_labels (Columna 11)

import os
import joblib
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def check_output_dir():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, 'output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Cargar el dataset de Breast Cancer Wisconsin (Diagnostic)
def load_breast_cancer_data():
    dataset = fetch_ucirepo(id=17)
    
    # Extraer características y etiquetas
    X = dataset.data.features
    y = dataset.data.targets
    
    # Convertir a DataFrame
    data = pd.DataFrame(X)
    data['target'] = y
    
    return data

def normalize_data(data):
    scaler = StandardScaler()
    # Normalizar todas las columnas excepto la última (target)
    data_scaled = scaler.fit_transform(data.iloc[:, :-1])
    norm_data = pd.DataFrame(data_scaled, columns=data.columns[:-1])
    
    # Codificar la variable objetivo (Hacerla binaria)
    le = LabelEncoder()
    norm_data['target'] = le.fit_transform(data['target'])
    
    return norm_data, le

def split_data(data, test_size):
    # Seleccionar 6 columnas aleatorias para entrenamiento y prueba
    feature_columns = list(data.columns[:-1])
    selected_columns = random.sample(feature_columns, 6)
    
    train_data, test_data = train_test_split(data, test_size=test_size)
    
    train_input = train_data[selected_columns]
    train_output = train_data['target']

    test_input = test_data[selected_columns]
    test_output = test_data['target']

    return train_input, train_output, test_input, test_output, selected_columns

def logistic_regression(train_input, train_output):
    model = LogisticRegression()
    model.fit(train_input, train_output)
    return model

def get_accuracy(model, test_input, test_output):
    predictions = model.predict(test_input)
    return accuracy_score(test_output, predictions)

def get_confusion_matrix(model, test_input, test_output):
    predictions = model.predict(test_input)
    return confusion_matrix(test_output, predictions)

def save_model(model, output_dir):
    model_path = os.path.join(output_dir, 'logistic_regression_model.pkl')
    joblib.dump(model, model_path)
    print(f"Modelo guardado como '{model_path}'")

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def plot_roc_curve(model, test_input, test_output, output_dir):
    y_pred_proba = model.predict_proba(test_input)[:, 1]
    fpr, tpr, _ = roc_curve(test_output, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    print("Curva ROC guardada como 'roc_curve.png'")
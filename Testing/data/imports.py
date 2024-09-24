# Regresion Lineal
# 1) Hacer un split data con los datos normalizados de diabetesnorm.csv
# - El 70% de columnas (6) tienen que ser aleatorias asi como el otro 30% (5 Columnas)
# - Dataframe llamado test y train, dividir entre 70 y 30, el 30 es con la ultima columna.
# 2) Generar el modelo de regresion lineal: linear_regression, con train_data y train_labels
# - Columnas: 1-10 con train_data, 11 para train_labels
# 3) Hacer la evaluacion como: y_pred = model.predict(test)
# - Calcular el RMSE y MSE: MSE(y, y_pred), RMSE(y, y_pred)
# 4) Hacer una funcion para cada uno de los pasos.
# 5) Hacer linear_regression con train_data (1-10 columnas), train_labels (Columna 11)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib  # Para guardar y cargar el modelo

# Función para crear el directorio de salida si no existe
def check_output_dir():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, 'output')
    
    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Función para cargar y dividir los datos
def load_and_split_data(file_path, test_size=0.3, random_state=42):
    # Cargar los datos normalizados desde el archivo CSV
    df = pd.read_csv(file_path)
    
    # Separar las características (X) y la variable objetivo (y)
    X = df.iloc[:, :-1]  # Todas las columnas excepto la última
    y = df.iloc[:, -1]   # La última columna
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Función para generar y entrenar el modelo de regresión lineal, si no existe
def train_linear_regression(X_train, y_train):
    output_dir = check_output_dir()
    model_path = os.path.join(output_dir, 'linear_regression_model.pkl')
    
    # Verificar si el modelo ya existe
    if os.path.exists(model_path):
        print("Cargando modelo existente desde", model_path)
        model = joblib.load(model_path)
    else:
        # Crear el modelo de regresión lineal
        model = LinearRegression()
        
        # Entrenar el modelo con los datos de entrenamiento
        model.fit(X_train, y_train)
        
        # Guardar el modelo en la carpeta 'output'
        joblib.dump(model, model_path)
        print("Modelo entrenado y guardado en", model_path)
    
    return model

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_test):
    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular el Error Cuadrático Medio (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calcular la Raíz del Error Cuadrático Medio (RMSE)
    rmse = np.sqrt(mse)
    
    return mse, rmse, y_pred

# Función para visualizar los resultados
def plot_results(y_test, y_pred):
    # Obtener el directorio de salida
    output_dir = check_output_dir()
    
    # Configurar y guardar el gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Valores reales vs Predicciones')
    plt.tight_layout()
    
    # Guardar el gráfico en el directorio 'output'
    output_file = os.path.join(output_dir, 'regression_results.png')
    plt.savefig(output_file)
    plt.close()
    
    print(f"El gráfico de resultados se ha guardado en {output_file}")

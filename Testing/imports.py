# imports.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# Función para generar y entrenar el modelo de regresión lineal
def train_linear_regression(X_train, y_train):
    # Crear el modelo de regresión lineal
    model = LinearRegression()
    
    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)
    
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
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Valores reales vs Predicciones')
    plt.tight_layout()
    plt.savefig('regression_results.png')
    plt.close()
    
    print("El gráfico de resultados se ha guardado como 'regression_results.png'")

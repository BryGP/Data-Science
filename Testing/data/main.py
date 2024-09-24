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

from imports import load_and_split_data, train_linear_regression, evaluate_model, plot_results

# Ruta al archivo CSV
file_path = r'C:\Users\bryan\Documents\ITQ\Semestre 8\Ciencia de Datos\DS env\Testing\data\diabetesnorm.csv'
    
# Cargar y dividir los datos
X_train, X_test, y_train, y_test = load_and_split_data(file_path)
    
# Entrenar el modelo
model = train_linear_regression(X_train, y_train)
    
# Evaluar el modelo
mse, rmse, y_pred = evaluate_model(model, X_test, y_test)
    
# Imprimir los resultados
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")
    
# Visualizar los resultados
plot_results(y_test, y_pred)
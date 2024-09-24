# Ciencia de Datos
# Modelo entrenamiento y testing
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
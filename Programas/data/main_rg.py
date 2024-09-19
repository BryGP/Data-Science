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

import sci-kit learn as sk

split data

train test
70%    30%

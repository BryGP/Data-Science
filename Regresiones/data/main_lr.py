# Ciencia de Datos
# Modelo entrenamiento y testing
# Regresión Logística

import process_lr as proc

# Cargar el dataset de cáncer de mama
dataset = proc.load_breast_cancer_data()

# Normalizar los datos y codificar la variable objetivo
norm_dataset, label_encoder = proc.normalize_data(dataset)

# Dividir los datos
training_input, training_output, test_input, test_output, selected_columns = proc.split_data(norm_dataset, 0.3)

# Entrenar el modelo
model = proc.logistic_regression(training_input, training_output)

# Calcular el accuracy
accuracy = proc.get_accuracy(model, test_input, test_output)
print(f"Accuracy: {accuracy}")

# Obtener la matriz de confusión
conf_matrix = proc.get_confusion_matrix(model, test_input, test_output)
print(f"Matriz de confusión:\n{conf_matrix}")
print(f"Forma de los datos de entrenamiento: {training_input.shape}")
print(f"Distribución de la variable objetivo: {training_output.value_counts(normalize=True)}")

# Guardar el modelo
output_dir = proc.check_output_dir()
proc.save_model(model, output_dir)

# Generar la curva ROC
<<<<<<< HEAD:Testing/data/main_lr.py
proc.plot_roc_curve(model, test_input, test_output, output_dir)
=======
proc.plot_roc_curve(model, test_input, test_output, output_dir)

print(f"Forma de los datos de entrenamiento: {training_input.shape}")
print(f"Distribución de la variable objetivo: {training_output.value_counts(normalize=True)}")

# Imprimir las clases codificadas
print(f"Clases codificadas: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
>>>>>>> a25c0f7d85ed39769da5f9bff9b26c7a47871a57:Regresiones/data/main_lr.py

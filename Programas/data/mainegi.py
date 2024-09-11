import pandas as pd
import numpy as np
import os

df = pd.read_csv(r'C:\Users\bryan\Documents\ITQ\Semestre 8\Ciencia de Datos\DS env\Programas\data\datos_inegi.csv')

# Seleccionar las columnas de interés
columns_normalize = df.columns[:10]

# Normalizar cada columna aplicando la formula Z
for col in columns_normalize:
    # Convertir a numerico en caso que no sea numerico
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calcular el valor promedio
    mean = df[col].mean()
    std = df[col].std() # Calcular la desviación estándar
    n = len(df[col].dropna()) # Eliminar los valores nulos

    # Aplicar la formula de normalización
    df[col] = (1/np.sqrt(n)) * ((df[col] - mean) / std)

# Guardar el archivo normalizado
df.to_csv("output/datos_inegi_normalizados.csv", index=False)
print("Normalización finalizada... :)")
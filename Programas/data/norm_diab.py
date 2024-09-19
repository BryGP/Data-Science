import pandas as pd

# Lee el archivo TXT
df = pd.read_csv(r'C:\Users\bryan\Documents\ITQ\Semestre 8\Ciencia de Datos\DS env\Programas\data\diabetes.tab.txt', sep='\t')

# Convierte a tipo float
df = df.astype(float)

# Normalización manual
df_scaled = (df - df.mean()) / df.std()

# Guarda los datos normalizados
df_scaled.to_csv('diabetesnorm.csv', index=False)

print("Datos normalizados guardados en 'diabetesnorm.csv'")
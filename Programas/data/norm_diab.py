import pandas as pd
from sklearn.preprocessing import StandardScaler

# Lee el archivo TXT
df = pd.read_csv(r'C:\Users\bryan\Documents\ITQ\Semestre 8\Ciencia de Datos\DS env\Programas\data\diabetes.tab.txt', sep='\t')

# Convierte a tipo float
df = df.astype(float)

# Escala los valores
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Guarda los datos normalizados
df_scaled.to_csv('diabetesnorm.csv', index=False)

print("Datos normalizados guardados en 'diabetesnorm.csv'")
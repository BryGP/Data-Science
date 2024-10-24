import pandas as pd # Se encarga de leer datos desde txt

pd.set_option('display.max_rows', None) # Mostrar todas las filas
pd.set_option('display.max_columns', None) # Mostrar todas las columnas
# Para volver al set default, solo quitamos el valor None, o bien, borrar las lineas anteriores.

def read_diabetes(path):
    # Lee el archivo y lo retorna
    return pd.read_csv(path, delimiter='\t') # Agregamos el delimitador para que haga el split correctamente

def read_inegi_dataset(path):
    data = pd.read_csv(path, sep=(","))
    return data

# Función para seleccionar las primeras 10 columnas
def select_columns(data, num_columns=10):
    return data.iloc[:, :num_columns]  # Selecciona las primeras 10 columnas
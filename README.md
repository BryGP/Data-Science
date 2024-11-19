# Data Science

Trabajos a desarrollar hasta el momento:

- Limpieza y preparación de datos
- Análisis exploratorio de datos
- Modelado predictivo
- Visualización de datos (histogramas, graficas de correlacion, heatmaps)
- Uso de bibliotecas como Pandas, NumPy, y Matplotlib

El objetivo principal del proyecto es aplicar estos conocimientos para resolver problemas reales mediante el análisis de datos.

## Estructura del Proyecto

- `utils/`: Contiene las utilidades por proyecto.
- `scripts/`: Scripts de Python utilizados para el procesamiento de datos.
- `output/`: Resultados y visualizaciones generadas.

## Entorno de Desarrollo

Para trabajar en este proyecto, se usara un entorno virtual (venv). Aquí te dejamos los pasos para configurarlo:

1. Crea un entorno virtual:
    ```bash
    python -m venv venv
    ```
2. Activa el entorno virtual:
    - En Windows:
      ```bash
      .\venv\Scripts\activate
      ```
3. Instala las dependencias necesarias:
    ```bash
    pip install numpy
    pip install matplotlib
    pip install pandas
    pip install scikitlearn
    ```

## Detalles Adicionales

### Normalización de Datos

La normalización de datos es un paso crucial en el análisis de datos. En este proyecto, hemos utilizado técnicas como la estandarización y la normalización min-max para asegurar que nuestros modelos funcionen correctamente.

### Entrenamiento de Datos
Lo que hicimos en la parte de **Testing** fue entrenar un modelo de datos normalizado acerca de unas estadisticas del Inegi, tanto como para uso visual como la parte del bacj

### Herramientas Utilizadas

- **Pandas**: Para la manipulación y análisis de datos.
- **NumPy**: Para operaciones numéricas.
- **Matplotlib**: Para la visualización de datos.
- **Scikit-learn**: Para el modelado predictivo.

# Adult Income -- Dashboard Interactivo y Predicción

Este proyecto implementa un dashboard interactivo en **Streamlit** para
explorar el dataset **Adult Census Income (ACI)** y realizar
predicciones de ingreso (\>50K) usando un modelo de Árbol de Decisión
previamente entrenado.

## Características

-   **Exploración interactiva (EDA):**
    -   Histogramas, diagramas de dispersión, boxplots y correlaciones.
    -   Filtros dinámicos (numéricos y categóricos).
    -   Descarga de subconjuntos filtrados en CSV.
-   **Predicción (\>50K):**
    -   Formulario para introducir atributos de una persona.
    -   Preprocesamiento con pipelines de `scikit-learn`.
    -   Modelo cargado desde `decision_tree_ACI.pkl`.

## Estructura

    ├── app.py                 # Dashboard principal con Streamlit
    ├── eda.py                 # Limpieza ligera y preparación inicial del dataset
    ├── transformes.py         # Pipelines de preprocesamiento (numéricos y categóricos)
    ├── adult.csv              # Dataset
    ├── decision_tree_ACI.pkl  # Modelo entrenado
    ├── dataset_analysis.ipynb # Notebook de análisis exploratorio
    └── requirements.txt       # Dependencias del proyecto

## Instalación

1.  Clonar este repositorio o descargar los archivos.
2.  Crear un entorno virtual (opcional, pero recomendado):

``` bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

3.  Instalar dependencias:

``` bash
pip install -r requirements.txt
```

## Uso

1.  Asegúrate de tener el archivo `adult.csv` y el modelo
    `decision_tree_ACI.pkl` en la raíz del proyecto.
2.  Ejecuta el dashboard con:

``` bash
streamlit run app.py
```

3.  Abre el enlace local que muestra la terminal (generalmente
    `http://localhost:8501`).

## Notas

-   El dataset **Adult Census Income** se puede obtener desde [UCI
    Machine Learning
    Repository](https://archive.ics.uci.edu/ml/datasets/adult).
-   El archivo `decision_tree_ACI.pkl` debe estar entrenado con el mismo
    preprocesamiento que está definido en `transformes.py`.

------------------------------------------------------------------------

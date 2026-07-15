# TFE Master-Inteligencia-Artificial

## Trabajo Fin de Máster

## Descripción

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Máster del Máster en Inteligencia Artificial (UNIR).

El objetivo del proyecto es desarrollar un modelo de predicción de ventas para la optimización de la producción, basado en entornos empresariales de consultorías que trabajan con el sector HORECA.

Por motivos de confidencialidad, este repositorio únicamente incluye el código fuente. Los datos utilizados pertenecen a un entorno empresarial y no pueden ser distribuidos.

## Tecnologías utilizadas

- Python 3.10+
- pandas / NumPy — tratamiento y análisis de datos
- scikit-learn — modelos baseline y preprocesado
- XGBoost / LightGBM — modelos de gradient boosting
- Optuna — optimización automática de hiperparámetros
- PyTorch — red neuronal LSTM
- SHAP — interpretabilidad de modelos
- Matplotlib / Seaborn — visualización de resultados

## Estructura del repositorio
├── modelo.py    

└── README.md

## Datos

El proyecto utiliza información procedente de distintos maestros empresariales.

En concreto, durante el desarrollo se utilizaron:

- MaestroClientes
- MaestroFacturas
- MaestroProductos

Estos ficheros contienen información confidencial y, por tanto, **no se incluyen en este repositorio**.

Para ejecutar el proyecto será necesario disponer de ficheros propios con la misma estructura y nomenclatura de columnas descrita en la memoria del TFM.

## Cómo ejecutar el proyecto

1. Colocar los ficheros de datos propios en la misma carpeta que `modelo.py`, siguiendo el formato descrito en la memoria.
2. Ejecutar el script:

```bash
python modelo.py
```

El script incluye de forma secuencial la limpieza y tratamiento de anomalías, la ingeniería de variables, el entrenamiento de los modelos baseline, XGBoost, LightGBM y la red LSTM, y la evaluación final con el análisis SHAP.

## Resultados

El modelo LightGBM con parámetros por defecto obtiene el mejor resultado global, reduciendo el error (MAPE) más de un 39% respecto al mejor modelo baseline en una de las dos categorías de producto analizadas. Los detalles completos de la metodología, resultados y discusión se encuentran en la memoria del TFM.

## Notas

Este repositorio se publica exclusivamente con fines académicos. No contiene información confidencial, datos de clientes ni información comercial.

## Autora

Paula Martín Merino
Máster en Inteligencia Artificial — UNIR

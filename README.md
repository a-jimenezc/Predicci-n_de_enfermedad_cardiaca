## Predicción de Enfermedad Cardiaca

### Objetivo

El objetivo del presente análisis es desarrollar un modelo de red neuronal que permita predecir si un paciente presenta un cuadro de enfermedad de arterias coronarias.

### Prerequisitos

Las librerias necesarias están listadas en requirements.txt. También se incluye environment.yml para los usuarios de Anaconda.

### Datos

Los datos seleccionados fueron descargados de Kaggle bajo el nombre "Heart Attack Analysis & Prediction Dataset" y fueron subidos por Rashik Rahman. A su vez, la base de datos orginial fue recolectada por: 
1.	Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2.	University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3.	University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4.	V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

Ver la carpeta de referencias para más información.

### Exploración Inicial de Datos

El conjunto de datos se compone de variables numericas, ordinales y categoricas. La mayoria de los datos proviene de personas por encima de los 40 años y, además, en su mayoria varones.

<img src="referencias/images/age.png" alt="Alt text 1" width="300"/> <img src="referencias/images/gender.png" alt="Alt text 2" width="300"/>

Se puede observar que existe correlación entre ciertas variables numéricas, lo cual sugiere que el modelo se puede beneficiar del uso de regularizacion. 

<img src="referencias/images/corr.png" alt="Alt text 1" width="400"/>

Se puede observar que la variable objetivo esta relativamente balanceada. Esto simplifica el desarrollo del modelo.

<img src="referencias/images/output.png" alt="Alt text 1" width="400"/>

### Construcción del modelo

Se probaron tres arquitecturas diferentes de algoritmo Perceptron, con una, dos y tres capas ocultas. En cada caso se utilizó Grid Search con Crossvalidation para la selección de los hiperparametros. Los modelos se construyeron utilizando la libreria Keras. Además, se usó la implementacion de la libreria Scikit-learn para Grid Search.

La métrica usada como referencia es Accuracy. Esta nos da una medida general del desempeño del modelo cuando se tiene una variable objetivo balanceada.

### Selección del modelo

A continuación se ilustra los resultados de cada uno de los modelos.



### Model Evaluation

[Insert information about the model evaluation process here]

### Model Explanation

[Insert information about the model explanation here]

### Contacto

jimenezc.antn@gmail.com

jimenezc.bo@gmail.com  

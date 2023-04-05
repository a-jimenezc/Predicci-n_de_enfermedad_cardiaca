## Predicción de Enfermedad Cardiaca con Redes Neuronales

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

Se probaron tres arquitecturas diferentes de algoritmo Perceptron, con una, dos y tres capas ocultas. En cada caso se utilizó Grid Search con Crossvalidation para la selección de los hiperparametros. Los modelos se construyeron utilizando la libreria Keras. Además, se usó la implementacion de la libreria Scikit-earn para Grid Search.

La métrica usada como referencia es "Accuracy". Esta nos da una medida general del desempeño del modelo cuando se tiene una variable objetivo balanceada.

### Selección del modelo

A continuación se ilustra los resultados de cada uno de los modelos.

<img src="referencias/images/scores1.png" alt="Alt text 1" width="400"/>

Se observa que el modelo con una capa oculta es el que mejor se desempeña en el test set bajo la métrica "Accuracy". Este modelo tiene la función tanh como función de activación en la capa oculta y consta de 35 neuronas en la misma. 

### Explicabilidad del Modelo

Se desea conocer la importancia de cada una de las variables en el momento de hacer la predicción por parte del modelo. Para ello se usa la técnica de permutación. Esta consiste en permutar los valores de las filas de cada columna, una columna a la vez, y calcular como el desempeño es afectado por esta permutación. Aquellas columnas (esto es, las variables) cuya permutación cause la mayor caída en el desepeño serán las mas importantes para el modelo.

A continuación se detallan los resultados de permutación para el trainin-set y test-set.

<img src="referencias/images/importances_train.png" alt="Alt text 1" width="300"/> <img src="referencias/images/imprtances_test.png" alt="Alt text 2" width="300"/>

En ambos casos, caa (número de vasos principales) es una varible muy importante. También se puede ver que oldpeak (depresión de ST inducida por ejercicio en relación con el reposo) es una variable muy importante cuando se usa el training set, pero no con el test set. Esto puede ser una señal de que esta varible esta causando overfitting. Finalmente, cp_0 (tipo de dolor de pecho. Valor 0: angina típica) es muy importante en el test set, pero no así cuando se usa en training set. Esto sugiere que el modelo puede no estar capturando parte de la complejidad en los datos.


### Conclusiones

El modelo que mejor se desempeña con la métrica escogida es el de arquitectura más simple. Sin embargo, todavía es posible mejorar este modelo buscando hiperparámetros más óptimos realizando una busqueda exaustiva con "Grid Search".

### Contacto


jimenezc.bo@gmail.com  

jimenezc.antn@gmail.com


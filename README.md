### scientific_role
# Introducción
Se presenta todo el flujo para la generación de un modelo que tenga la capacidad de predecir la probailidad de pago de un usuario. Se crea un producto simulando un posible escenario donde el modelo es necesario desplegarlo, re entrenarlo, ajustar parámetros y otro.
## Contenido:
El proyecto se encuentra conformado por los siguientes archivos:
 - data_exploration.ipynb: En este archivo se realiza el análisis descriptivo de algunas variables.
 - variable_analysis.ipynb: En este archivo se realiza un análisis de la calidad de los datos y algunos métodos de selección de variables. En este archivo también se concluyen sobre algunas estrategias en la creación de algunas variables que le den valor al modelo.
  - clean_data.py: En este archivo se limpia los datos garantizando la estructura de las variables. Como se mencionó, el proyecto se realizó simulando una ejecución real, es por esto que este procedimiento se realiza en primer lugar simulando la comunicación con una base de datos que en muchas ocasiones la calidad de los datos no puede ser garantizada.
  - fixture_engineer.py: Este archivo se relaciona con el análisis realizado en variable_analysis.ipynb. En este se realiza la creación de algunas variables y la eliminación de algunas por altos valores de correlación.
  - preprocess_data.py: En este archivo se realiza el procesamiento de los datos antes de evaluarlos en diferentes modelos, en este realiza: Eliminación de outliers que disminuya la capacidad de generalizar, transformaciones de las variables continuas para evitar que altos valores tomen diferentes pesos en los parámetros de los modelos, balanceo de la base de datos de entrenamiento utilizando submuestreo de la clase mayoritaria y transformaciones de la base de datos de evaluación con los parámetros de las transformaciones del entrenamiento.
  - train_model.py: En este se evalúan 4 modelos: RandomForestClassifier, lightgradientboost, Gradient Boosting classifier y Support vector machine. Se utiliza una grilla de parámetros para encontrar la mejor combinación y se fija un hiperparámetro de validación cruzada igual a 3. La métrica con la que se juzga y se optimiza es el "accuracy" que me permite minimizar los falsos positivos.

El código fue estructurado de manera que sea fácilmente editable en caso de detectar alguna falla o si se desea realizar un cambio en algunas estrategia de modelo.

## Consideraciones de la base de datos:
La base de datos present algunas incoherencias entre los posibles valores que se describen en la url y los valores que se otorgan en la base. Es por eso que se opta en no generar  estrategias de manipulación sobre algunas variables. 
## Consideraciones técnicas:
Se realizaron dos testeos, uno mas riguroso que otro. El primero aquel que se genera en la validación cruzada de la búsqueda de parámetro, y la segunda más rigurosa es aquella que se prueba sobre datos reales del fénomeno anteriormente adecuados.
## Flujo:
![process](https://user-images.githubusercontent.com/94578395/227894719-c8cd5803-99f3-42f8-8b64-b8cba6d3912f.png)


## Resultados.
Se presenta a continuación los resultados de los modelos entrenados.

| Modelo | Accuracy validación cruzada | Accuraccy en evaluación |
|-----------|-----------|-----------|
| SVC | 0.70 | 0.68 |
| RFC | 0.72 | 0.73|
| GBC | 0.72 | 0.712 |
| LGB | 0.71 | 0.69 |

El modelo más apto para reconocer defaults en los pagos es el random forest classifier. Debido a que su capacidad de generalización se mantiene en los valores reales de fénomeno, además este modelo cuenta con la característca que no se sobre entrena como se evidencia en los resultados.
## Futuros trabajos (Mejoras en el modelo).
 - Hay gran cantidad de variables que son similares pero con rezagos diferentes, es por esto que resulta más adecuado llenar los valores faltantes con las variables análogas a cada cliente y no al comportamiento general de la columna.
 - Evaluar diferentes estrategias de balanceo.
 - La carga de los datos debe ser fluida, esto en una comunicación más directa.
 - Agregar un módulo de selección de variables según el modelo elegido.
 - En caso de desplegar el modelo generar un módulo para el tratamiento de datos en el testeo.
## POO.
 - Para evitar trabajar directamente con los datos, se pueden crear clases con métodos que facilten la reutilización de código para procesarlos mucho más efcientemente.
 - Debido a la repetición de las líneas de código en el momento de evluar las diferentes parámetros a elegir, resulta más adecuado instanciar una clase que tenga los métodos de la evaluaciín de parámetros.
 - Al utilizar POO es mucho más práctico para utilizar técnicas de clean code y SOLID.

## Conclusión final.

Durante el presente proyecto no se busco únicamente generar un modelo con métricas de rendimiento, se realizó un esfuerzo por generar un producto de datos que tuviera la capacidad de re entrenar el modelo con la menor intervención posible. Con esto se buscaba detallar la estructuración de código y el manejo de buenas prácticas en programación y en machine learnign

*Simón Gómez Giraldo*

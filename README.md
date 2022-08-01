# Simple-Supported-Plate-Deflection
Este repositorio fue hecho por ESAU IDROVO. Estudiante de la carrera de Ingeniería naval en la Escuela Superior Politécnica del Litoral. Para la materia de "Ship Structures"
El script [main,py](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/main.py) realiza una regresión lineal para los datos de los experimentos [ss1.txt](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/ss1.txt) y [ss2.txt](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/ss2.txt) además de hacer una regresión lineal teórica.
Para la obtención de los valores teóricos se usó una ampliación de series sinusoidales, con el algoritmo que describe la siguientes fórmulas:
| ![formulas.png](https://www.efunda.com/formulae/solid_mechanics/plates/calculators/images/SSSS_PPoint_mx.gif)| 
|:--:| 
| ![formulas2.png](https://www.efunda.com/formulae/solid_mechanics/plates/calculators/images/SSSS_PPoint_my.gif)|
|| 
| *Formulas extraidas de [aquí](https://www.efunda.com/formulae/solid_mechanics/plates/calculators/SSSS_PPoint.cfm)* |

# NOTA DE DATOS DE ENTRADA.
Los datos necesarios para que el programa funcione se encuentran en [data.json](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/data.json).
* E: Modulo de Young
* pois: Coeficiente de Poisson
* a: Longitud de la plancha en el eje X (lado mas largo)
* b: Longitud de la plancha en el eje Y
* t: Espesor de la plancha
* xf: Posicion de la carga en el eje x
* yf: Posicion de la carga en el eje y
* s1,2,3,4: Cada S# hace referencia a una lista con una posicion para cada "strain gauge" en pares tipo [x,y]
* w1,2,3,4: Pesos en kg. El script fue hecho para 4 pesos.
* Series: Cantidad de series sinusoidales que se calcular

## EJEMPLO DE OUTPUT PARA LOS EXPERIMENTOS ss1 y ss2
| ![exp1.png](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/Simple%20supported/exp1.png) | 
|:--:| 
| *Experimento 1. Visualización de la deformación unitaria experimental* |

| ![lr_exp1.png](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/Simple%20supported/Experimental1.png) | 
|:--:| 
| *Experimento 1. Regresión lineal* |

| ![exp2.png](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/Simple%20supported/exp2.png) | 
|:--:| 
| *Experimento 2. Visualización de la deformación unitaria experimental* |

| ![lr_exp2.png](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/Simple%20supported/Experimental2.png) | 
|:--:| 
| *Experimento 2. Regresión lineal* |

| ![lr_theo.png](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/Simple%20supported/theory.png) | 
|:--:| 
| *Regresión lineal teórica* |

# Extras
También, el script cuenta con funciones para graficación de la distribución de la carga,la deflexión y los momentos
## Ejemplos de salida

| ![carga.png](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/Simple%20supported/load.jpg) | 
|:--:| 
| *Distribución teórica de carga puntual* |

| ![w.png](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/Simple%20supported/displacement.jpg) | 
|:--:| 
| *Desplazamiento teórico* |

| ![mx.png](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/Simple%20supported/momentx.jpg) | 
|:--:| 
| *Distribución teórica del momento en x* |

| ![my.png](https://github.com/eiidrovo/Simple-Supported-Plate-Deflection/blob/main/Simple%20supported/momenty.jpg) | 
|:--:| 
| *Distribución teórica del momento en y* |

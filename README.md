# Ejercicio 1 - Análisis de Sentimiento en Reseñas de Yelp

## Descripción

Este proyecto compara dos enfoques de análisis de sentimiento sobre reseñas de usuarios obtenidas del dataset Yelp Reviews:

* Método basado en léxicos.
* Método basado en modelo preentrenado (TextBlob).

El objetivo es evaluar y comparar ambos enfoques mediante métricas de clasificación.

## Dataset

Se utilizó el dataset Yelp Reviews disponible en Kaggle:

https://www.kaggle.com/datasets/vivekhn/yelp-reviews

## Metodología

### 1. Carga y preparación de datos

Se cargó el dataset y se seleccionaron las variables:

* text
* stars

Posteriormente se creó una variable objetivo llamada sentiment:

* Positivo (1): reseñas con 4 o 5 estrellas.
* Negativo (0): reseñas con menos de 4 estrellas.

### 2. Método basado en léxicos

Se construyó un diccionario simple de palabras positivas y negativas.

La clasificación se realizó comparando la cantidad de palabras positivas y negativas presentes en cada reseña.

### 3. Método basado en TextBlob

Se utilizó la librería TextBlob para calcular automáticamente la polaridad de cada reseña.

Las reseñas con polaridad positiva fueron clasificadas como positivas y las restantes como negativas.

### 4. Evaluación

Se calcularon las siguientes métricas:

* Accuracy
* Precision
* Recall
* F1 Score

## Resultados

| Método   | Accuracy | Precision | Recall | F1    |
| -------- | -------- | --------- | ------ | ----- |
| Léxico   | 0.726    | 0.769     | 0.851  | 0.812 |
| TextBlob | 0.735    | 0.731     | 0.970  | 0.834 |

## Conclusiones

Los resultados muestran que el modelo basado en TextBlob obtiene el mejor desempeño general, especialmente en Recall y F1 Score.

Mientras que el enfoque léxico es simple y fácil de implementar, los modelos preentrenados permiten capturar mejor el contexto y las características del lenguaje natural, logrando una clasificación más precisa.

Por este motivo, para aplicaciones reales de análisis de sentimiento se recomienda utilizar modelos preentrenados o enfoques más avanzados de NLP.

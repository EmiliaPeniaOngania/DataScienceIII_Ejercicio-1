Ejercicio #1 – Análisis de Sentimientos
Alumna: María Emilia Peña Onganía
Introducción
El análisis de sentimiento es una de las tareas más utilizadas dentro del Procesamiento de Lenguaje Natural (NLP). Su objetivo consiste en identificar la opinión o emoción expresada en un texto, clasificándola generalmente como positiva, negativa o neutra.
Este tipo de análisis tiene aplicaciones en múltiples áreas, como monitoreo de marcas, análisis de reseñas de productos, atención al cliente y redes sociales.
En este trabajo se comparan dos enfoques diferentes para la clasificación de sentimientos utilizando un dataset de reseñas de Yelp:
Método basado en léxicos.
Modelo preentrenado utilizando TextBlob.
Marco Teórico
Polaridad en análisis de sentimiento
La polaridad representa la orientación emocional de un texto.
Un texto puede clasificarse como:
Positivo: expresa satisfacción o aprobación.
Negativo: expresa descontento o crítica.
Neutro: no presenta una emoción claramente identificable.
La polaridad es una de las métricas más utilizadas para cuantificar opiniones dentro de grandes volúmenes de texto.
Tipos de clasificadores de sentimiento
Métodos basados en léxicos
Utilizan diccionarios o listas de palabras positivas y negativas para determinar el sentimiento de un texto.
Ventajas:
Simples de implementar.
No requieren entrenamiento.
Limitaciones:
No comprenden contexto.
No detectan ironía o sarcasmo.
Modelos supervisados
Aprenden patrones a partir de ejemplos previamente etiquetados.
Ventajas:
Mayor precisión.
Mejor adaptación a distintos dominios.
Limitaciones:
Requieren datos etiquetados.
Necesitan entrenamiento.
Modelos preentrenados
Utilizan conocimiento lingüístico previamente aprendido.
En este trabajo se utiliza TextBlob como ejemplo de modelo preentrenado para análisis de sentimiento.


Dataset utilizado
Se utilizó un conjunto de reseñas provenientes de Yelp.
Para construir una variable objetivo binaria:
4 y 5 estrellas → Sentimiento positivo (1)
1, 2 y 3 estrellas → Sentimiento negativo (0)
Esta transformación permitió comparar las predicciones de ambos métodos contra una referencia real.

Preprocesamiento de Texto
Antes de realizar el análisis se aplicó un preprocesamiento básico.
Las tareas realizadas incluyeron:
Conversión de texto a minúsculas.
Normalización del contenido textual.
Preparación de los datos para los métodos de clasificación.
En proyectos de NLP más avanzados también suelen utilizarse técnicas como:
Tokenización.
Eliminación de stopwords.
Lematización.

División de Datos
Para evaluar correctamente los modelos se realizó una separación entre datos de entrenamiento y validación utilizando la función train_test_split de Scikit-Learn.
La división utilizada fue:
80% entrenamiento.
20% validación.
De esta forma las métricas se calcularon únicamente sobre el conjunto de validación, evitando evaluar los modelos sobre los mismos datos utilizados durante el desarrollo.
Aunque los métodos utilizados no requieren entrenamiento supervisado, se realizó la división train/test para evaluar el desempeño sobre datos no vistos y seguir buenas prácticas de validación. 


Metodología
Método basado en léxicos
Este enfoque utiliza una lista predefinida de palabras positivas y negativas. La clasificación se realiza contando la cantidad de términos positivos y negativos presentes en cada reseña.
Si predominan las palabras positivas, la reseña se clasifica como positiva; en caso contrario, como negativa.
Código utilizado
positive_words = [
    "good","great","excellent","amazing",
    "love","best","nice","perfect",
    "wonderful","awesome","fantastic"
]
negative_words = [
    "bad","terrible","worst","awful",
    "poor","hate","horrible",
    "disappointing","boring","slow"
]
def lexicon_sentiment(text):
    text = str(text).lower()
    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)
    if pos > neg:
        return 1
    else:
        return 0
df["lexicon_prediction"] = df["text"].apply(
    lexicon_sentiment
)
df[["text","sentiment","lexicon_prediction"]].head()

Modelo preentrenado: TextBlob
Se utilizó la librería TextBlob para calcular automáticamente la polaridad de cada texto.
Las reseñas con polaridad positiva fueron clasificadas como positivas y las restantes como negativas.

Código utilizado


!pip install textblob


from textblob import TextBlob


def textblob_sentiment(text):


    polarity = TextBlob(str(text)).sentiment.polarity


    if polarity > 0:
        return 1
    else:
        return 0


df["textblob_prediction"] = df["text"].apply(
    textblob_sentiment
)



from sklearn.model_selection import train_test_split


train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)
print("Train:", train_df.shape)
print("Test:", test_df.shape)



Métricas de evaluación
Para comparar ambos enfoques se utilizaron las siguientes métricas:
Accuracy
Precision
Recall
F1 Score
Estas métricas permiten evaluar tanto la exactitud general como la capacidad de identificar correctamente cada clase.
Resultados

| Método   | Accuracy | Precision | Recall | F1 Score |
| -------- | -------- | --------- | ------ | -------- |
| Léxico   | 0.733    | 0.781     | 0.861  | 0.819    |
| TextBlob | 0.746    | 0.781     | 0.972  | 0.843    |


Código utilizado
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


print("===== METODO LEXICO =====")
print("Accuracy:",
      accuracy_score(
    test_df["sentiment"],
    test_df["lexicon_prediction"]
))
print("Precision:",
      precision_score(test_df["sentiment"],
                      test_df["lexicon_prediction"]))
print("Recall:",
      recall_score(test_df["sentiment"],
                   test_df["lexicon_prediction"]))
print("F1:",
      f1_score(test_df["sentiment"],
               test_df["lexicon_prediction"]))



print("===== TEXTBLOB =====")
print("Accuracy:",
      accuracy_score(test_df["sentiment"],
                     test_df["textblob_prediction"]))
print("Precision:",
      precision_score(test_df["sentiment"],
                      test_df["textblob_prediction"]))
print("Recall:",
      recall_score(test_df["sentiment"],
                   test_df["textblob_prediction"]))
print("F1:",
      f1_score(test_df["sentiment"],




Matriz de Confusión

Figura 1. Matriz de confusión del modelo TextBlob.
La matriz de confusión permite visualizar la cantidad de clasificaciones correctas e incorrectas realizadas por el modelo.
Se observa que la mayoría de las reseñas fueron clasificadas correctamente, aunque existen algunos errores de predicción que afectan las métricas generales.
Código utilizado
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


cm = confusion_matrix(
    test_df["sentiment"],
    test_df["textblob_prediction"]
)
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Matriz de Confusión - TextBlob")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

Análisis de resultados
Los resultados muestran que ambos enfoques logran un desempeño aceptable en la tarea de clasificación.
El método basado en léxicos presenta una implementación sencilla y rápida, aunque depende fuertemente de la calidad del diccionario utilizado y no logra capturar adecuadamente el contexto del lenguaje.
Por otro lado, TextBlob obtiene mejores resultados generales, especialmente en Recall y F1 Score. Esto indica una mayor capacidad para identificar correctamente las reseñas positivas y una mejor adaptación a variaciones lingüísticas presentes en los textos.
El Recall obtenido por TextBlob (0.972) indica una alta capacidad para detectar correctamente reseñas positivas. Asimismo, el F1 Score superior respecto al método léxico sugiere un mejor equilibrio entre precisión y cobertura. Estos resultados muestran que los modelos preentrenados pueden capturar mejor el contexto lingüístico que los enfoques basados exclusivamente en diccionarios de palabras. 

Comparación de Resultados

| Método   | Accuracy | Precision | Recall | F1 Score |
| -------- | -------- | --------- | ------ | -------- |
| Léxico   | 0.733    | 0.781     | 0.861  | 0.819    |
| TextBlob | 0.746    | 0.781     | 0.972  | 0.843    |




El modelo TextBlob obtuvo un desempeño superior al método basado en léxicos, especialmente en Recall y F1 Score.
Esto sugiere que los modelos preentrenados poseen una mejor capacidad para capturar información contextual presente en los textos.

Ventajas y Limitaciones
Método Léxico
Ventajas:
Fácil implementación.
Bajo costo computacional.
Limitaciones:
Dependencia de listas de palabras.
Escasa comprensión del contexto.
TextBlob
Ventajas:
Mejor interpretación lingüística.
Mayor capacidad de generalización.
Limitaciones:
Dependencia del modelo preentrenado.
Menor control sobre las reglas internas.

Conclusiones
Se compararon dos enfoques para clasificación de sentimiento utilizando reseñas de Yelp.
Los resultados mostraron que TextBlob obtuvo mejores métricas generales que el enfoque basado en léxicos.
Esto demuestra la ventaja de utilizar modelos preentrenados cuando se busca capturar información contextual más compleja.
Como trabajo futuro podrían explorarse modelos más avanzados basados en Transformers, como BERT o RoBERTa, para mejorar aún más el desempeño del análisis de sentimiento.




# Preprocesamiento, Validaciones de Datos y Tratamiento de Desbalanceo en Machine Learning

**Guía Técnica Completa:** Verificación de Datos, Aislamiento de Sets, Algoritmo SMOTE y Métricas de Evaluación

---

## 1. Resumen Ejecutivo e Introducción

En el desarrollo de sistemas de Aprendizaje Automático (Machine Learning), la etapa de preprocesamiento y validación de datos es la base fundamental que determina la fiabilidad, capacidad de generalización y rendimiento en producción de cualquier modelo predictivo. Un error común en equipos de ciencia de datos es avanzar directamente al entrenamiento de algoritmos complejos (como XGBoost o Redes Neuronales) sin realizar una auditoría previa minuciosa sobre la estructura y relaciones del dataset de entrenamiento (Train) y de prueba (Test).

Este documento constituye un manual exhaustivo sobre las validaciones requeridas antes del modelado, el aislamiento estricto de trasformaciones para prevenir la fuga de información (Data Leakage), y el tratamiento metódico del desbalanceo de clases, abordando en detalle el funcionamiento matemático y geométrico del algoritmo SMOTE.

> 📌 **PRINCIPIO FUNDAMENTAL**
>
> El principio *'Garbage In, Garbage Out'* (Basura entra, basura sale) rige el comportamiento de los modelos de ML. Si los datos de entrenamiento contienen sesgos ocultos, fuga de respuestas o deformaciones en la distribución, el modelo aprenderá patrones espurios que colapsarán al desplegarse en entorno real.

---

## 2. Validaciones y Estudios Previos sobre los Datos (Train y Test)

Previo a la ejecución de cualquier algoritmo, se deben llevar a cabo 6 estudios principales para certificar la integridad y validez de la partición de datos.

### 2.1. Verificación de Fuga de Datos (Data Leakage)

La fuga de datos ocurre cuando la información del conjunto de prueba o del 'futuro' se filtra inadvertidamente en el conjunto de entrenamiento durante la etapa de construcción de características o preprocesamiento. Esto produce métricas de rendimiento falsamente optimistas durante la validación que no se replican en producción.

- **Fuga por Target Leakage:** incluir variables de entrada que solo se conocen después de que el evento a predecir ya ocurrió (ej: incluir 'días en mora' para predecir si un cliente incurrirá en mora).
- **Fuga Temporal:** realizar divisiones aleatorias en series temporales en lugar de cortes cronológicos estrictos, permitiendo que el modelo entrenes con datos del futuro.
- **Fuga por Identificadores (ID Leakage):** utilizar columnas clave o correlativas ordenadas cronológicamente como características predictivas.

*Figura: Flujo correcto de partición y aislamiento de transformaciones para evitar Data Leakage.*

### 2.2. Aislamiento Estricto del Preprocesamiento (Fit vs. Transform)

**Regla de Oro en ML:** todo cálculo de parámetros (media, desviación estándar, valores mínimos/máximos, frecuencias de categoría o imputadores de valores faltantes) debe realizarse exclusivamente sobre el set de entrenamiento mediante el método `.fit()`. Posteriormente, dichos parámetros calculados se aplican mediante `.transform()` tanto al conjunto de entrenamiento como al conjunto de prueba.

> 📌 **ERROR COMÚN DE PREPROCESAMIENTO**
>
> Ejemplo de error grave: si calculas la media global de una columna (Train + Test) para rellenar nulos antes de la partición, el set de entrenamiento habrá absorbido información estadística directa del set de prueba, violando la independencia de los datos.

### 2.3. Similitud de Distribuciones (Covariate Shift)

El **Covariate Shift** ocurre cuando la distribución marginal de las variables independientes P(X) cambia entre el set de entrenamiento y el de prueba, aunque la relación condicional P(Y|X) se mantenga. Se deben aplicar las siguientes pruebas:

- **Prueba de Kolmogorov-Smirnov (KS):** para variables continuas, compara si dos muestras proceden de la misma distribución estadística continua.
- **Population Stability Index (PSI):** mide cuánto ha cambiado la distribución de una variable a lo largo del tiempo o entre particiones (PSI > 0.25 indica un cambio significativo).
- **Adversarial Validation:** entrenar un clasificador (ej. Random Forest) para distinguir si una fila pertenece a Train o a Test. Si el modelo logra clasificarlas fácilmente (AUC > 0.7), existe un sesgo severo entre conjuntos.

*Figura: Desplazamiento de distribución (Covariate Shift) en una variable entre Train y Test.*

### 2.4. Muestreo Estratificado (Stratified Train-Test Split)

En problemas de clasificación (especialmente desbalanceados), una división aleatoria simple puede alterar drásticamente la proporción de la clase objetivo en cada partición. La partición estratificada garantiza que la proporción de clases se mantenga exactamente idéntica tanto en Train como en Test.

### 2.5. Manejo de Categorías No Vistas (Unseen Categories)

En variables cualitativas, es común que el set de prueba o producción contenga niveles o categorías que jamás aparecieron durante el entrenamiento. Para evitar fallos en tiempo de ejecución o errores de codificación:

- **One-Hot Encoding:** configurar `handle_unknown='ignore'` para asignar vectores nulos a categorías desconocidas.
- **Target Encoding:** implementar suavizado (*smoothing*) y asignar la media global del target a categorías no vistas.

### 2.6. Auditoría de Integridad y Duplicados entre Sets

Se debe verificar formalmente que no existan filas duplicadas compartidas entre el conjunto de entrenamiento y el de prueba. La contaminación por registros idénticos genera métricas infladas artificialmente, haciendo que el modelo parezca memorizar ejemplos exactos en lugar de aprender patrones de generalización.

### Resumen de Validaciones Pre-Entrenamiento

| Estudio / Verificación | Riesgo Principal si no se realiza | Técnica / Herramienta de Solución |
|---|---|---|
| Fuga de Datos (Data Leakage) | Métricas falsas e inoperatividad en producción | Partición temporal estricta y aislamiento de variables |
| Aislamiento de Preprocesamiento | Contaminación estadística de datos de testeo | Pipeline con `fit` solo en Train y `transform` en Test |
| Covariate Shift | Bajo rendimiento del modelo por cambio de dominio | Pruebas KS, PSI y Adversarial Validation |
| Estratificación de Clases | Sesgo en la proporción del target entre sets | `StratifiedKFold` / Stratified Train-Test Split |
| Categorías No Vistas | Excepciones en código o fallos al codificar | `handle_unknown='ignore'` y agrupación 'Otros' |
| Registros Duplicados | Memorización y sobreajuste no detectado | Eliminar duplicados cruzados entre Train y Test |

---

## 3. Técnicas para Solucionar el Desbalanceo de Clases

El desbalanceo de clases es un fenómeno común en problemas del mundo real como detección de fraude, diagnóstico médico, fallos en maquinaria o predicción de abandono de clientes (churn). Ocurre cuando la clase de interés (clase minoritaria o positiva) representa un porcentaje muy pequeño del total de observaciones.

### 3.1. Diagnóstico y la Paradoja de la Precisión (Accuracy Paradox)

Evaluar un dataset desbalanceado mediante la métrica de 'Exactitud' o 'Precisión Global' (Accuracy) conduce a conclusiones completamente engañosas.

**Ejemplo Numérico:** en un dataset de 10,000 transacciones bancarias donde solo 100 son fraudulentas (1%), un modelo ingenuo que clasifique absolutamente TODAS las transacciones como 'No Fraude' obtendrá un Accuracy del 99%. A pesar del elevado porcentaje, el modelo es completamente inútil ya que su tasa de detección de fraude es del 0%.

### 3.2. Técnicas a Nivel de Datos (Resampling)

Consisten en reequilibrar la distribución de clases en el conjunto de entrenamiento antes de alimentar el algoritmo.

- **Random Undersampling:** elimina aleatoriamente instancias de la clase mayoritaria. *Riesgo:* pérdida de información crítica.
- **Random Oversampling:** duplica aleatoriamente instancias de la clase minoritaria. *Riesgo:* sobreajuste (Overfitting) al repetir filas exactas.

*Figura: Comparativa visual entre Dataset Original, Undersampling, Oversampling y SMOTE.*

### 3.3. Algoritmo SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE supera las limitaciones del sobremuestreo simple generando observaciones sintéticas plausibles mediante la interpolación lineal entre instancias reales de la clase minoritaria en el espacio de características.

**Funcionamiento Paso a Paso de SMOTE:**

1. **Filtrado de la clase minoritaria:** se seleccionan únicamente los puntos pertenecientes a la clase minoritaria `C_min`.
2. **Búsqueda de Vecinos ($k$-NN):** para cada punto `x_i` en `C_min`, se identifican sus *k* vecinos más cercanos (habitualmente k=5) mediante distancia euclidiana en el espacio n-dimensional.
3. **Selección aleatoria de vecino:** se escoge al azar uno de los *k* vecinos más cercanos, denominado `x_hat_i`.
4. **Cálculo del Vector Diferencia:** se obtiene la distancia y dirección entre el punto origen y el vecino: `(x_hat_i - x_i)`.
5. **Generación del punto sintético:** se elige un número aleatorio lambda distribuido uniformemente entre 0 y 1. El nuevo punto sintético se calcula según la ecuación:

> 📌 **FORMULACIÓN MATEMÁTICA**
>
> Ecuación Fundamental de SMOTE:
>
> ```text
> x_sintético = x_i + λ · (x_hat_i - x_i)      donde  λ ~ U(0, 1)
> ```

**Ejemplo Numérico Resuelto Paso a Paso:**

Consideremos un modelo de detección de fraude con dos características: [Monto de Transacción en miles, Hora del día].

- Punto origen seleccionado (`x_i`): [10.0, 14.0]
- Vecino k-NN seleccionado (`x_hat_i`): [14.0, 18.0]
- Factor aleatorio generado (λ): 0.25

```text
1. Cálculo del Vector Diferencia:
   (x_hat_i - x_i) = [14.0 - 10.0, 18.0 - 14.0] = [4.0, 4.0]

2. Escalado por el factor λ = 0.25:
   0.25 · [4.0, 4.0] = [1.0, 1.0]

3. Cálculo del Punto Sintético Final:
   x_sintético = [10.0, 14.0] + [1.0, 1.0] = [11.0, 15.0]
```

*Figura: Representación geométrica de la interpolación vectorial en SMOTE.*

**Variantes Avanzadas de SMOTE:**

- **SMOTE-NC (Nominal & Continuous):** permite trabajar con datasets mixtos que contienen variables continuas y categóricas.
- **Borderline-SMOTE:** genera puntos sintéticos únicamente en la frontera de decisión entre clases (puntos en riesgo de ser mal clasificados).
- **ADASYN (Adaptive Synthetic):** genera más datos sintéticos en regiones donde la densidad de la clase minoritaria es menor y más difícil de aprender.

### 3.4. Técnicas a Nivel de Algoritmo (Cost-Sensitive Learning)

En lugar de alterar la cantidad de datos, se modifica la función de pérdida del algoritmo asignando mayor penalización a los errores cometiendo sobre la clase minoritaria.

**Parámetro `class_weight='balanced'`:** ajusta automáticamente los pesos de forma inversamente proporcional a las frecuencias de clase en los datos de entrada:

> 📌 **FÓRMULA DE PESOS BALANCED**
>
> ```text
> Peso de Clase (w_j) = N_total / (N_clases · N_j)
> ```
>
> Donde `N_total` es el número total de muestras, `N_clases` es el número de clases y `N_j` es el número de muestras de la clase *j*.

### 3.5. Métricas de Evaluación y Ajuste de Umbral (Thresholding)

Para evaluar modelos en escenarios desbalanceados se deben utilizar métricas derivadas de la Matriz de Confusión:

- **Precision (Precisión):** `TP / (TP + FP)`. De los casos predichos como positivos, ¿cuántos eran reales?
- **Recall (Sensibilidad):** `TP / (TP + FN)`. De todos los casos positivos reales, ¿cuántos logró capturar el modelo?
- **F1-Score:** `2 · (Precision · Recall) / (Precision + Recall)`. Media armónica entre Precision y Recall.
- **PR-AUC (Area Under Precision-Recall Curve):** métrica superior a ROC-AUC cuando el desbalanceo es extremo (> 99% vs < 1%).

**Ajuste del Umbral de Decisión (Moving Threshold):** por defecto, los clasificadores asignan la clase positiva si la probabilidad `p >= 0.50`. En detección de fraude o diagnóstico médico, se puede optimizar el umbral (por ejemplo, bajándolo a `p >= 0.20`) para maximizar el Recall sin colapsar la Precision.

### Comparativa General de Estrategias ante Desbalanceo

| Estrategia | Mecanismo | Ventaja Principal | Riesgo / Desventaja |
|---|---|---|---|
| Undersampling | Elimina datos mayoritarios | Rapidez de entrenamiento | Pérdida de información valiosa |
| Oversampling | Duplica datos minoritarios | Sencillo de implementar | Alto riesgo de overfitting |
| SMOTE | Interpola puntos sintéticos | Conserva estructura geométrica | Sensible a ruido y outliers |
| Class Weights | Aumenta la penalización | No altera el tamaño del dataset | Requiere calibrar probabilidades |
| Threshold Tuning | Modifica el umbral (0.5 -> 0.2) | Optimiza métrica de negocio sin reentrenar | Requiere validación cuidadosa |

---

## 4. Lista de Comprobación Final (Checklist Pre-Entrenamiento)

Antes de lanzar el entrenamiento definitivo de cualquier modelo en producción, verifique el cumplimiento de las siguientes pautas:

- [x] ¿Se separó el dataset en Train y Test **ANTES** de cualquier operación de escalado, imputación o codificación?
- [x] ¿Se utilizó muestreo estratificado (`StratifiedSplit`) para preservar la proporción del target?
- [x] ¿Se aplicó `.fit()` únicamente sobre el conjunto de entrenamiento?
- [x] ¿Se verificó que no existan filas duplicadas compartidas entre Train y Test?
- [x] ¿Se evaluó la presencia de Covariate Shift mediante pruebas estadísticas (KS / PSI / Adversarial Validation)?
- [x] ¿Se seleccionaron métricas inmunes al desbalanceo (F1-Score, PR-AUC) en lugar de Accuracy?
- [x] Si se utilizó SMOTE, ¿se aplicó **SOLAMENTE** sobre el conjunto de entrenamiento y **NUNCA** sobre el conjunto de prueba?
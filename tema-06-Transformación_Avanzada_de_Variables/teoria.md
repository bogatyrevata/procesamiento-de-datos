# Transformación Avanzada de Variables

## Bloque 1 – Teórico: Fundamentos Matemáticos y Geométricos

La transformación de variables no es un mero paso de preprocesamiento, sino una alteración geométrica del espacio de características que impacta directamente en la función de costo y la convergencia de los algoritmos.

### 1. Modelos sensibles a la escala y la topología del error

- **Algoritmos basados en distancia (KNN, SVM, K-Means):** utilizan métricas como la distancia Euclidiana o de Manhattan. Si una variable tiene un rango de [0, 1000] y otra de [0, 1], la primera dominará el cálculo de la norma espacial, anulando la varianza de la segunda.

  ```text
  X = 2*10 + 1*1000000000000
  ```

- **Algoritmos de optimización (Regresión Logística, Redes Neuronales):** en métodos basados en Descenso de Gradiente, variables no escaladas generan una superficie de error elíptica. El escalado transforma la superficie en una forma más esférica, permitiendo una convergencia directa y rápida.

### 2. Escalado Numérico Avanzado

- **Normalización (Min-Max Scaling):** comprime los datos a un hipercubo acotado, típicamente [0, 1]. Es ideal cuando la distribución subyacente no es Gaussiana.
- **Estandarización (Z-score Scaling):** asume y fuerza propiedades de una distribución normal, centrando en μ = 0 con σ = 1. Preserva la información sobre los valores atípicos (outliers) mejor que Min-Max.

### 3. Codificación de Variables Categóricas

- **Label Encoding:** transforma categorías en enteros (0, 1, 2...). **Riesgo crítico:** impone una ordinalidad matemática donde no existe. Solo válido para variables estrictamente ordinales.
- **One-Hot Encoding:** ortogonaliza las categorías creando un hiperplano donde cada categoría es equidistante. Para evitar multicolinealidad, se debe aplicar `drop='first'` en modelos lineales.

---

## Bloque 2 – Práctico: Implementación Robusta con sklearn

Este bloque demuestra cómo integrar transformaciones evitando la filtración de datos (**data leakage**) usando `Pipeline` y `ColumnTransformer`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Carga de datos y adición de columna categórica
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
np.random.seed(42)
df['tipo_flor'] = np.random.choice(['A', 'B', 'C'], size=len(df))

X = df.drop(columns=[])
y = iris.target

# Split ANTES de cualquier transformación (Prevención de Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ColumnTransformer para aplicar transformaciones simultáneas
num_cols = iris.feature_names
cat_cols = ['tipo_flor']

preprocessor = ColumnTransformer(
    transformers=[
        ('num_minmax', MinMaxScaler(), num_cols),
        ('cat_ohe', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
```

---

## Bloque 3 – Teórico: Transformaciones No Lineales y Patrones

### 1. Problemas, Cardinalidad y Data Leakage

- **Regla de Oro:** todo escalador (`fit`) debe ajustarse únicamente sobre los datos de entrenamiento.
- **Alta Cardinalidad:** One-Hot Encoding sobre columnas con miles de categorías genera "La Maldición de la Dimensionalidad".

### 2. Transformaciones No Lineales (Power Transforms)

- **Objetivo:** homogeneizar la varianza y acercar distribuciones sesgadas a una distribución Normal.
- **Logarítmica y Raíz Cuadrada:** reducen el impacto de las colas largas.
- **Box-Cox y Yeo-Johnson:** maximizan la normalidad.

### 3. Ejemplos en Arquitecturas IA

- **Redes Neuronales (Deep Learning):** reclaman inputs estandarizados para evitar que los gradientes se desvanezcan o exploten.
- **PCA:** el escalado es obligatorio para no seleccionar erróneamente variables por su escala.

---

## Bloque 4 – Práctico: Asimetría y Power Transforms

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder

# Generamos datos sintéticos con sesgo
np.random.seed(0)
df = pd.DataFrame({
    'residual_sugar': np.random.exponential(scale=2, size=1000),
    'alcohol': np.random.normal(loc=10, scale=1.5, size=1000),
    'quality': np.random.randint(3, 9, size=1000)
})

# Transformación Logarítmica
df['sugar_log'] = np.log1p(df['residual_sugar'])

# Yeo-Johnson (Avanzado)
pt = PowerTransformer(method='yeo-johnson')
df['sugar_yj'] = pt.fit_transform(df[['residual_sugar']])

# Categorización de variable objetivo continua a ordinal
condiciones = [
    (df['quality'] <= 4),
    (df['quality'] >= 5) & (df['quality'] <= 6),
    (df['quality'] >= 7)
]
etiquetas = ['baja', 'media', 'alta']
df['calidad_cat'] = np.select(condiciones, etiquetas)

# Codificación Ordinal (LabelEncoding seguro)
encoder = OrdinalEncoder(categories=[['baja', 'media', 'alta']])
df['calidad_cat_encoded'] = encoder.fit_transform(df[['calidad_cat']])
```

---

## Estructura del Entregable 6

El alumno deberá:

1. Generar set de datos sintéticos.
2. Entregar un notebook estructurado con un resumen de las transformaciones, análisis de histogramas y el impacto potencial en modelos.

---

## Anexo: ¿Por qué PCA falla catastróficamente si no se aplica StandardScaler?

El Análisis de Componentes Principales (**PCA**) es un algoritmo de reducción de dimensionalidad cuyo objetivo central es encontrar proyecciones ortogonales que maximicen la varianza de los datos. Si las variables originales no están estandarizadas, PCA simplemente identificará la variable con la mayor escala numérica como el "componente principal", ignorando por completo la estructura de correlación real del dataset.

### La Explicación Matemática

PCA busca un vector de pesos `w` que maximice la varianza de la proyección. Si X1 se mide en una escala enorme y X2 en una pequeña, entonces `Var(X1) >> Var(X2)`. El autovector dominante simplemente devolverá el eje de la variable X1. Matemáticamente, el algoritmo no fracasó, pero analíticamente es un fracaso catastrófico porque el Primer Componente Principal (PC1) no descubrió un patrón latente, sino que simplemente devolvió la variable original de mayor escala.

### Ejemplo Práctico: Altura y Peso

Imagina que estás analizando datos biométricos. Tienes la Altura medida en metros (ej. 1.75 m) y el Peso en gramos (ej. 75,000 g).

- **Escenario 1: PCA sin StandardScaler.** La nube de puntos en el espacio 2D será extremadamente alargada a lo largo del eje del Peso. El PC1 será paralelo al eje del Peso, explicando casi toda la varianza. La Altura será tratada como ruido estadístico.
- **Escenario 2: PCA con StandardScaler.** Al aplicar `StandardScaler`, fuerzas a ambas variables a tener media 0 y varianza 1. Ahora ambas variables compiten en igualdad de condiciones. PC1 capturará la correlación real: el hecho de que las personas más altas tienden a pesar más.
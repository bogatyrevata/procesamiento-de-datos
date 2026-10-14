# Módulo 5 – Identificación y Tratamiento de Valores Atípicos (Outliers)

## Bloque 1 – Teórico (1 h): Conceptos y Detección Univariante

Un valor atípico (**outlier**) es una observación que se desvía significativamente del resto de los datos, suscitando la sospecha de que fue generada por un mecanismo distinto. La clave analítica no es simplemente detectarlos, sino comprender su naturaleza: ¿es un error que debe purgarse o información vital sobre un evento extremo?

### Diferencia entre Error y Dato Extremo Válido

- **Error:** una persona con edad de 150 años (error de tipeo) o un sensor de temperatura marcando -273 °C en un horno (falla del sensor). Estos deben ser tratados.
- **Dato extremo válido:** una transacción de fraude de $500,000 en una cuenta promedio de $1,000. Eliminar este dato destruye exactamente el patrón que los algoritmos buscan.

### Causas Comunes

- **Errores de medición/carga:** factores humanos, problemas de conversión de unidades, fallas instrumentales.
- **Variabilidad natural:** eventos raros pero esperados en distribuciones de cola pesada.

### Métodos de Detección

#### 1. Rango Intercuartílico (IQR)

Robusto contra la presencia de otros outliers. Define un límite superior e inferior basado en la dispersión central de los datos (entre el percentil 25 y 75).

Fórmulas:

```text
IQR = Q3 - Q1
Límite Inferior = Q1 - 1.5 * IQR
Límite Superior = Q3 + 1.5 * IQR
```

#### 2. Z-Score

Mide a cuántas desviaciones estándar (σ) se encuentra un punto (x) de la media (μ). Se asume una distribución normal. Típicamente, si |Z| > 3, el valor es un outlier.

Fórmula:

```text
Z = (x - μ) / σ
```

### Tratamiento de Outliers

- **Eliminación:** solo justificada si se comprueba que es un error de medición irrecuperable.
- **Transformación:** aplicar logaritmo o raíz cuadrada para comprimir grandes escalas y normalizar la distribución.
- **Winsorización:** sustituir los valores que exceden ciertos percentiles (ej. p1 y p99) por el valor exacto de esos percentiles. Evita la pérdida de registros preservando la distribución central.

---

## Bloque 2 – Práctico (1 h): Dataset House Prices

**Dataset:** House Prices (Kaggle).
**Archivo:** `Outliers_HousePrices.ipynb`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Paso 0: Carga del dataset (versión reducida)
df = pd.read_csv('house_prices.csv')
precio = df['SalePrice']

# Paso 1: Detectar outliers en precios con boxplot
plt.figure(figsize=(10, 4))
sns.boxplot(x=precio)
plt.title('Detección visual de outliers: SalePrice')
plt.show()

# Paso 2: Calcular Z-score para variables numéricas
z_scores = stats.zscore(precio)
abs_z_scores = np.abs(z_scores)
outliers_z = df[abs_z_scores > 3]
print(f"Outliers detectados por Z-Score (>3): {len(outliers_z)}")

# Paso 3: Detectar outliers usando IQR y filtrar
Q1 = precio.quantile(0.25)
Q3 = precio.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtramos el dataset
df_iqr_filtered = df[(precio >= lower_bound) & (precio <= upper_bound)]
print(f"Outliers eliminados por IQR: {len(df) - len(df_iqr_filtered)}")
```

---

## Bloque 3 – Teórico (1 h): Detección Multivariante

Los valores atípicos multivariantes son observaciones que no son extremas en ninguna variable de forma individual, pero que presentan una combinación inusual de valores.

### Análisis Bivariante y Multivariante

Un departamento de 20 metros cuadrados no es un outlier. Un departamento que cuesta $500,000 no es un outlier. Pero un departamento de 20 metros cuadrados que cuesta $500,000 es claramente una anomalía. Se visualizan fácilmente mediante **Scatterplots**.

### Métodos Avanzados de Detección

Algoritmos de aprendizaje no supervisado:

#### 1. Isolation Forest

Algoritmo basado en árboles que funciona "aislando" anomalías. En lugar de perfilar datos normales, aísla de forma explícita los datos anómalos. Realiza cortes aleatorios; las anomalías (escasas y diferentes) requieren menos cortes para quedar aisladas en un nodo.

#### 2. DBSCAN

Algoritmo de clustering basado en densidad. Agrupa puntos muy cercanos basándose en una métrica de distancia y un número mínimo de puntos. Los puntos en regiones de baja densidad se marcan como anomalías.

### Decisiones Éticas y Analíticas

- **Sesgo por eliminación:** purgar clientes con ingresos muy bajos puede sesgar un modelo de crédito haciéndolo discriminatorio.
- **Regla de Oro:** nunca elimines un outlier sin documentar exactamente por qué lo hiciste. Si no puedes confirmar que es un error, debes mantenerlo y usar algoritmos robustos.

---

## Bloque 4 – Práctico (1 h): Dataset Titanic

**Dataset:** Titanic (versión numérica).
**Archivo:** `Outliers_Titanic.ipynb`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Paso 0: Cargar dataset
df = pd.read_csv('titanic_numeric.csv')

# Paso 1: Detectar tarifas (Fare) extremas con boxplot e IQR
Q1_fare = df['Fare'].quantile(0.25)
Q3_fare = df['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
upper_fare = Q3_fare + 1.5 * IQR_fare

# Paso 2: Detectar edades extremas con Z-score
edades = df['Age'].dropna()
z_scores_age = stats.zscore(edades)
outliers_age = edades[np.abs(z_scores_age) > 3]

# Paso 3: Eliminar (Edad) y recalcular estadísticas
df_sin_outliers_edad = df[~df['Age'].isin(outliers_age)]

# Paso 4: Winsorizar tarifas extremas (Percentiles 1 y 99)
p1 = df['Fare'].quantile(0.01)
p99 = df['Fare'].quantile(0.99)
df['Fare_Winsorized'] = np.clip(df['Fare'], p1, p99)
```

---

## 📌 Entregable 5

Un único notebook que contenga:

1. Generar set de datos sintéticos.
2. Realizar estudio de outliers con todos los métodos vistos en clase.
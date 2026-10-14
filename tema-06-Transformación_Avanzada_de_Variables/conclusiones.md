---
tema: 6
asignatura: Procesamiento de datos
tags: [transformacion, conclusiones]
---

# Тема 6. Преобразование переменных: выводы / Conclusiones

## Español (para el notebook)

**1. Min-Max vs. estandarización.** Ambos métodos cambian la escala pero no la forma
de la distribución: la asimetría de `ingreso` fue 2.07 en los tres casos (original,
Min-Max y estandarizado). Min-Max concentró la mayor parte de los clientes en el rango
[0, 0.22] porque unos pocos ingresos altos estiraron la escala hasta 1; la
estandarización no fija un rango y se ve menos afectada por esos valores extremos.
En el conjunto de prueba los valores no son exactamente [0, 1] ni media 0 / σ = 1
(p. ej. σ = 0.84 en `ingreso`), lo que confirma que los parámetros se ajustaron solo
con train y no hubo data leakage.

**2. Rendimiento de KNN.** Sin escalar, KNN obtuvo una exactitud de 0.485 en test y
0.508 en validación cruzada, es decir, al nivel del azar (el baseline era 0.515),
porque `ingreso` (decenas de miles) dominó la distancia euclidiana y anuló a `edad` y
`antiguedad_meses`, que son las variables que realmente influyen en `compra`.
Con escalado la validación cruzada subió a ≈ 0.63–0.64. Las diferencias entre
StandardScaler y MinMaxScaler no son concluyentes (0.634 vs. 0.641 en validación
cruzada).

**3. PCA.** Sin estandarizar, PC1 explicó el 100 % de la varianza y su vector fue
prácticamente el eje de `ingreso` (peso 1.0), pues su varianza (≈1.26·10⁹) es millones de
veces mayor que la de `edad` (≈216) o `antiguedad_meses` (≈4.5·10³). PC1 no descubrió
ninguna estructura, solo devolvió la variable de mayor escala. Con StandardScaler, la
varianza se repartió de forma pareja (0.35 / 0.34 / 0.31), ya que en estos datos
sintéticos las variables son independientes.

**4. Transformación elegida para `ingreso`.** Yeo-Johnson (λ ≈ −0.003), que en la
práctica equivale a un logaritmo: redujo la asimetría de 2.07 a 0.00 y dejó el
histograma simétrico. Log1p dio un resultado casi idéntico (0.01); se prefiere
Yeo-Johnson porque estandariza el resultado y admite ceros y valores negativos. Hay
que tener en cuenta que, al generar `ingreso` con una distribución lognormal, el
logaritmo "corrige" casi perfectamente la asimetría; en datos reales el ajuste
sería menos limpio.

---

## Русский (для конспекта)

**1. Min-Max и стандартизация.** Оба метода меняют масштаб, но не форму распределения:
асимметрия `ingreso` была 2.07 во всех трёх вариантах (исходный, Min-Max и
стандартизированный). Min-Max сжал большую часть клиентов в диапазон [0; 0.22],
потому что несколько высоких доходов растянули шкалу до 1. Стандартизация диапазон не
фиксирует и меньше зависит от этих крайних значений. На тестовой выборке значения не
равны точно [0, 1] и не имеют среднего 0 и σ = 1 (например, σ = 0.84 у `ingreso`), и
это подтверждает, что параметры подбирались только по train и утечки данных
(data leakage) не было.

**2. Качество KNN.** Без масштабирования KNN показал точность 0.485 на тесте и 0.508 по
кросс-валидации, то есть на уровне случайного угадывания (базовый уровень был 0.515):
`ingreso` (десятки тысяч) доминировал в евклидовом расстоянии и «заглушил» `edad` и
`antiguedad_meses`, которые на самом деле влияют на `compra`. После масштабирования
кросс-валидация поднялась примерно до 0.63–0.64. Разница между StandardScaler и
MinMaxScaler не является убедительной (0.634 против 0.641 по кросс-валидации).

**3. PCA.** Без стандартизации PC1 объяснил 100 % дисперсии, а его вектор практически
совпал с осью `ingreso` (вес 1.0), потому что дисперсия дохода (≈1.26·10⁹) в миллионы раз
больше, чем у `edad` (≈216) и `antiguedad_meses` (≈4.5·10³). PC1 не обнаружил никакой
структуры, а просто вернул признак с самым большим масштабом. После StandardScaler
дисперсия распределилась ровно (0.35 / 0.34 / 0.31), так как в этих синтетических данных
признаки независимы.

**4. Выбранное преобразование для `ingreso`.** Yeo-Johnson (λ ≈ −0.003), который на
практике равносилен логарифму: он снизил асимметрию с 2.07 до 0.00 и сделал гистограмму
симметричной. `log1p` дал почти такой же результат (0.01). Я предпочитаю Yeo-Johnson,
потому что он сразу стандартизирует результат и допускает нули и отрицательные
значения. Нужно помнить, что `ingreso` сгенерирован из логнормального распределения,
поэтому логарифм «исправляет» асимметрию почти идеально. На реальных данных результат
был бы менее чистым.
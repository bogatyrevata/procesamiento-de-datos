---
tema: 5
asignatura: Procesamiento de datos
tags: [outliers, conclusiones]
---

# Tema 5. Выбросы: выводы / Conclusiones

## Español (para el notebook)

### 6.1 ¿Qué método detectó mejor cada tipo de outlier?

**Outliers univariantes** (4 precios entre 950.000 y 2.000.000): IQR y Z-score los
detectaron todos. El Z-score no tuvo falsos positivos (precision = 1.00), mientras que
el IQR marcó además 1 registro legítimo (precision = 0.80). Sin embargo, el Z-score es
sensible al *masking*: los 4 valores extremos inflaron la desviación estándar de
≈30.000 a ≈153.000, y un registro con Z = 0.43 pasó a Z = 2.76 al calcularlo sin ellos.
El IQR, basado en cuartiles, resultó más robusto.

**Outliers multivariantes** (superficie 33–41 m² con precio 175.000–200.000): IQR y
Z-score no detectaron ninguno (0 de 6), porque cada variable por separado tiene valores
normales. Isolation Forest y DBSCAN detectaron los 6 (recall = 1.0), a costa de
6 y 8 falsos positivos respectivamente (precision 0.62 y 0.56).
En nuestro caso Isolation Forest fue algo más preciso y menos sensible a sus parámetros:
DBSCAN pasó de encontrar 10 de 10 (eps = 0.20) a 5 de 10 (eps = 0.25).

### 6.2 ¿Qué tratamiento aplicaría a cada tipo de outlier?

- **Univariantes (precios absurdos):** un departamento de 85 m² a 2.000.000 es
  inverosímil frente al resto, por lo que, previa verificación de que es un error de
  carga, se eliminarían o corregirían. Si no se puede confirmar, la winsorización con los
  límites del IQR conserva los 310 registros y reduce la desviación estándar de ≈153.000
  a ≈31.000. La winsorización con p1/p99 resultó poco eficaz (p99 = 885.046), porque los
  outliers superan el 1 % de los datos y contaminan el propio percentil.
  La transformación logarítmica reduce la asimetría, pero no elimina los outliers.
- **Multivariantes:** no los eliminaría automáticamente. Primero revisaría cada caso
  (¿un departamento pequeño pero céntrico o de lujo?, ¿error de carga de la superficie?).
  Si son legítimos, los conservaría y usaría modelos robustos o una variable indicadora.

### 6.3 Riesgos de eliminar estos datos en un caso real

- Podría eliminarse información valiosa: un valor extremo legítimo (como un fraude
  real) es justamente el patrón que se quiere detectar.
- Los falsos positivos son datos legítimos: DBSCAN marcó 8 departamentos normales muy
  pequeños o muy grandes; eliminarlos sesgaría el modelo hacia los casos "promedio".
- Los resultados dependen de parámetros elegidos (contamination, eps, umbral de Z), y
  sin una etiqueta real como `es_outlier_real` no se puede validar la elección.
- Regla de oro: toda eliminación debe documentarse (qué, cuántos y por qué).

---

## Русский (для конспекта)

### 6.1 Какой метод лучше обнаружил каждый тип выбросов?

**Одномерные выбросы** (4 цены от 950 000 до 2 000 000): IQR и Z-score нашли все четыре.
У Z-score не было ложных срабатываний (precision = 1.00), а IQR дополнительно пометил
1 нормальную запись (precision = 0.80). Однако Z-score чувствителен к эффекту
маскировки (*masking*): четыре экстремальных значения раздули стандартное отклонение
с ≈30 000 до ≈153 000, и у одной записи Z-оценка изменилась с 0.43 до 2.76, когда мы
посчитали её без этих значений. IQR опирается на квартили и оказался устойчивее.

**Многомерные выбросы** (площадь 33–41 м² при цене 175 000–200 000): IQR и Z-score
не нашли ни одного (0 из 6), потому что по каждой переменной в отдельности значения
выглядят нормально. Isolation Forest и DBSCAN нашли все 6 (recall = 1.0), но ценой
6 и 8 ложных срабатываний соответственно (precision 0.62 и 0.56).
В нашем случае Isolation Forest оказался чуть точнее и менее чувствительным к
параметрам: у DBSCAN результат упал с 10 найденных из 10 (eps = 0.20) до 5 из 10
(eps = 0.25), а у Isolation Forest число находок менялось плавно.

### 6.2 Какую обработку я бы применила к каждому типу выбросов и почему?

- **Одномерные (абсурдные цены):** квартира 85 м² за 2 000 000 неправдоподобна на фоне
  остальных, поэтому после проверки, что это ошибка ввода, такие значения нужно
  удалить или исправить. Если подтвердить ошибку нельзя, подойдёт винзоризация по
  границам IQR: она сохраняет все 310 записей и снижает стандартное отклонение
  с ≈153 000 до ≈31 000. Винзоризация по p1/p99 оказалась малоэффективной
  (p99 = 885 046), потому что выбросов больше 1 % данных и они «загрязнили» сам
  перцентиль. Логарифмическое преобразование снижает асимметрию, но выбросы не убирает.
- **Многомерные:** автоматически я бы их не удаляла. Сначала нужно проверить каждый
  случай (маленькая, но дорогая квартира в центре или ошибка в площади?). Если
  значения легитимны, их стоит оставить и использовать устойчивые модели или
  дополнительный признак-индикатор.

### 6.3 Риски удаления этих данных в реальной задаче

- Можно потерять ценную информацию: легитимное экстремальное значение (например,
  настоящее мошенничество) как раз и есть тот паттерн, который мы ищем.
- Ложные срабатывания это нормальные данные: DBSCAN пометил 8 обычных квартир с очень
  малой или очень большой площадью, и их удаление сместило бы модель в сторону
  «средних» случаев.
- Результат зависит от выбранных параметров (contamination, eps, порог Z), а без
  настоящей разметки вроде `es_outlier_real` проверить этот выбор невозможно.
- Золотое правило: любое удаление нужно документировать (что, сколько и почему).
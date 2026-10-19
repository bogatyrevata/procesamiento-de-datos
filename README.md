# Procesamiento de datos

Конспекты и практические ноутбуки по курсу *Procesamiento de datos* (обработка данных для машинного обучения). Каждая тема лежит в отдельной папке: теория на испанском и русском плюс практический блокнот. Итоговое задание «Actividad Práctica Completa» лежит в папке `actividad-completa/`.

## Содержание

| Тема                                         | Теория ES | Теория RU | Практика      |
|----------------------------------------------|:---------:|:---------:|:-------------:|
| 5. Выбросы: выявление и обработка            | [ES][t5-es] | [RU][t5-ru] | [Блокнот][t5-p] |
| 6. Продвинутое преобразование переменных     | [ES][t6-es] | [RU][t6-ru] | [Блокнот][t6-p] |
| 7. Предобработка, валидация и балансировка   | [ES][t7-es] | скоро       | скоро           |
| Итоговое задание: Parte 1-2 (синтетические данные), Parte 3 (реальные датасеты) | [ES][ac-es] | скоро | [Parte 1][ac-p1] · [Parte 2][ac-p2] · [Adult][ac-p3] |

## Структура репозитория

````
procesamiento-de-datos/
├── actividad-completa/           # итоговое задание (3 части)
│   ├── data/                     # данные (Adult и др.)
│   ├── actividad_completa_1.ipynb
│   ├── actividad_completa_2.ipynb
│   ├── dataset_riesgo_crediticio_dirty.csv
│   ├── teoria_parte2.md
│   ├── teoria_parte3.md
│   └── parte3_adult.ipynb
├── tema-05-.../
│   ├── teoria.md             # теория (испанский)
│   ├── teoria-rus.md         # теория (русский)
│   ├── conclusiones.md       # выводы по практике
│   └── practica.ipynb        # практический блокнот
├── tema-06-.../
├── tema-07-.../
├── requirements.txt
└── README.md
````

## Запуск проекта

````bash
git clone https://github.com/USER/procesamiento-de-datos.git
cd procesamiento-de-datos

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

Затем откройте нужный блокнот (`practica.ipynb` темы или ноутбук из `actividad-completa/`) в VS Code или Jupyter:

````bash
jupyter lab
````

Рекомендуется сначала прочитать `teoria.md` (или `teoria-rus.md`) темы, а потом переходить к практике. Пути к данным в блокнотах относительные (`data/...`), поэтому запускайте ноутбук из папки его темы.

## Итоговое задание: реальные датасеты (Parte 3)

Задание «Búsqueda de datasets reales» даёт только каталог датасетов в четырёх группах, без явных инструкций. Моя трактовка: применить методы Parte 1-2 к реальным данным, по одному датасету на группу, по ноутбуку на каждый, с адаптацией под тип данных. В начале каждого ноутбука есть блок «Interpretación del enunciado» (ES + RU).

| Датасет | Особенность | Статус |
|---|---|---|
| Adult (UCI) | скрытые пропуски (`?`), редкие категории, служебное значение 99999 | готово |
| Credit Card Fraud (Kaggle) | дисбаланс: SMOTE только на train, порог, PR-AUC, Isolation Forest | план |
| Beijing PM2.5 (UCI) | временной ряд: интерполяция, ресемплинг, split по времени | план |
| NYC Yellow Taxi (один месячный файл) | большой объём: память, Haversine, Dask/PySpark | план |

Сквозные принципы:

- split train/test (со стратификацией при дисбалансе) делается до любых вычислений по данным;
- всё, что извлекается из данных (масштабирование, кодирование, частоты), обучается только на train внутри пайплайна;
- ячейки, меняющие колонки, начинаются с копии из исходного датафрейма, поэтому повторяемы.

## Просмотр без установки

- Блокноты и `.md`-файлы открываются прямо на GitHub, выводы и графики сохранены в `.ipynb`.
- Небольшие CSV GitHub показывает таблицей в браузере.
- Если блокнот не отображается, откройте его через [nbviewer](https://nbviewer.org/).

## Терминология

В русских конспектах в начале каждой темы есть таблица терминов «Español → Русский».

<!-- Ссылки на файлы тем -->
[t5-es]: <tema-05-Identificación_y_Tratamiento_de_Valores_Atípicos_(Outliers)/teoria.md>
[t5-ru]: <tema-05-Identificación_y_Tratamiento_de_Valores_Atípicos_(Outliers)/teoria-rus.md>
[t5-p]:  <tema-05-Identificación_y_Tratamiento_de_Valores_Atípicos_(Outliers)/practica.ipynb>
[t6-es]: <tema-06-Transformación_Avanzada_de_Variables/teoria.md>
[t6-ru]: <tema-06-Transformación_Avanzada_de_Variables/teoria-rus.md>
[t6-p]:  <tema-06-Transformación_Avanzada_de_Variables/practica.ipynb>
[t7-es]: <tema-07-Preprocesamiento_Validacion_y_Balanceo_ML/teoria.md>

<!-- Итоговое задание -->
[ac-es]: actividad-completa/teoria_parte2.md
[ac-p1]: actividad-completa/actividad_completa_1.ipynb
[ac-p2]: actividad-completa/actividad_completa_2.ipynb
[ac-p3]: actividad-completa/parte3_adult.ipynb

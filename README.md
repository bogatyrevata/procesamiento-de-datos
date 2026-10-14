# Procesamiento de datos

Конспекты и практические ноутбуки по курсу *Procesamiento de datos* (обработка данных для машинного обучения). Каждая тема лежит в отдельной папке: теория на испанском и русском плюс практический блокнот.

## Содержание

| Тема                                         | Теория ES | Теория RU | Практика      |
|----------------------------------------------|:---------:|:---------:|:-------------:|
| 5. Выбросы: выявление и обработка            | [ES][t5-es] | [RU][t5-ru] | [Блокнот][t5-p] |
| 6. Продвинутое преобразование переменных     | [ES][t6-es] | [RU][t6-ru] | [Блокнот][t6-p] |
| 7. Предобработка, валидация и балансировка   | [ES][t7-es] | скоро       | скоро           |

## Структура репозитория

```
procesamiento-de-datos/
├── tema-05-.../
│   ├── data/             # данные для практики
│   ├── teoria.md         # теория (испанский)
│   ├── teoria-rus.md     # теория (русский)
│   └── practica.ipynb    # практический блокнот
├── tema-06-.../
├── tema-07-.../
├── requirements.txt
└── README.md
```

## Запуск проекта

```bash
git clone https://github.com/USER/procesamiento-de-datos.git
cd procesamiento-de-datos

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Затем откройте `practica.ipynb` нужной темы в VS Code или Jupyter:

```bash
jupyter lab
```

Рекомендуется сначала прочитать `teoria.md` (или `teoria-rus.md`) темы, а потом переходить к практике. Пути к данным в блокнотах относительные (`data/...`), поэтому запускайте ноутбук из папки его темы.

## Просмотр без установки

- Блокноты и `.md`-файлы открываются прямо на GitHub, выводы и графики сохранены в `.ipynb`.
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
# Проект

Ассистент по региону Кавказские Минеральные Воды, собранный как пайплайн из
четырёх моделей внутри NVIDIA Triton. Грубые сообщения отсекаются, остальные
уходят в поиск по корпусу и генерацию ответа по найденным документам.

Моделями дирижирует Business Logic Scripting на python-бэкенде.

# Как работает пайплайн

Один вход, две ветки. Клиент шлёт текст в `assistant_bls`, обратно получает
`ANSWER`, `ROUTE` и `DOCS`.

```
                    TEXT
                     |
              [ assistant_bls ]  python backend, BLS
                     |
              toxicity_clf      onnxruntime, rubert-tiny-toxicity
                 /       \
          грубость      норма
             |              |
          отказ        e5_embedder   onnxruntime, multilingual-e5-small
                            |
                     поиск по корпусу   806 документов, векторы посчитаны заранее
                            |
                     text_generator     vllm, Qwen3-0.6B
                            |
                         ответ
```

Что делает BLS ([model.py](model_repository/assistant_bls/1/model.py)):

- токенизирует текст словарём rubert и зовёт `toxicity_clf`. Сигмоида ниже 0.5 — сообщение грубое, 
  дальше цепочка не идёт;
- иначе считает вектор вопроса через `e5_embedder` и умножает его на матрицу
  корпуса, берест топ-3 документа;
- собирает промпт из документов, размечает роли `apply_chat_template`
  и зовёт `text_generator`. 

Токенизация внутри BLS, словари лежат рядом с ним. 
Векторы корпуса считаются офлайн ноутбуком, в рантайме через ONNX
проходит только вопрос.

# Структура

```
model_repository/
    assistant_bls/      дирижёр: BLS на python-бэкенде
        1/
            model.py            ветвление, поиск, сборка ответа
            bls_prompt.py       сборка промпта, общая с замерами
            corpus/             documents.json и vectors.npy
            tokenizers/         словари ru, e5, qwen
    toxicity_clf/       классификатор грубости, onnxruntime
    e5_embedder/        эмбеддер, onnxruntime
    text_generator/     генератор, vllm
notebooks/
    export_onnx.ipynb           экспорт двух моделей в ONNX, словари в BLS
    build_corpus_index.ipynb    отбор документов и векторы корпуса
    bench.ipynb                 нагрузочное тестирование, таблицы, график
scripts/client.py       проверка обеих веток пайплайна
data/                   исходный корпус
bench/                  входные данные для Perf Analyzer
report/                 отчёт, сводка, график, сырые логи прогонов

```

# Как воспроизвести

Нужны Docker с доступом к видеокарте и `uv`. Проверялось на Windows, GTX 1650,
4 ГБ видеопамяти.

```bash
# 1. окружение для ноутбуков
uv sync

# 2. образ: базовый с vllm плюс скопированный бэкенд onnxruntime
docker build -t triton-hw .
```

Дальше выполнить два ноутбука — они кладут модели и корпус в `model_repository`:

- [notebooks/export_onnx.ipynb](notebooks/export_onnx.ipynb) — `.onnx` для классификатора и эмбеддера, словари в BLS
- [notebooks/build_corpus_index.ipynb](notebooks/build_corpus_index.ipynb) — `documents.json` и `vectors.npy`

```bash
# 3. сервер: 8000 HTTP, 8001 gRPC, 8002 метрики
docker compose up -d

# 4. проверка обеих веток
uv run python scripts/client.py
```

Замеры — [notebooks/bench.ipynb](notebooks/bench.ipynb): готовит входные файлы,
гоняет Perf Analyzer из образа `tritonserver:25.05-py3-sdk`, складывает CSV
и логи в `report/raw/`, строит сводную таблицу и график.

> ⚠️ **Про место на диске.** При сборке образа временно качается донорский
> образ с бэкендом onnxruntime, около 20 ГБ. После сборки его можно удалить:
> ```bash
> docker image rm hubimage/nvcr-io-nvidia-tritonserver:25.05-py3
> ```

# Отчёт

Результаты нагрузочного тестирования, таблицы и выводы: [report/results.md](report/results.md).

Коротко: генератор дороже остального пайплайна на два порядка. Одна и та же
цепочка выдаёт 0.18 запроса в секунду с генерацией и 26.9 — на ветке отказа,
где генератор не вызывается.

# Данные

Корпус по КМВ от другого учебного проекта, используется как готовый набор документов для поиска.

`data/search_documents.csv` — документы об объектах региона Кавказские
Минеральные Воды, собранные из открытых источников:

- **OpenStreetMap** — © участники OpenStreetMap,
  [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
  Файл является производной базой данных и распространяется на тех же условиях.
- **ЕГРКН, Минкультуры России** — opendata.mkrf.ru, типовые условия
  использования открытых данных РФ.
- **Wikivoyage и Википедия** — © авторы,
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  Тексты сокращены и переформатированы.
- **Wikidata** — CC0 1.0.

Цены для части объектов синтетические и к реальным отношения не имеют. Датасет сырой и собранный наспех - корректность данных не гарантируется (гарантируется некорректность).

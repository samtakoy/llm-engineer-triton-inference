# Отчёт: оркестрация моделей в Triton

## 1. Дерево Model Repository

```
model_repository
├── assistant_bls
│   ├── 1
│   │   ├── __pycache__
│   │   │   ├── bls_prompt.cpython-312.pyc
│   │   │   └── model.cpython-312.pyc
│   │   ├── corpus
│   │   │   ├── documents.json
│   │   │   └── vectors.npy
│   │   ├── tokenizers
│   │   │   ├── e5
│   │   │   │   ├── sentencepiece.bpe.model
│   │   │   │   ├── special_tokens_map.json
│   │   │   │   ├── tokenizer.json
│   │   │   │   └── tokenizer_config.json
│   │   │   ├── qwen
│   │   │   │   ├── added_tokens.json
│   │   │   │   ├── merges.txt
│   │   │   │   ├── special_tokens_map.json
│   │   │   │   ├── tokenizer.json
│   │   │   │   ├── tokenizer_config.json
│   │   │   │   └── vocab.json
│   │   │   └── ru
│   │   │       ├── special_tokens_map.json
│   │   │       ├── tokenizer.json
│   │   │       ├── tokenizer_config.json
│   │   │       └── vocab.txt
│   │   ├── bls_prompt.py
│   │   └── model.py
│   └── config.pbtxt
├── e5_embedder
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
├── text_generator
│   ├── 1
│   │   └── model.json
│   └── config.pbtxt
└── toxicity_clf
    ├── 1
    │   └── model.onnx
    └── config.pbtxt
```

## 2. Конфигурация моделей

### `assistant_bls/config.pbtxt`

```protobuf
name: "assistant_bls"
backend: "python"
max_batch_size: 0

input [
  { name: "TEXT", data_type: TYPE_STRING, dims: [ 1 ] }
]

output [
  { name: "ANSWER", data_type: TYPE_STRING, dims: [ 1 ] },
  { name: "ROUTE",  data_type: TYPE_STRING, dims: [ 1 ] },
  { name: "DOCS",   data_type: TYPE_STRING, dims: [ 1 ] }
]

instance_group [ { count: 2, kind: KIND_CPU } ]
```

### `e5_embedder/config.pbtxt`

```protobuf
name: "e5_embedder"
backend: "onnxruntime"
max_batch_size: 8

input [
  { name: "input_ids",      data_type: TYPE_INT64, dims: [ 128 ] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [ 128 ] }
]

output [
  { name: "sentence_embedding", data_type: TYPE_FP32, dims: [ 384 ] }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 3000
}

instance_group [ { count: 4, kind: KIND_CPU } ]
parameters {
  key: "intra_op_thread_count"
  value: { string_value: "2" }
}
```

### `text_generator/config.pbtxt`

```protobuf
backend: "vllm"

instance_group [ { count: 1, kind: KIND_MODEL } ]
```

### `toxicity_clf/config.pbtxt`

```protobuf
name: "toxicity_clf"
backend: "onnxruntime"
max_batch_size: 8

input [
  { name: "input_ids",      data_type: TYPE_INT64, dims: [ 128 ] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [ 128 ] },
  { name: "token_type_ids", data_type: TYPE_INT64, dims: [ 128 ] }
]

output [
  { name: "logits", data_type: TYPE_FP32, dims: [ 5 ] }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 3000
}

instance_group [ { count: 2, kind: KIND_CPU } ]
```

Код BLS — [`model_repository/assistant_bls/1/model.py`](../model_repository/assistant_bls/1/model.py),
сборка промпта — [`bls_prompt.py`](../model_repository/assistant_bls/1/bls_prompt.py).

## 3. Все модели READY

```
Модели в репозитории:
   ✅ ready  assistant_bls
   ✅ ready  e5_embedder
   ✅ ready  text_generator
   ✅ ready  toxicity_clf
```

## 4. Замеры

| scenario      |   Concurrency |   Inferences/Second |   avg latency |   p95 latency |   Server Queue |   Server Compute Infer |
|:--------------|--------------:|--------------------:|--------------:|--------------:|---------------:|-----------------------:|
| bls_questions |             1 |                0.18 |          5.43 |          8.83 |           0    |                   5.43 |
| bls_questions |             2 |                0.27 |          7.15 |         12.62 |           0    |                   7.14 |
| bls_questions |             4 |                0.23 |         17.09 |         35.13 |           8.42 |                   8.67 |
| bls_rude      |             1 |               26.87 |          0.04 |          0.05 |           0    |                   0.03 |
| bls_rude      |             2 |               30.34 |          0.06 |          0.12 |           0    |                   0.06 |
| bls_rude      |             4 |               26.91 |          0.15 |          0.24 |           0.06 |                   0.07 |
| e5            |             1 |               16.58 |          0.06 |          0.08 |           0    |                   0.05 |
| e5            |             8 |               25.97 |          0.31 |          0.47 |           0.13 |                   0.17 |
| e5            |            16 |               28.08 |          0.56 |          0.79 |           0.3  |                   0.25 |
| e5_count1     |            16 |               28.08 |          0.57 |          0.82 |           0.31 |                   0.24 |
| e5_count2     |            16 |               20.42 |          0.78 |          1.14 |           0.21 |                   0.56 |
| e5_count4x2   |            16 |               26.16 |          0.6  |          0.94 |           0.1  |                   0.49 |
| e5_count4x4   |            16 |               34.16 |          0.46 |          0.72 |           0.07 |                   0.38 |
| e5_nobatch    |             1 |               10.66 |          0.09 |          0.11 |           0    |                   0.09 |
| e5_nobatch    |             8 |               30.96 |          0.26 |          0.29 |           0.12 |                   0.13 |
| e5_nobatch    |            16 |               30.25 |          0.52 |          0.59 |           0.38 |                   0.13 |
| gen           |             1 |                0.13 |          7.44 |         11.13 |           0    |                   0    |
| gen           |             4 |                0.19 |         20.8  |         39.6  |           0    |                   0    |
| gen           |             8 |                0.18 |         41.47 |         76.13 |           0    |                   0    |
| tox           |             1 |               32.86 |          0.03 |          0.05 |           0    |                   0.02 |
| tox           |             8 |               74.64 |          0.11 |          0.19 |           0.03 |                   0.07 |
| tox           |            16 |               89.33 |          0.18 |          0.29 |           0.05 |                   0.12 |
| tox_nobatch   |             1 |               24.47 |          0.04 |          0.07 |           0    |                   0.03 |
| tox_nobatch   |             8 |               54.36 |          0.15 |          0.21 |           0.11 |                   0.04 |
| tox_nobatch   |            16 |               49.52 |          0.32 |          0.46 |           0.27 |                   0.04 |

Задержки в миллисекундах, `avg latency` — сумма всех этапов, от отправки
клиентом до получения ответа.

![Пропускная способность и задержка](bench.png)

## 5. Батчинг и без него

Сценарии `*_nobatch` — те же модели с вырезанным блоком `dynamic_batching`.

TODO: сравнить строки `tox` и `tox_nobatch`, `e5` и `e5_nobatch`, написать вывод.

## 6. Загрузка железа

`nvidia-smi` в покое и под нагрузкой генератора:
[`raw/nvidia_smi_idle.txt`](raw/nvidia_smi_idle.txt),
[`raw/nvidia_smi_gen_load.txt`](raw/nvidia_smi_gen_load.txt).
Метрики Triton с порта 8002: [`raw/metrics_gen_load.txt`](raw/metrics_gen_load.txt),
смотреть `nv_inference_request_duration_us` и `nv_inference_queue_duration_us`
по каждой модели.

TODO: вписать занятую видеопамять и утилизацию GPU под нагрузкой.

## 7. Узкое место

TODO: сравнить `bls_questions` и `bls_rude` — во втором генератор не вызывается.
Разница в RPS показывает его долю в стоимости запроса.

## 8. Чем Triton лучше FastAPI

- промежуточные тензоры не сериализуются и не ходят по сети: BLS вызывает
  модели внутри сервера
- батчинг настраивается конфигом, а не пишется руками
- число копий модели меняется строкой `instance_group`, без правки кода
- метрики по каждой модели отдельно есть из коробки, порт 8002

## 9. Предложения по оптимизации

TODO: опереться на свои цифры. Кандидаты: `instance_group` у BLS, квантование
эмбеддера в int8, переменная длина вместо фиксированных 128 токенов, ранний
отсев грубых сообщений до генератора.

## 10. Сырые логи

Все прогоны Perf Analyzer: [`raw/`](raw/), сводка: [`summary.csv`](summary.csv).

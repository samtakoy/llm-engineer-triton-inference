"""BLS-дирижёр: отсев грубых сообщений, затем поиск по корпусу и генерация."""

import json
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

from bls_prompt import build_messages

NON_TOXIC_THRESHOLD = 0.5
TOP_K = 3
MAX_LENGTH = 128
REFUSAL_TEXT = "Переформулируйте вопрос, пожалуйста, и я помогу."
SAMPLING_PARAMETERS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "max_tokens": 256,
}

class TritonPythonModel:
    """Принимает вопрос, отвечает отказом либо ответом по документам."""

    def initialize(self, args):
        """Загружает словари и корпус.

            args: пути и конфиг от Triton.

        Возвращает: Ничего.
        """
        version_dir = Path(args["model_repository"]) / args["model_version"]

        tokenizers_dir = version_dir / "tokenizers"
        self.tokenizers = {
            "ru": AutoTokenizer.from_pretrained(str(tokenizers_dir / "ru")),
            "e5": AutoTokenizer.from_pretrained(str(tokenizers_dir / "e5")),
        }
        # токенизатор генератора нужен ради разметки ролей. 
        self.chat_tokenizer = AutoTokenizer.from_pretrained(str(tokenizers_dir / "qwen"))

        corpus_dir = version_dir / "corpus"
        self.documents = json.loads((corpus_dir / "documents.json").read_text(encoding = "utf-8"))
        self.vectors = np.load(corpus_dir / "vectors.npy")

    def _tokenize(self, name: str, text: str) -> dict:
        """Превращает строку в тензоры для модели.

            name: какой словарь взять, "ru" или "e5".
            text: строка для токенизации.

        Возвращает: Словарь массивов numpy.
        """
        encoded = self.tokenizers[name](
            [text],
            padding = "max_length",
            truncation = True,
            max_length = MAX_LENGTH,
            return_tensors = "np",
        )

        return {key: value.astype(np.int64) for key, value in encoded.items()}

    def _call(self, model_name: str, inputs: list, outputs: list):
        """Синхронно вызывает другую модель внутри Triton.

            model_name: имя модели в репозитории.
            inputs: входные тензоры.
            outputs: имена нужных выходов.

        Возвращает: Ответ вызванной модели.
        """
        request = pb_utils.InferenceRequest(
            model_name = model_name,
            requested_output_names = outputs,
            inputs = inputs,
        )
        response = request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())

        return response

    def _tensor(self, response, name: str) -> np.ndarray:
        """Достаёт именованный тензор из ответа модели.

            response: ответ модели.
            name: имя выходного тензора.

        Возвращает: Массив numpy.
        """
        return pb_utils.get_output_tensor_by_name(response, name).as_numpy()

    def _is_toxic(self, text: str) -> bool:
        """Проверяет сообщение на грубость.

            text: исходный текст пользователя.

        Возвращает: True, если сообщение грубое.
        """
        encoded = self._tokenize("ru", text)
        response = self._call(
            "toxicity_clf",
            [
                pb_utils.Tensor("input_ids", encoded["input_ids"]),
                pb_utils.Tensor("attention_mask", encoded["attention_mask"]),
                # у rubert архитектура BERT, этот вход обязателен;
                # предложение одно, поэтому все нули
                pb_utils.Tensor(
                    "token_type_ids",
                    encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])),
                ),
            ],
            ["logits"],
        )

        # сигмоида на каждую метку,
        # нулевая метка — «нетоксично»
        logits = self._tensor(response, "logits")[0]
        non_toxic_score = 1.0 / (1.0 + np.exp(-logits[0]))

        return bool(non_toxic_score < NON_TOXIC_THRESHOLD)

    def _search(self, text: str) -> list:
        """Находит документы, ближайшие к вопросу.

            text: исходный текст вопроса.

        Возвращает: Список из TOP_K документов.
        """
        encoded = self._tokenize("e5", f"query: {text}")
        response = self._call(
            "e5_embedder",
            [
                pb_utils.Tensor("input_ids", encoded["input_ids"]),
                pb_utils.Tensor("attention_mask", encoded["attention_mask"]),
            ],
            ["sentence_embedding"],
        )

        # усреднение и нормализация уже сделаны внутри ONNX-графа
        query_vector = self._tensor(response, "sentence_embedding")[0]

        similarities = self.vectors @ query_vector
        best = np.argsort(-similarities)[:TOP_K]

        return [self.documents[index] for index in best]

    def _generate(self, text: str, documents: list) -> str:
        """Собирает промпт и получает ответ от генератора.

            text: вопрос пользователя.
            documents: найденные документы.

        Возвращает: Текст ответа.
        """
        prompt = self.chat_tokenizer.apply_chat_template(
            build_messages(text, documents),
            tokenize = False,
            add_generation_prompt = True,
            enable_thinking = False,
        )

        request = pb_utils.InferenceRequest(
            model_name = "text_generator",
            requested_output_names = ["text_output"],
            inputs = [
                pb_utils.Tensor("text_input", np.array([prompt.encode("utf-8")], dtype = object)),
                pb_utils.Tensor("stream", np.array([False], dtype = bool)),
                pb_utils.Tensor(
                    "sampling_parameters",
                    np.array([json.dumps(SAMPLING_PARAMETERS).encode("utf-8")], dtype = object),
                ),
            ],
        )

        pieces = []

        for response in request.exec(decoupled = True):
            if response is None:
                continue
            if response.has_error():
                raise pb_utils.TritonModelException(response.error().message())
            tensor = pb_utils.get_output_tensor_by_name(response, "text_output")
            pieces.append(tensor.as_numpy().reshape(-1)[0].decode("utf-8"))

        return "".join(pieces)

    def execute(self, requests):
        """Обрабатывает запросы: отказ либо поиск с генерацией.

            requests: список запросов от Triton.

        Возвращает: Список ответов.
        """
        responses = []

        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            text = raw.reshape(-1)[0].decode("utf-8")

            if self._is_toxic(text):
                answer, route, found = REFUSAL_TEXT, "refused", []
            else:
                found = self._search(text)
                answer, route = self._generate(text, found), "rag"

            output_tensors = [
                pb_utils.Tensor("ANSWER", np.array([answer.encode("utf-8")], dtype = object)),
                pb_utils.Tensor("ROUTE", np.array([route.encode("utf-8")], dtype = object)),
                pb_utils.Tensor(
                    "DOCS",
                    np.array([";".join(doc["document_id"] for doc in found).encode("utf-8")], dtype = object),
                ),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors = output_tensors))

        return responses

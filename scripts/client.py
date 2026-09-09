"""Проверяет обе ветки пайплайна и печатает результат."""

import numpy as np
import tritonclient.grpc as grpcclient

QUESTIONS = [
    "где в Кисловодске покататься на канатной дороге",
    "куда сходить с детьми в Железноводске",
    "ты тупой бот, отвечай нормально",
]


def ask(client: grpcclient.InferenceServerClient, text: str) -> None:
    """Отправляет вопрос в BLS и печатает ветку, документы и ответ.

        client: подключённый клиент Triton.
        text: текст вопроса.

    Возвращает: Ничего.
    """
    # у BLS max_batch_size: 0, поэтому форма как в dims: [1].
    # Лишнее измерение пачки Triton не примет
    payload = grpcclient.InferInput("TEXT", [1], "BYTES")
    payload.set_data_from_numpy(np.array([text.encode("utf-8")], dtype = object))

    result = client.infer(model_name = "assistant_bls", inputs = [payload])

    route = result.as_numpy("ROUTE")[0].decode("utf-8")
    docs = result.as_numpy("DOCS")[0].decode("utf-8")
    answer = result.as_numpy("ANSWER")[0].decode("utf-8")

    print(f"[{route}] {text}")
    if docs:
        print(f"  документы: {docs}")
    print(f"  ответ: {answer}\n")


def main() -> None:
    """Прогоняет список вопросов через пайплайн.

    Возвращает: Ничего.
    """
    client = grpcclient.InferenceServerClient(url = "localhost:8001")
    for question in QUESTIONS:
        ask(client, question)


if __name__ == "__main__":
    main()

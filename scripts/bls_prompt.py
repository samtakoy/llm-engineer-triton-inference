"""Сборка промпта для генератора. Общая для BLS и для замеров."""

SYSTEM_PROMPT = (
    "Ты помощник по региону Кавказские Минеральные Воды. "
    "Отвечай кратко и только по приведённым документам. "
    "Если в документах ответа нет, так и скажи."
)


def build_messages(question: str, documents: list) -> list:
    """Собирает диалог из системной инструкции и найденных документов.

        question: вопрос пользователя.
        documents: найденные документы.

    Возвращает: Список сообщений в формате чата.
    """
    context = "\n\n".join(
        f"[{index + 1}] {doc['name']} ({doc['city']}): {doc['context_text']}"
        for index, doc in enumerate(documents)
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Документы:\n{context}\n\nВопрос: {question}"},
    ]

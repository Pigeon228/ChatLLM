from uuid import uuid4
from typing import Dict, List
from functools import lru_cache

from fastapi import HTTPException

from ..models.schemas import Msg
from ..core.config import get_settings

_settings = get_settings()

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_PROMPT = ""

# id → {title, messages, model, temperature, stream, prompt}
_chat_store: Dict[str, Dict] = {}


# ──────────── CRUD ────────────

def create_chat() -> dict:
    """Создаёт новый чат и возвращает его метаданные."""
    pk = uuid4().hex
    _chat_store[pk] = {
        "title": f"Чат {len(_chat_store) + 1}",
        "messages": [],
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "stream": True,
        "prompt": DEFAULT_PROMPT,
    }
    return _chat_meta(pk)


def delete_chat(cid: str) -> None:
    if cid not in _chat_store:
        raise HTTPException(404, "chat not found")
    _chat_store.pop(cid)


# ─────────── helpers ───────────

def _chat_meta(pk: str) -> dict:
    obj = _chat_store[pk]
    return {
        "id": pk,
        "title": obj["title"],
        "model": obj["model"],
        "temperature": obj["temperature"],
        "stream": obj["stream"],
        "prompt": obj["prompt"],
    }


def list_chats() -> List[dict]:
    return [_chat_meta(cid) for cid in _chat_store]


def get_messages(cid: str) -> List[Msg]:
    return _chat_store[cid]["messages"]


@lru_cache
def list_models() -> List[str]:
    """
    Возвращает список идентификаторов моделей, доступных в OpenRouter.
    Кэшируем, чтобы не дёргать API каждый раз.
    """
    return [m.id for m in _settings.client.models.list().data]


# ─────────── LLM wrapper ───────────

def _llm_answer(messages: List[dict], model: str, temperature: float) -> str:
    completion = _settings.client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content


def _llm_answer_stream(messages: List[dict], model: str, temperature: float):
    completion = _settings.client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in completion:
        part = chunk.choices[0].delta.content
        if part:
            yield part


# ─────────── messaging ───────────

def _choose(chat: Dict, model: str | None, temperature: float | None) -> tuple[str, float]:
    """Определяем, какую модель и температуру использовать."""
    chosen_model = model or chat["model"] or DEFAULT_MODEL
    chosen_temp = temperature if temperature is not None else chat["temperature"]
    return (chosen_model, chosen_temp)


def _messages_with_prompt(chat: Dict, messages: List[dict] | None = None) -> List[dict]:
    msgs = messages if messages is not None else chat["messages"]
    prompt = chat.get("prompt")
    return ([{"role": "system", "content": prompt}] + msgs) if prompt else msgs


def send_message(cid: str, user_text: str,
                 model: str | None, temperature: float | None) -> str:
    chat = _chat_store[cid]
    chat["messages"].append({"role": "user", "content": user_text})

    m, t = _choose(chat, model, temperature)
    answer = _llm_answer(_messages_with_prompt(chat), m, t)
    chat["messages"].append({"role": "assistant", "content": answer})

    if model:
        chat["model"] = model
    if temperature is not None:
        chat["temperature"] = temperature
    return answer


def stream_message(cid: str, user_text: str,
                   model: str | None, temperature: float | None):
    chat = _chat_store[cid]
    chat["messages"].append({"role": "user", "content": user_text})

    m, t = _choose(chat, model, temperature)
    parts = []
    for part in _llm_answer_stream(_messages_with_prompt(chat), m, t):
        parts.append(part)
        yield part
    answer = "".join(parts)
    chat["messages"].append({"role": "assistant", "content": answer})

    if model:
        chat["model"] = model
    if temperature is not None:
        chat["temperature"] = temperature


def edit_message(cid: str, idx: int, new_content: str,
                 model: str | None, temperature: float | None) -> str:
    messages = _chat_store[cid]["messages"]

    if not 0 <= idx < len(messages):
        raise HTTPException(404, "message not found")

    messages[idx]["content"] = new_content
    role = messages[idx]["role"]

    # редактируем ответ ассистента → только обновляем текст
    if role == "assistant":
        return new_content

    # редактируем сообщение пользователя → отбрасываем хвост и генерируем новый ответ
    del messages[idx + 1 :]

    chat = _chat_store[cid]
    m, t = _choose(chat, model, temperature)
    answer = _llm_answer(_messages_with_prompt(chat, messages), m, t)
    messages.append({"role": "assistant", "content": answer})

    if model:
        chat["model"] = model
    if temperature is not None:
        chat["temperature"] = temperature
    return answer


def regenerate_assistant(cid: str, idx: int,
                         model: str | None, temperature: float | None) -> str:
    messages = _chat_store[cid]["messages"]

    if not 0 <= idx < len(messages):
        raise HTTPException(404, "message not found")
    if messages[idx]["role"] != "assistant":
        raise HTTPException(400, "can regenerate only assistant messages")

    chat = _chat_store[cid]
    m, t = _choose(chat, model, temperature)

    context = messages[:idx]      # всё до ответа бота
    answer = _llm_answer(_messages_with_prompt(chat, context), m, t)
    messages[idx]["content"] = answer

    if model:
        chat["model"] = model
    if temperature is not None:
        chat["temperature"] = temperature
    return answer


def update_chat(cid: str, title: str, model: str, temperature: float, stream: bool, prompt: str) -> dict:
    if cid not in _chat_store:
        raise HTTPException(404, "chat not found")
    _chat_store[cid].update({
        "title": title,
        "model": model or DEFAULT_MODEL,
        "temperature": temperature,
        "stream": stream,
        "prompt": prompt,
    })
    return _chat_meta(cid)

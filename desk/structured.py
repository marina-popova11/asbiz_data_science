# -*- coding: utf-8 -*-
"""Ответ по схеме

Просим у модели JSON, проверяем его схемой, а при ошибке возвращаем модели её
ответ и текст ошибки и просим исправить. Строгий режим провайдера проверяет
синтаксис, но смысл ответа проверяет только этот цикл
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredError(Exception):
    """Ответ не прошёл проверку ни с одной попытки"""

    def __init__(self, attempts: int, last_error: str, last_text: str):
        super().__init__(
            "ответ не прошёл проверку за %d попыток: %s" % (attempts, last_error)
        )
        self.attempts, self.last_error, self.last_text = attempts, last_error, last_text


def extract_json(text: str) -> Dict[str, Any]:
    """Первый объект JSON в тексте, даже если вокруг него слова или ограда ```"""
    start = text.find("{")
    if start == -1:
        raise ValueError("нет объекта JSON")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError("нет объекта JSON")

def format_errors(err: Exception) -> str:
    """Ошибки проверки одной строкой: поле и причина"""
    if isinstance(err, ValidationError):
        return "; ".join(
            "поле %s: %s" % (".".join(map(str, e["loc"])) or "?", e["msg"])
            for e in err.errors()
        )
    return str(err)


def _complaint(err: Exception) -> str:
    return (
        "Ответ не прошёл проверку: %s. Верни исправленный JSON целиком, без пояснений."
        % format_errors(err)
    )


def structured(
    llm: Any,
    messages: List[dict],
    schema: Type[T],
    *,
    context: Optional[dict] = None,
    max_attempts: int = 3,
    **kw: Any,
) -> Tuple[T, int]:
    """Ответ модели по схеме и число попыток

    Если за max_attempts попыток ответ не прошёл проверку, бросает StructuredError
    """
    history = list(messages)
    err: Exception = ValueError("не было ни одной попытки")
    text = ""
    for attempt in range(1, max_attempts + 1):
        text = llm.chat(history, **kw).text
        try:
            return schema.model_validate(extract_json(text), context=context), attempt
        except (ValueError, ValidationError) as e:
            err = e
            history += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": _complaint(e)},
            ]
    raise StructuredError(max_attempts, format_errors(err), text)


async def astructured(
    llm: Any,
    messages: List[dict],
    schema: Type[T],
    *,
    context: Optional[dict] = None,
    max_attempts: int = 3,
    **kw: Any,
) -> Tuple[T, int]:
    """То же, что structured, но асинхронно"""
    history = list(messages)
    err: Exception = ValueError("не было ни одной попытки")
    text = ""
    for attempt in range(1, max_attempts + 1):
        text = (await llm.achat(history, **kw)).text
        try:
            return schema.model_validate(extract_json(text), context=context), attempt
        except (ValueError, ValidationError) as e:
            err = e
            history += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": _complaint(e)},
            ]
    raise StructuredError(max_attempts, format_errors(err), text)


# -*- coding: utf-8 -*-
"""Раунд 1. Запрос к модели без обёртки

    python 1_raw_request.py

Скрипт отправляет несколько запросов прямо через httpx и печатает то,
что обычно прячет SDK: блоки ответа, причину остановки и счёт токенов.
Наблюдения перенесите в отчёт_1.md
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx

from desk.config import Settings, settings
from desk.llm import RateLimit

import time
SYSTEM = (
    "Отнеси обращение в поддержку платёжного сервиса к одной категории: платежи, "
    "возвраты, доступ, тарифы, интеграция, другое. Ответь одним словом."
)
TICKET = "с меня два раза сняли 3490 за один заказ!!! разберитесь"

# Одна и та же фраза по-русски и по-английски
PAIRS = [
    (
        "Возврат на карту занимает от трёх до десяти рабочих дней "
        "и зависит от банка клиента.",
        "A refund to the card takes three to ten business days "
        "and depends on the client's bank.",
    ),
    (
        "Суточный лимит обнуляется в полночь по московскому времени.",
        "The daily limit resets at midnight Moscow time.",
    ),
]


def headers(cfg: Settings) -> Dict[str, str]:
    """Заголовки запроса: версия формата, тип содержимого и ключ"""
    out = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    if cfg.auth_scheme == "bearer":
        out["authorization"] = "Bearer " + cfg.api_key
    else:
        out["x-api-key"] = cfg.api_key
    return out


def body(
    cfg: Settings,
    messages: List[Dict[str, Any]],
    *,
    system: Optional[str] = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Тело запроса к модели"""
    out: Dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        out["system"] = system
    out.update(cfg.extra_body)
    out.update(extra or {})
    return out


def send(
    cfg: Settings, payload: Dict[str, Any], client: Optional[httpx.Client] = None
) -> Dict[str, Any]:
    """Отправляет запрос и возвращает ответ шлюза

    Если код ответа не 200, бросает ошибку с текстом ответа. Тесты
    передают свой client, чтобы не ходить в сеть
    """
    own = client is None
    client = client or httpx.Client(timeout=cfg.timeout_s, verify=cfg.verify_ssl)
    try:
        resp = client.post(
            cfg.base_url + "/v1/messages", headers=headers(cfg), json=payload
        )
    finally:
        if own:
            client.close()
    if resp.status_code != 200:
        raise RuntimeError("шлюз ответил %d: %s" % (resp.status_code, resp.text[:300]))
    return resp.json()

def send_with_retry(
    cfg: Settings,
    payload: Dict[str, Any],
    client: Optional[httpx.Client] = None,
    attempts: int = 8,
    base_delay: float = 7.0,
) -> Dict[str, Any]:
    """Повторяет запрос при 429 с нарастающей паузой"""
    last: Optional[RuntimeError] = None
    for n in range(attempts):
        try:
            return send(cfg, payload, client=client)
        except RuntimeError as e:
            last = e
            if "429" not in str(e):
                raise
            time.sleep(base_delay * (n + 1))
    raise last  # type: ignore[misc]

def text_of(resp: Dict[str, Any]) -> str:
    """Текст ответа из всех блоков text"""
    # parts = []
    # for b in resp.get("content") or []:
    #     if b.get("type") != "text":
    #         continue
    #     t = b.get("text", "")
    #     if isinstance(t, str):
    #         parts.append(t)
    #     elif isinstance(t, dict):
    #         # нестандартный формат: берём первое строковое значение
    #         parts.append(str(next((v for v in t.values() if isinstance(v, str)), "")))
    #     else:
    #         parts.append(str(t))
    # return "".join(parts).strip()
    return "".join(
        b.get("text", "") for b in resp.get("content") or [] if b.get("type") == "text"
    ).strip()


def user(text: str) -> List[Dict[str, Any]]:
    """Сообщение пользователя"""
    return [{"role": "user", "content": text}]


def main() -> None:
    cfg = settings()
    if not cfg.base_url or not cfg.api_key:
        raise SystemExit("впишите LLM_BASE_URL и LLM_API_KEY в .env")
    limit = RateLimit(cfg.rpm)

    def call(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Отправка с паузой, если у ключа лимит запросов в минуту"""
        time.sleep(limit.reserve())
        return send(cfg, payload)

    print("1. Обычный запрос: весь ответ как есть\n")
    resp = call(body(cfg, user(TICKET), system=SYSTEM, max_tokens=16))
    print(json.dumps(resp, ensure_ascii=False, indent=2))

    print("\n2. Ответ, оборванный лимитом max_tokens=2\n")
    resp = call(body(cfg, user("Объясни, почему небо голубое."), max_tokens=2))
    print("stop_reason: %s, текст: %r" % (resp.get("stop_reason"), text_of(resp)))

    print("\n3. Токены русского и английского текста с одним смыслом\n")
    for ru, en in PAIRS:
        n_ru = call(body(cfg, user(ru), max_tokens=1))["usage"]["input_tokens"]
        n_en = call(body(cfg, user(en), max_tokens=1))["usage"]["input_tokens"]
        print(
            "по-русски %3d, по-английски %3d, отношение %.2f"
            % (n_ru, n_en, n_ru / max(1, n_en))
        )

    print("\n4. Разброс ответов: пять одинаковых запросов при температуре 0 и 1\n")
    prompt = user("Продолжи фразу одним словом, без точки: «Сегодня погода»")
    for temperature in (0.0, 1.0):
        answers = [
            text_of(call(body(cfg, prompt, max_tokens=8, temperature=temperature)))
            for _ in range(5)
        ]
        print(
            "температура %.1f: различных ответов %d из 5: %s"
            % (temperature, len(set(answers)), answers)
        )

    print("\n5. Сервер не хранит диалог\n")
    first = user("Меня зовут Аня. Запомни это.")
    resp1 = call(body(cfg, first, max_tokens=40))
    alone = call(body(cfg, user("Как меня зовут? Ответь одним словом."), max_tokens=16))
    history = first + [
        {"role": "assistant", "content": text_of(resp1)},
        {"role": "user", "content": "Как меня зовут? Ответь одним словом."},
    ]
    with_history = call(body(cfg, history, max_tokens=16))
    print(
        "без истории: %r, входных токенов %d"
        % (text_of(alone), alone["usage"]["input_tokens"])
    )
    print(
        "с историей:  %r, входных токенов %d"
        % (text_of(with_history), with_history["usage"]["input_tokens"])
    )

    print("\n6. Что будет, если модель начнёт рассуждать, а места мало\n")
    thinking = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    resp = call(body(cfg, user(TICKET), system=SYSTEM, max_tokens=16, extra=thinking))
    kinds = [b["type"] for b in resp.get("content") or []]
    print(
        "блоки ответа: %s, причина остановки: %s, текст: %r, выходных токенов %d"
        % (
            kinds,
            resp.get("stop_reason"),
            text_of(resp),
            resp["usage"]["output_tokens"],
        )
    )
    print(
        "Поэтому в проекте рассуждение выключено добавкой из .env, а кандидат с "
        "рассуждением в замере получает max_tokens больше бюджета рассуждения."
    )


if __name__ == "__main__":
    main()

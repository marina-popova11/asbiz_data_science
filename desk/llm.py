# -*- coding: utf-8 -*-
"""Клиент к модели

Все вызовы модели в проекте идут через класс LLM. Он собирает запрос
в формате Anthropic Messages, повторяет его при сбоях, считает токены
и деньги и хранит кэш ответов
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Tuple

import httpx

from .config import Settings
from .config import settings as load_settings

if TYPE_CHECKING:
    from .stream import StreamResult

# После этих кодов запрос повторяем. 529 значит, что провайдер перегружен
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
ANTHROPIC_VERSION = "2023-06-01"
WINDOW_S = 61.0
_THINK = re.compile(r"<think>.*?</think>", re.S)


class RateLimit:
    """Не даёт отправить больше rpm запросов за минуту"""

    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic):
        self.rpm = rpm
        self.clock = clock
        self.sent: Deque[float] = deque()

    def reserve(self) -> float:
        """Занимает место для запроса и говорит, сколько секунд ждать до отправки"""
        if self.rpm <= 0:
            return 0.0
        now = self.clock()
        while self.sent and self.sent[0] <= now - WINDOW_S:
            self.sent.popleft()
        at = now
        if len(self.sent) >= self.rpm:
            at = max(now, self.sent[-self.rpm] + WINDOW_S)
        self.sent.append(at)
        return at - now


class LLMError(Exception):
    """Ошибка, которую не исправит повтор запроса"""


class LLMUnavailable(LLMError):
    """Модель не ответила ни на одну попытку"""


@dataclass
class Usage:
    """Счёт одного вызова модели"""

    input_tokens: int = 0  # без кэша
    output_tokens: int = 0  # вместе с рассуждением
    cache_read_tokens: int = 0  # прочитано из кэша провайдера
    cache_write_tokens: int = 0  # записано в кэш провайдера
    latency_s: float = 0.0
    cost: float = 0.0  # в условных единицах
    weighted: float = 0.0  # в них шлюз считает лимит
    calls: int = 0
    retries: int = 0
    cached: int = 0  # ответ взят из нашего кэша

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            *(getattr(self, f) + getattr(other, f) for f in self.__dataclass_fields__)
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.output_tokens
        )


@dataclass
class ToolCall:
    """Просьба модели вызвать инструмент"""

    id: str
    name: str
    input: Any  # аргументы, обычно словарь

    @property
    def arguments(self) -> str:
        """Аргументы одной строкой"""
        if isinstance(self.input, str):
            return self.input
        return json.dumps(self.input, ensure_ascii=False, sort_keys=True)


@dataclass
class Reply:
    """Ответ модели"""

    text: str
    tool_calls: List[ToolCall]
    usage: Usage
    stop_reason: str = "end_turn"
    content: List[Dict[str, Any]] = field(default_factory=list)  # блоки как есть
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def truncated(self) -> bool:
        """Ответ оборван по max_tokens"""
        return self.stop_reason == "max_tokens"

    def as_message(self) -> Dict[str, Any]:
        """Ответ как сообщение для истории диалога"""
        if self.content:
            return {"role": "assistant", "content": self.content}
        return {"role": "assistant", "content": self.text or "(пустой ответ)"}


def tool_result(
    call: ToolCall, observation: Any, is_error: bool = False
) -> Dict[str, Any]:
    """Результат инструмента для следующего запроса"""
    content = (
        observation
        if isinstance(observation, str)
        else json.dumps(observation, ensure_ascii=False)
    )
    block: Dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


def split_system(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Отделяет системную постановку от остальных сообщений"""
    system, rest = [], []
    for m in messages:
        if m.get("role") == "system":
            if rest:
                raise ValueError(
                    "системная инструкция должна стоять в начале списка сообщений"
                )
            system.append(m["content"])
        else:
            rest.append(m)
    return "\n\n".join(system), rest


def backoff_delay(
    attempt: int,
    *,
    base: float = 0.5,
    cap: float = 20.0,
    retry_after: Optional[float] = None,
    rnd: Callable[[], float] = random.random,
) -> float:
    """Пауза перед повтором

    С каждой попыткой пауза растёт вдвое, но не больше cap, и умножается
    на случайное число от 0,5 до 1,5. Если провайдер сам назвал срок
    в Retry-After, ждём столько
    """
    if retry_after is not None:
        return min(cap, max(0.0, retry_after))
    return min(cap, base * (2**attempt)) * (0.5 + rnd())


def should_retry(status: Optional[int], exc: Optional[BaseException]) -> bool:
    """Поможет ли повтор после такой ошибки"""
    if exc is not None:
        return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))
    return status in RETRY_STATUS


def cache_key(payload: Dict[str, Any]) -> str:
    """Ключ кэша: хэш запроса. Порядок полей на него не влияет"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class LLM:
    """Клиент к модели через шлюз курса"""

    def __init__(
        self,
        cfg: Optional[Settings] = None,
        *,
        max_retries: int = 4,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
        cache: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg or load_settings()
        if not self.cfg.base_url:
            raise LLMError("не задан адрес шлюза: впишите LLM_BASE_URL в .env")
        self.max_retries = max_retries
        self.sleep = sleep
        self.cache = cache and self.cfg.cache_dir is not None
        self.ledger: List[Usage] = []
        self.limit = RateLimit(self.cfg.rpm, clock)
        headers = {
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if self.cfg.auth_scheme == "bearer":
            headers["authorization"] = "Bearer " + self.cfg.api_key
        else:
            headers["x-api-key"] = self.cfg.api_key
        self._common = dict(
            base_url=self.cfg.base_url,
            headers=headers,
            timeout=self.cfg.timeout_s,
            verify=self.cfg.verify_ssl,
        )
        self._client = httpx.Client(transport=transport, **self._common)
        self._async_transport = async_transport
        self._aclient: Optional[httpx.AsyncClient] = None
        self._aloop: Any = None

    def _async_client(self) -> httpx.AsyncClient:
        """Асинхронный клиент для текущего цикла событий"""
        import asyncio

        loop = asyncio.get_running_loop()
        if self._aclient is None or self._aloop is not loop:
            self._aclient = httpx.AsyncClient(
                transport=self._async_transport, **self._common
            )
            self._aloop = loop
        return self._aclient

    def variant(self, **changes: Any) -> "LLM":
        """Копия клиента с другими настройками, например с другой моделью"""
        if "base_url" in changes or "api_key" in changes:
            raise ValueError(
                "для другого провайдера создайте отдельный LLM(Settings(...))"
            )
        other = LLM.__new__(LLM)
        other.__dict__.update(self.__dict__)
        other.cfg = replace(self.cfg, **changes)
        other.ledger = []
        return other

    # Запрос и ответ

    def payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tool_choice: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Тело запроса к модели

        Если в добавке из настроек включено рассуждение, температуру
        убираем: с рассуждением провайдер её не принимает
        """
        system, dialog = split_system(messages)
        body: Dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": dialog,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = {"type": tool_choice}
        if stream:
            body["stream"] = True
        body.update(self.cfg.extra_body)
        if (body.get("thinking") or {}).get("type") in ("enabled", "adaptive"):
            body.pop("temperature", None)
        return body

    def usage_from(self, u: Dict[str, Any], latency_s: float, retries: int) -> Usage:
        """Счёт вызова по полю usage: токены, деньги, взвешенные токены"""
        cfg = self.cfg
        inp = int(u.get("input_tokens") or 0)
        write = int(u.get("cache_creation_input_tokens") or 0)
        read = int(u.get("cache_read_input_tokens") or 0)
        out = int(u.get("output_tokens") or 0)
        billed_in = inp + cfg.weight_cache_write * write + cfg.weight_cache * read
        cost = (billed_in * cfg.price_in + out * cfg.price_out) / 1_000_000
        weighted = billed_in + cfg.weight_out * out
        return Usage(
            inp, out, read, write, latency_s, cost, weighted, calls=1, retries=retries
        )

    def parse(self, data: Dict[str, Any], latency_s: float, retries: int) -> Reply:
        """Разбирает ответ модели: текст, вызовы инструментов, счёт"""
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = [
            ToolCall(b.get("id") or "toolu_%d" % i, b["name"], b.get("input") or {})
            for i, b in enumerate(blocks)
            if b.get("type") == "tool_use"
        ]
        usage = self.usage_from(data.get("usage") or {}, latency_s, retries)
        text = _THINK.sub("", text).strip()
        return Reply(
            text, calls, usage, data.get("stop_reason") or "end_turn", blocks, data
        )

    # Повторы

    def pause_or_raise(
        self,
        attempt: int,
        status: Optional[int],
        exc: Optional[BaseException],
        resp: Optional[httpx.Response],
    ) -> float:
        """Что делать после неудачной попытки

        Возвращает паузу перед следующей попыткой. Если повтор не поможет
        или попытки кончились, бросает ошибку с началом ответа провайдера
        """
        detail = resp.text[:300] if resp is not None else repr(exc)
        if not should_retry(status, exc):
            raise LLMError("провайдер ответил %s: %s" % (status, detail))
        if attempt >= self.max_retries:
            raise LLMUnavailable(
                "нет ответа после %d попыток (последний ответ %s: %s)"
                % (self.max_retries + 1, status, detail)
            )
        return backoff_delay(attempt, retry_after=retry_after(resp))

    def _post(
        self, path: str, body: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[BaseException], Optional[httpx.Response]]:
        """Отправка запроса. Возвращает код ответа, ошибку сети и сам ответ"""
        wait = self.limit.reserve()
        if wait:
            self.sleep(wait)
        try:
            resp = self._client.post(path, json=body)
            return resp.status_code, None, resp
        except httpx.HTTPError as e:
            return None, e, None

    async def _apost(
        self, path: str, body: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[BaseException], Optional[httpx.Response]]:
        """То же, что _post, но асинхронно"""
        import asyncio

        wait = self.limit.reserve()
        if wait:
            await asyncio.sleep(wait)
        try:
            resp = await self._async_client().post(path, json=body)
            return resp.status_code, None, resp
        except httpx.HTTPError as e:
            return None, e, None

    def _record(self, body: Dict[str, Any], reply: Reply) -> Reply:
        """Записывает вызов в журнал и в кэш"""
        self.ledger.append(reply.usage)
        self._cache_put(body, reply)
        return reply

    # Обычный вызов

    def chat(self, messages: List[Dict[str, Any]], **kw: Any) -> Reply:
        """Отправляет запрос модели и возвращает ответ

        Если такой запрос уже был, берёт ответ из кэша. При сбоях повторяет
        """
        body = self.payload(messages, **kw)
        hit = self._cache_get(body)
        if hit is not None:
            return hit
        started = time.perf_counter()
        attempt = 0
        while True:
            status, exc, resp = self._post("/v1/messages", body)
            if status == 200:
                reply = self.parse(resp.json(), time.perf_counter() - started, attempt)
                return self._record(body, reply)
            self.sleep(self.pause_or_raise(attempt, status, exc, resp))
            attempt += 1

    # Асинхронный вызов

    async def achat(self, messages: List[Dict[str, Any]], **kw: Any) -> Reply:
        """То же, что chat, но асинхронно"""
        import asyncio

        body = self.payload(messages, **kw)
        hit = self._cache_get(body)
        if hit is not None:
            return hit
        started = time.perf_counter()
        attempt = 0
        while True:
            status, exc, resp = await self._apost("/v1/messages", body)
            if status == 200:
                reply = self.parse(resp.json(), time.perf_counter() - started, attempt)
                return self._record(body, reply)
            await asyncio.sleep(self.pause_or_raise(attempt, status, exc, resp))
            attempt += 1

    # Потоковая выдача

    def stream(self, messages: List[Dict[str, Any]], **kw: Any) -> "StreamResult":
        """Запрос с потоковой выдачей

        Повторяем, только пока ответ не начал приходить: оборванный
        посередине поток уже оплачен
        """
        from .stream import collect, iter_sse

        body = self.payload(messages, stream=True, **kw)
        started = time.perf_counter()
        attempt = 0
        while True:
            status, exc, resp = None, None, None
            try:
                with self._client.stream("POST", "/v1/messages", json=body) as r:
                    status = r.status_code
                    if status == 200:
                        result = collect(
                            iter_sse(r.iter_lines()),
                            time.perf_counter,
                            started,
                            self.usage_from,
                            attempt,
                        )
                        self.ledger.append(result.usage)
                        return result
                    r.read()
                    resp = r
            except httpx.HTTPError as e:
                exc = e
            self.sleep(self.pause_or_raise(attempt, status, exc, resp))
            attempt += 1

    async def astream(
        self, messages: List[Dict[str, Any]], **kw: Any
    ) -> "StreamResult":
        """То же, что stream, но асинхронно"""
        import asyncio

        from .stream import collect, iter_sse

        body = self.payload(messages, stream=True, **kw)
        attempt = 0
        while True:
            wait = self.limit.reserve()
            if wait:
                await asyncio.sleep(wait)
            if attempt == 0:
                started = time.perf_counter()
            status, exc, resp = None, None, None
            try:
                async with self._async_client().stream(
                    "POST", "/v1/messages", json=body
                ) as r:
                    status = r.status_code
                    if status == 200:
                        stamped = [
                            (time.perf_counter(), line)
                            async for line in r.aiter_lines()
                        ]
                        now = {"t": started}

                        def replay():
                            for moment, line in stamped:
                                now["t"] = moment
                                yield line

                        result = collect(
                            iter_sse(replay()),
                            lambda: now["t"],
                            started,
                            self.usage_from,
                            attempt,
                        )
                        self.ledger.append(result.usage)
                        return result
                    await r.aread()
                    resp = r
            except httpx.HTTPError as e:
                exc = e
            await asyncio.sleep(self.pause_or_raise(attempt, status, exc, resp))
            attempt += 1

    # Учёт и кэш

    def total(self) -> Usage:
        """Сумма по всем вызовам"""
        out = Usage()
        for u in self.ledger:
            out = out + u
        return out

    def _cache_get(self, body: Dict[str, Any]) -> Optional[Reply]:
        if not self.cache or body.get("temperature", 0) != 0:
            return None
        path = self.cfg.cache_dir / (cache_key(body) + ".json")
        if not path.exists():
            return None
        reply = self.parse(json.loads(path.read_text(encoding="utf-8")), 0.0, 0)
        reply.usage = replace(reply.usage, cost=0.0, weighted=0.0, cached=1)
        self.ledger.append(reply.usage)
        return reply

    def _cache_put(self, body: Dict[str, Any], reply: Reply) -> None:
        if not self.cache or body.get("temperature", 0) != 0:
            return
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.cache_dir / (cache_key(body) + ".json")
        path.write_text(json.dumps(reply.raw, ensure_ascii=False), encoding="utf-8")


def retry_after(resp: Optional[httpx.Response]) -> Optional[float]:
    """Сколько секунд просит подождать провайдер, если он это указал"""
    if resp is None:
        return None
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None

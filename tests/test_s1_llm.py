# -*- coding: utf-8 -*-
"""Семинар 1. Клиент к модели: формат запроса, повторы, учёт, кэш. Сеть не нужна"""
import json

import httpx
import pytest

from desk.config import Settings
from desk.llm import (LLM, LLMError, LLMUnavailable, RateLimit, backoff_delay, cache_key,
                      retry_after, should_retry, split_system, tool_result)

OK = {"type": "message", "role": "assistant", "model": "deepseek-flash",
      "content": [{"type": "text", "text": "платежи"}], "stop_reason": "end_turn",
      "usage": {"input_tokens": 1000, "output_tokens": 500}}


def make(handler, tmp_path=None, **kw):
    cfg = Settings("https://gw.example", "key", "deepseek-flash", price_in=2.0, price_out=8.0,
                   weight_out=4.0, weight_cache=0.1, cache_dir=tmp_path)
    sleeps = []
    llm = LLM(cfg, transport=httpx.MockTransport(handler), sleep=sleeps.append, **kw)
    return llm, sleeps


# Формат запроса

def test_request_goes_to_messages_endpoint_with_headers():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=OK)

    llm, _ = make(handler)
    llm.chat([{"role": "system", "content": "Отнеси к категории."},
              {"role": "user", "content": "Списали дважды"}], max_tokens=16)
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.path == "/v1/messages"
    assert req.headers["x-api-key"] == "key" and req.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "Отнеси к категории."
    assert body["messages"] == [{"role": "user", "content": "Списали дважды"}]
    assert body["model"] == "deepseek-flash" and body["max_tokens"] == 16


def test_bearer_scheme_for_gateways_that_want_it():
    seen = []
    cfg = Settings("https://gw.example", "key", auth_scheme="bearer")
    llm = LLM(cfg, transport=httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200, json=OK)),
              cache=False)
    llm.chat([{"role": "user", "content": "x"}])
    assert seen[0].headers["authorization"] == "Bearer key" and "x-api-key" not in seen[0].headers


def test_system_must_come_first():
    assert split_system([{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
                         {"role": "user", "content": "q"}]) == ("a\n\nb", [{"role": "user", "content": "q"}])
    with pytest.raises(ValueError):
        split_system([{"role": "user", "content": "q"}, {"role": "system", "content": "поздно"}])


def test_payload_has_max_tokens_and_extra_body():
    cfg = Settings("https://gw.example", "key", extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}})
    body = LLM(cfg, cache=False).payload([{"role": "user", "content": "x"}])
    assert body["max_tokens"] > 0 and "temperature" not in body     # с рассуждением температуры нет
    assert body["thinking"]["type"] == "enabled" and "system" not in body


def test_thinking_drops_temperature():
    cfg = Settings("https://gw.example", "key", extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}})
    assert "temperature" not in LLM(cfg, cache=False).payload([{"role": "user", "content": "x"}])
    off = Settings("https://gw.example", "key", extra_body={"thinking": {"type": "disabled"}})
    assert LLM(off, cache=False).payload([{"role": "user", "content": "x"}])["temperature"] == 0


def test_missing_base_url_is_reported():
    with pytest.raises(LLMError):
        LLM(Settings("", "key"), cache=False)


# Повторы

def test_backoff_grows_and_is_capped():
    mid = lambda: 0.5                                     # разброс ровно 1.0
    assert [backoff_delay(a, base=1, cap=10, rnd=mid) for a in range(5)] == [1, 2, 4, 8, 10]


def test_backoff_has_jitter():
    lo = backoff_delay(2, base=1, cap=100, rnd=lambda: 0.0)
    hi = backoff_delay(2, base=1, cap=100, rnd=lambda: 1.0)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(6.0)


def test_backoff_respects_retry_after():
    assert backoff_delay(0, retry_after=7) == 7
    assert backoff_delay(0, cap=5, retry_after=60) == 5


@pytest.mark.parametrize("status,expected", [(429, True), (500, True), (503, True), (529, True),
                                             (400, False), (401, False), (403, False), (404, False)])
def test_retry_only_what_can_heal(status, expected):
    assert should_retry(status, None) is expected


def test_retry_on_timeout_but_not_on_bug():
    assert should_retry(None, httpx.ReadTimeout("t"))
    assert should_retry(None, httpx.ConnectError("c"))
    assert not should_retry(None, ValueError("ошибка в нашем коде"))


def test_retries_then_succeeds():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) < 3:
            return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}})
        return httpx.Response(200, json=OK)

    llm, sleeps = make(handler)
    reply = llm.chat([{"role": "user", "content": "привет"}])
    assert reply.text == "платежи"
    assert len(seen) == 3 and len(sleeps) == 2
    assert reply.usage.retries == 2


def test_gives_up_after_max_retries_and_tells_why():
    llm, sleeps = make(lambda r: httpx.Response(503, text="DeepSeek недоступен в часы пиковой нагрузки",
                                                headers={"retry-after": "1"}), max_retries=2)
    with pytest.raises(LLMUnavailable) as e:
        llm.chat([{"role": "user", "content": "привет"}])
    assert sleeps == [1, 1] and "пиковой нагрузки" in str(e.value) and "503" in str(e.value)


def test_client_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error"}})

    llm, sleeps = make(handler)
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "привет"}])
    assert len(calls) == 1 and sleeps == []


# Учёт

def test_usage_cost_and_weighted_tokens_are_counted():
    llm, _ = make(lambda r: httpx.Response(200, json=OK))   # веса: выход 4, кэш 0,1 и 1,25
    llm.chat([{"role": "user", "content": "раз"}])
    llm.chat([{"role": "user", "content": "два"}])
    total = llm.total()
    assert (total.calls, total.input_tokens, total.output_tokens) == (2, 2000, 1000)
    # 2000 входных по 2.0 и 1000 выходных по 8.0 за миллион
    assert total.cost == pytest.approx((2000 * 2.0 + 1000 * 8.0) / 1e6)
    # взвешенные: вход + 4 × выход
    assert total.weighted == pytest.approx(2000 + 4 * 1000)


def test_cache_tokens_have_their_own_weights():
    data = dict(OK, usage={"input_tokens": 100, "cache_read_input_tokens": 900,
                           "cache_creation_input_tokens": 200, "output_tokens": 10})
    llm, _ = make(lambda r: httpx.Response(200, json=data))
    u = llm.chat([{"role": "user", "content": "x"}]).usage
    assert (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens) == (100, 900, 200)
    assert u.total_tokens == 1210
    # чтение из кэша дешевле обычного входа, запись дороже
    assert u.weighted == pytest.approx(100 + 0.1 * 900 + 1.25 * 200 + 4 * 10)
    assert u.cost == pytest.approx(((100 + 0.1 * 900 + 1.25 * 200) * 2.0 + 10 * 8.0) / 1e6)


def test_default_weights_and_thinking_match_the_gateway():
    from desk.config import Settings as S
    d = S("https://gw.example", "key")
    assert (d.weight_out, d.weight_cache, d.weight_cache_write) == (1.0, 0.1, 1.25)
    assert d.extra_body == {"thinking": {"type": "disabled"}}     # deepseek-flash думает без спроса
    assert LLM(d, cache=False).payload([{"role": "user", "content": "x"}])["temperature"] == 0


# Кэш ответов

def test_cache_key_ignores_key_order():
    a = {"model": "deepseek-flash", "messages": [{"role": "user", "content": "ё"}], "temperature": 0}
    b = dict(reversed(list(a.items())))
    assert cache_key(a) == cache_key(b)
    assert cache_key(a) != cache_key({**a, "temperature": 0.7})


def test_cache_saves_second_call(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=OK)

    llm, _ = make(handler, tmp_path)
    llm.chat([{"role": "user", "content": "один и тот же вопрос"}])
    again = llm.chat([{"role": "user", "content": "один и тот же вопрос"}])
    assert len(calls) == 1 and again.usage.cached == 1
    assert again.usage.cost == 0 and again.usage.weighted == 0


def test_no_cache_when_temperature_above_zero(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=OK)

    llm, _ = make(handler, tmp_path)
    for _ in range(2):
        llm.chat([{"role": "user", "content": "вариант"}], temperature=0.8)
    assert len(calls) == 2


# Разбор ответа

def test_tool_use_blocks_are_parsed_and_replayed():
    data = {"content": [{"type": "text", "text": "Проверю статус."},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_payment_status",
                         "input": {"payment_id": "P-88121"}}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 5}}
    llm, _ = make(lambda r: httpx.Response(200, json=data))
    reply = llm.chat([{"role": "user", "content": "почему не прошёл платёж"}])
    call = reply.tool_calls[0]
    assert (call.name, call.input, reply.stop_reason) == ("get_payment_status", {"payment_id": "P-88121"}, "tool_use")
    assert reply.as_message() == {"role": "assistant", "content": data["content"]}
    block = tool_result(call, {"result": {"status": "declined"}})
    assert block["tool_use_id"] == "toolu_1" and json.loads(block["content"])["result"]["status"] == "declined"
    assert tool_result(call, {"error": "нет"}, is_error=True)["is_error"] is True


def test_thinking_blocks_do_not_leak_into_text():
    data = {"content": [{"type": "thinking", "thinking": "рассуждаю…", "signature": "s"},
                        {"type": "text", "text": "доступ"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 300}}
    llm, _ = make(lambda r: httpx.Response(200, json=data))
    reply = llm.chat([{"role": "user", "content": "Забыл пароль"}])
    assert reply.text == "доступ" and reply.usage.output_tokens == 300


# Разное

def test_retry_after_header():
    assert retry_after(httpx.Response(429, headers={"retry-after": "3"})) == 3.0
    assert retry_after(httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None
    assert retry_after(httpx.Response(429)) is None and retry_after(None) is None


def test_pause_or_raise_decides():
    llm, _ = make(lambda r: httpx.Response(200, json=OK), max_retries=2)
    assert 0 < llm.pause_or_raise(0, 503, None, httpx.Response(503)) <= 20
    assert llm.pause_or_raise(1, 429, None, httpx.Response(429, headers={"retry-after": "4"})) == 4
    with pytest.raises(LLMUnavailable):
        llm.pause_or_raise(2, 503, None, httpx.Response(503))        # попытки кончились
    with pytest.raises(LLMError) as e:
        llm.pause_or_raise(0, 400, None, httpx.Response(400, text="bad field max_tokens"))
    assert "400" in str(e.value) and "max_tokens" in str(e.value)
    with pytest.raises(LLMError):
        llm.pause_or_raise(0, None, ValueError("баг"), None)


def test_network_errors_are_retried():
    seen = []

    def handler(request):
        seen.append(1)
        if len(seen) == 1:
            raise httpx.ConnectError("обрыв")
        return httpx.Response(200, json=OK)

    llm, sleeps = make(handler)
    assert llm.chat([{"role": "user", "content": "x"}]).usage.retries == 1 and len(sleeps) == 1


def test_async_chat_retries_the_same_way():
    import asyncio
    seen = []

    async def handler(request):
        seen.append(1)
        return httpx.Response(503) if len(seen) < 2 else httpx.Response(200, json=OK)

    cfg = Settings("https://gw.example", "key", cache_dir=None)
    llm = LLM(cfg, async_transport=httpx.MockTransport(handler), cache=False)
    llm.sleep = lambda s: None
    import desk.llm as mod
    real = mod.backoff_delay
    mod.backoff_delay = lambda *a, **k: 0.0
    try:
        reply = asyncio.run(llm.achat([{"role": "user", "content": "x"}]))
    finally:
        mod.backoff_delay = real
    assert reply.text == "платежи" and len(seen) == 2 and llm.total().calls == 1


def test_truncated_and_stream_flag():
    data = dict(OK, stop_reason="max_tokens")
    llm, _ = make(lambda r: httpx.Response(200, json=data))
    assert llm.chat([{"role": "user", "content": "x"}]).truncated
    assert llm.payload([{"role": "user", "content": "x"}], stream=True)["stream"] is True
    assert "stream" not in llm.payload([{"role": "user", "content": "x"}])


def test_usage_from_separates_four_kinds_of_tokens():
    llm, _ = make(lambda r: httpx.Response(200, json=OK))
    u = llm.usage_from({"input_tokens": 10, "cache_creation_input_tokens": 90, "output_tokens": 5}, 0.5, 2)
    assert (u.input_tokens, u.cache_write_tokens, u.output_tokens) == (10, 90, 5)
    assert (u.retries, u.latency_s, u.calls) == (2, 0.5, 1)


def test_async_client_survives_several_event_loops():
    """Два asyncio.run подряд с одним клиентом: соединения первого цикла не переиспользуются"""
    import asyncio

    async def handler(request):
        return httpx.Response(200, json=OK)

    llm = LLM(Settings("https://gw.example", "key"), async_transport=httpx.MockTransport(handler), cache=False)
    first = asyncio.run(llm.achat([{"role": "user", "content": "раз"}]))
    client_one = llm._aclient
    second = asyncio.run(llm.achat([{"role": "user", "content": "два"}]))
    assert first.text == second.text == "платежи" and llm._aclient is not client_one


def test_rate_limit_waits_when_the_minute_is_full():
    clock = {"now": 0.0}
    limit = RateLimit(2, clock=lambda: clock["now"])
    assert limit.reserve() == 0 and limit.reserve() == 0
    assert limit.reserve() == pytest.approx(61)
    clock["now"] = 30.0
    assert limit.reserve() == pytest.approx(31)
    assert RateLimit(0).reserve() == 0


def test_client_paces_requests_by_rpm():
    cfg = Settings("https://gw.example", "key", "deepseek-flash", rpm=1)
    sleeps = []
    llm = LLM(cfg, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=OK)),
              sleep=sleeps.append, clock=lambda: 0.0)
    llm.chat([{"role": "user", "content": "раз"}])
    llm.chat([{"role": "user", "content": "два"}])
    assert sleeps == [pytest.approx(61)]


# -*- coding: utf-8 -*-
"""Настройки проекта. Всё, что зависит от окружения, читается здесь"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Читает .env. Переменные окружения важнее файла"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    base_url: str  # адрес шлюза без /v1
    api_key: str  # личный ключ
    model: str = "deepseek-flash"
    auth_scheme: str = "x-api-key"  # или "bearer"
    verify_ssl: bool = True
    timeout_s: float = 60.0
    rpm: int = 0
    # Цены за миллион токенов, условные
    price_in: float = 0.20
    price_out: float = 0.80
    # Веса взвешенных токенов, в них шлюз считает лимит
    weight_out: float = 1.0
    weight_cache: float = 0.1
    weight_cache_write: float = 1.25
    # Добавка к каждому запросу. По умолчанию выключает рассуждение
    extra_body: Dict[str, Any] = field(
        default_factory=lambda: {"thinking": {"type": "disabled"}}
    )
    cache_dir: Optional[Path] = None


def settings() -> Settings:
    """Настройки из .env и переменных окружения"""
    load_dotenv()
    env = os.environ
    cache = env.get("LLM_CACHE_DIR", str(ROOT / ".llm_cache"))
    base_url = env.get("LLM_BASE_URL", "").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return Settings(
        base_url=base_url,
        api_key=env.get("LLM_API_KEY", ""),
        model=env.get("LLM_MODEL", "deepseek-flash"),
        auth_scheme=env.get("LLM_AUTH_SCHEME", "x-api-key"),
        verify_ssl=env.get("LLM_VERIFY_SSL", "1") not in ("0", "false", "no"),
        timeout_s=float(env.get("LLM_TIMEOUT_S", "60")),
        rpm=int(env.get("LLM_RPM", "0")),
        price_in=float(env.get("LLM_PRICE_IN", "0.20")),
        price_out=float(env.get("LLM_PRICE_OUT", "0.80")),
        weight_out=float(env.get("LLM_WEIGHT_OUT", "1")),
        weight_cache=float(env.get("LLM_WEIGHT_CACHE", "0.1")),
        weight_cache_write=float(env.get("LLM_WEIGHT_CACHE_WRITE", "1.25")),
        extra_body=json.loads(
            env.get("LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
        ),
        cache_dir=Path(cache) if cache else None,
    )

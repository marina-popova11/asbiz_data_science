# -*- coding: utf-8 -*-
"""Схемы ответов

Из схемы строится и проверка ответа модели, и описание формата в постановке
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional, Type

from pydantic import BaseModel, Field, ValidationInfo, field_validator

Category = Literal["платежи", "возвраты", "доступ", "тарифы", "интеграция", "другое"]
PAYMENT_ID = re.compile(r"^P-\d{5}$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class Ticket(BaseModel):
    """Разобранное обращение"""

    reasoning: str = Field(
        description="одна-две фразы: что случилось и почему выбрана категория"
    )
    category: Category = Field(description="категория обращения из закрытого словаря")
    severity: int = Field(ge=1, le=5, description="срочность от 1 до 5 по правилам")
    needs_human: bool = Field(
        description="нужно ли передать обращение человеку по регламенту"
    )
    quote: str = Field(
        min_length=1,
        description="дословный фрагмент обращения, на котором основано решение",
    )
    payment_ids: List[str] = Field(
        default_factory=list,
        validate_default=True,
        description="все идентификаторы платежей вида P-12345",
    )
    amount: Optional[int] = Field(
        None, ge=0, description="сумма операции в рублях или null"
    )

    @field_validator("payment_ids")
    @classmethod
    def ids_look_right_and_come_from_text(
        cls, ids: List[str], info: ValidationInfo
    ) -> List[str]:
        """Номера подходят под шаблон, есть в тексте, и из текста взяты все"""
        # TODO С2: номера подходят под PAYMENT_ID, есть в тексте, и из текста взяты все
        raise NotImplementedError(
            "семинар 2: номера подходят под PAYMENT_ID, есть в тексте, и из текста взяты все"
        )

    @field_validator("quote")
    @classmethod
    def quote_is_verbatim(cls, quote: str, info: ValidationInfo) -> str:
        """Цитата дословно есть в обращении"""
        # TODO С2: цитата обязана быть подстрокой обращения (без учёта регистра и пробелов)
        raise NotImplementedError(
            "семинар 2: цитата обязана быть подстрокой обращения (без учёта регистра и пробелов)"
        )


def describe(schema: Type[BaseModel]) -> str:
    """Описание формата для постановки: поле, тип и пояснение"""
    lines = []
    for name, f in schema.model_fields.items():
        kind = str(f.annotation).replace("typing.", "")
        lines.append('  "%s": %s  // %s' % (name, kind, f.description or ""))
    return "{\n" + ",\n".join(lines) + "\n}"

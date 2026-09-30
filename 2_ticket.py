# -*- coding: utf-8 -*-
"""Разбор своего обращения или обращений из набора

    python 2_ticket.py "текст обращения"
    python 2_ticket.py T-017 T-025 T-085

Для обращений из набора рядом печатается эталон, а в конце число совпадений
"""

from __future__ import annotations

import asyncio
import re
import sys

from desk.data import tickets
from desk.llm import LLM
from desk.triage import Ticket, atriage_many


def human(flag: bool) -> str:
    return "человек" if flag else "без человека"


def main(args: list) -> None:
    if not args:
        sys.exit(__doc__)
    if all(re.fullmatch(r"T-\d{3}", a) for a in args):
        by_id = {r["id"]: r for r in tickets(None)}
        missing = [a for a in args if a not in by_id]
        if missing:
            sys.exit("нет таких обращений в наборе: %s" % ", ".join(missing))
        rows = [by_id[a] for a in args]
    else:
        rows = [{"id": "ваше", "text": " ".join(args), "gold": None}]
    answers = asyncio.run(atriage_many(LLM(), [r["text"] for r in rows]))
    same = 0
    for r, t in zip(rows, answers):
        if not isinstance(t, Ticket):
            print(r["id"], "не прошло проверку:", t)
            continue
        line = "%s  модель: %s, срочность %d, %s" % (
            r["id"],
            t.category,
            t.severity,
            human(t.needs_human),
        )
        g = r["gold"]
        if g:
            line += "  |  эталон: %s, срочность %d, %s" % (
                g["category"],
                g["severity"],
                human(g["needs_human"]),
            )
            same += (t.category, t.needs_human) == (g["category"], g["needs_human"])
        print(line)
        if not g:
            print("  обоснование:", t.reasoning)
    if rows[0]["gold"]:
        print(
            "совпало с эталоном по категории и передаче человеку: %d из %d"
            % (same, len(rows))
        )


if __name__ == "__main__":
    main(sys.argv[1:])

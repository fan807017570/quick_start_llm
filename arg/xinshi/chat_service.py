from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from arg.xinshi.application import answer_question, stream_answer_question
from arg.xinshi.config import MAX_HISTORY_MESSAGES
from arg.xinshi.logutil import preview_text
from arg.xinshi.schemas import ChatRequest, ChatResponse

log = logging.getLogger(__name__)

_APP_TIMEZONE = os.environ.get("APP_TIMEZONE", "Asia/Shanghai")
_MACHINE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~+=:/-]+$")
_HEX_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{8,}$")
_BASE64ISH_TOKEN_RE = re.compile(r"^[A-Za-z0-9+/]{12,}={0,2}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_DATE_QUERY_RE = re.compile(
    r"(今天|今日|现在|当前).*(几号|日期|星期几|周几|礼拜几)"
    r"|今天[是]?几月几号"
    r"|今天[是]?什么日期"
)
_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def is_non_natural_language_message(message: str) -> bool:
    text = message.strip()
    if not text:
        return True
    if re.search(r"[\u4e00-\u9fff]", text):
        return False
    if re.search(r"\s", text):
        return False
    if any(ch in "?!！？" for ch in text):
        return False

    if _UUID_RE.fullmatch(text) or text.isdigit() or _HEX_TOKEN_RE.fullmatch(text):
        return True
    if _BASE64ISH_TOKEN_RE.fullmatch(text):
        return True
    if _MACHINE_TOKEN_RE.fullmatch(text):
        has_digit = any(ch.isdigit() for ch in text)
        has_separator = any(ch in "._~+=:/-" for ch in text)
        return len(text) >= 16 or (has_digit and len(text) >= 6) or has_separator
    return not any(ch.isalpha() for ch in text)


def _direct_chat_response(message: str) -> ChatResponse:
    return ChatResponse(
        answer=message.strip(),
        sources=[],
        rewritten_query=None,
        standalone_query=None,
    )


def _now() -> datetime:
    try:
        return datetime.now(ZoneInfo(_APP_TIMEZONE))
    except ZoneInfoNotFoundError:
        log.warning("APP_TIMEZONE=%s is invalid, fallback to local timezone", _APP_TIMEZONE)
        return datetime.now()


def _date_answer(message: str) -> str | None:
    text = re.sub(r"\s+", "", message.strip())
    if not _DATE_QUERY_RE.search(text):
        return None

    now = _now()
    weekday = _WEEKDAYS[now.weekday()]
    return f"今天是{now.year}年{now.month}月{now.day}日，{weekday}。"


def _date_chat_response(message: str) -> ChatResponse | None:
    answer = _date_answer(message)
    if answer is None:
        return None
    log.info("chat short-circuited date question answer=%r", answer)
    return ChatResponse(
        answer=answer,
        sources=[],
        rewritten_query=None,
        standalone_query=None,
    )


def _history_for(req: ChatRequest) -> list[dict[str, str]]:
    return [m.model_dump() for m in req.history[-MAX_HISTORY_MESSAGES:]]


def handle_chat_request(req: ChatRequest) -> ChatResponse:
    date_resp = _date_chat_response(req.message)
    if date_resp is not None:
        return date_resp

    if is_non_natural_language_message(req.message):
        log.info(
            "chat short-circuited non-natural message_len=%d preview=%r",
            len(req.message),
            preview_text(req.message, 80),
        )
        return _direct_chat_response(req.message)

    hist = _history_for(req)
    log.info(
        "chat request use_rewrite=%s history=%d message_len=%d preview=%r",
        req.use_rewrite,
        len(hist),
        len(req.message),
        preview_text(req.message, 80),
    )
    out = answer_question(
        req.message.strip(),
        history=hist,
        use_rewrite=req.use_rewrite,
    )
    return ChatResponse(
        answer=out["answer"],
        sources=out["sources"],
        rewritten_query=out.get("rewritten_query"),
        standalone_query=out.get("standalone_query"),
    )


def iter_stream_chat_events(req: ChatRequest) -> Iterator[dict]:
    date_resp = _date_chat_response(req.message)
    if date_resp is not None:
        yield {
            "type": "meta",
            "sources": [],
            "rewritten_query": None,
            "standalone_query": None,
        }
        yield {"type": "delta", "content": date_resp.answer}
        yield {"type": "done", "answer": date_resp.answer, "sources": []}
        return

    hist = _history_for(req)
    log.info(
        "chat_stream request use_rewrite=%s history=%d message_len=%d preview=%r",
        req.use_rewrite,
        len(hist),
        len(req.message),
        preview_text(req.message, 80),
    )
    try:
        yield from stream_answer_question(
            req.message.strip(),
            history=hist,
            use_rewrite=req.use_rewrite,
        )
    except Exception as exc:
        log.exception("chat_stream endpoint error: %s", exc)
        yield {"type": "error", "detail": str(exc)}

"""
新实中学招生顾问 RAG — HTTP 服务。

启动方式（任选其一，均需已配置 Milvus / 模型）：

    cd /path/to/quick_start_llm
    python -m uvicorn arg.xinshi.server:app --host 0.0.0.0 --port 8765

    # 或在 PyCharm 里直接运行本文件（会内嵌启动 uvicorn）：
    python arg/xinshi/server.py

浏览器打开：http://127.0.0.1:8765/
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import time
from xml.etree import ElementTree
from pathlib import Path
from typing import Literal

# 保证可解析包 arg.xinshi
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 本地开发时自动加载 arg/xinshi/.env（Docker 环境中 .env 不存在，此行无副作用）
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from arg.xinshi.config import MAX_HISTORY_MESSAGES
from arg.xinshi.logutil import configure_logging, preview_text

configure_logging()
log = logging.getLogger(__name__)

from arg.xinshi.application import answer_question, stream_answer_question

STATIC_DIR = Path(__file__).resolve().parent / "static"
DOCS_DIR = Path(__file__).resolve().parent / "data" / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 允许上传的文件后缀
_ALLOWED_SUFFIXES = {".md", ".txt"}
# 安全文件名：只允许字母、数字、连字符、下划线、点
_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")
_MACHINE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~+=:/-]+$")
_HEX_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{8,}$")
_BASE64ISH_TOKEN_RE = re.compile(r"^[A-Za-z0-9+/]{12,}={0,2}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_WECHAT_TOKEN_ENV = "WECHAT_TOKEN"

app = FastAPI(title="新实中学招生顾问", version="1.0")


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        t0 = time.perf_counter()
        log.info("HTTP %s %s", request.method, request.url.path)
        try:
            response = await call_next(request)
            log.info(
                "HTTP %s %s -> %s in %.3fs",
                request.method,
                request.url.path,
                response.status_code,
                time.perf_counter() - t0,
            )
            return response
        except Exception:
            log.exception(
                "HTTP %s %s failed after %.3fs",
                request.method,
                request.url.path,
                time.perf_counter() - t0,
            )
            raise
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=32000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="当前用户问题")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="本轮之前的对话，按时间顺序；不含当前 message",
    )
    use_rewrite: bool = Field(True, description="是否做指代消解后再做检索同义词扩展")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    rewritten_query: str | None = None
    standalone_query: str | None = Field(
        None,
        description="多轮时指代消解后的检索句（再经同义词扩展后用于 Milvus）",
    )


def _verify_wechat_signature(signature: str, timestamp: str, nonce: str) -> bool:
    token = os.environ.get(_WECHAT_TOKEN_ENV, "").strip()
    if not token:
        log.warning("%s is not configured; reject WeChat validation request", _WECHAT_TOKEN_ENV)
        return False

    raw = "".join(sorted([token, timestamp, nonce]))
    expected = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return hmac.compare_digest(expected, signature)


def _is_non_natural_language_message(message: str) -> bool:
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
    text = message.strip()
    return ChatResponse(
        answer=text,
        sources=[],
        rewritten_query=None,
        standalone_query=None,
    )


def _parse_wechat_xml_message(xml_body: bytes) -> dict[str, str]:
    if len(xml_body) > 1024 * 1024:
        raise HTTPException(status_code=413, detail="XML 报文过大")

    try:
        root = ElementTree.fromstring(xml_body)
    except ElementTree.ParseError as exc:
        raise HTTPException(status_code=400, detail="XML 报文格式错误") from exc

    def text_of(tag: str) -> str:
        return (root.findtext(tag) or "").strip()

    return {
        "to_user": text_of("ToUserName"),
        "from_user": text_of("FromUserName"),
        "create_time": text_of("CreateTime"),
        "msg_type": text_of("MsgType"),
        "content": text_of("Content"),
        "msg_id": text_of("MsgId"),
        "msg_data_id": text_of("MsgDataId"),
        "idx": text_of("Idx"),
    }


def _cdata(text: str) -> str:
    return f"<![CDATA[{text.replace(']]>', ']]]]><![CDATA[>')}]]>"


def _wechat_text_reply(to_user: str, from_user: str, content: str) -> str:
    return (
        "<xml>"
        f"<ToUserName>{_cdata(to_user)}</ToUserName>"
        f"<FromUserName>{_cdata(from_user)}</FromUserName>"
        f"<CreateTime>{int(time.time())}</CreateTime>"
        f"<MsgType>{_cdata('text')}</MsgType>"
        f"<Content>{_cdata(content)}</Content>"
        "</xml>"
    )


def _handle_chat_request(req: ChatRequest) -> ChatResponse:
    if _is_non_natural_language_message(req.message):
        log.info(
            "chat short-circuited non-natural message_len=%d preview=%r",
            len(req.message),
            preview_text(req.message, 80),
        )
        return _direct_chat_response(req.message)

    hist = [m.model_dump() for m in req.history[-MAX_HISTORY_MESSAGES:]]
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


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat_wechat_message(
    request: Request,
    signature: str | None = Query(None, description="微信加密签名"),
    timestamp: str | None = Query(None, description="时间戳"),
    nonce: str | None = Query(None, description="随机数"),
):
    signature_parts = (signature, timestamp, nonce)
    if any(signature_parts):
        is_valid_signature = all(signature_parts) and _verify_wechat_signature(
            signature or "",
            timestamp or "",
            nonce or "",
        )
        if not is_valid_signature:
            log.warning("invalid WeChat POST signature timestamp=%s nonce=%s", timestamp, nonce)
            return PlainTextResponse("", status_code=403)

    body = await request.body()
    msg = _parse_wechat_xml_message(body)
    if msg["msg_type"] and msg["msg_type"] != "text":
        log.info("ignore unsupported WeChat msg_type=%s msg_id=%s", msg["msg_type"], msg["msg_id"])
        return PlainTextResponse("")
    if not msg["content"]:
        raise HTTPException(status_code=400, detail="XML 报文缺少 Content")

    log.info(
        "wechat message received from=%s to=%s msg_id=%s content_len=%d preview=%r",
        msg["from_user"],
        msg["to_user"],
        msg["msg_id"],
        len(msg["content"]),
        preview_text(msg["content"], 80),
    )
    try:
        chat_resp = _handle_chat_request(ChatRequest(message=msg["content"]))
    except Exception as e:
        log.exception("wechat message chat dispatch error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    reply_xml = _wechat_text_reply(
        to_user=msg["from_user"],
        from_user=msg["to_user"],
        content=chat_resp.answer,
    )
    return PlainTextResponse(reply_xml, media_type="application/xml; charset=utf-8")


@app.get("/api/chat", response_model=ChatResponse)
async def chat_wechat_validation(
    signature: str = Query(..., description="微信加密签名"),
    timestamp: str = Query(..., description="时间戳"),
    nonce: str = Query(..., description="随机数"),
    echostr: str = Query(..., description="随机字符串"),
):
    if not _verify_wechat_signature(signature, timestamp, nonce):
        log.warning(
            "invalid WeChat signature timestamp=%s nonce=%s echostr_len=%d",
            timestamp,
            nonce,
            len(echostr),
        )
        return PlainTextResponse("", status_code=403)

    log.info("valid WeChat signature; dispatch echostr to chat handler len=%d", len(echostr))
    if _is_non_natural_language_message(echostr):
        log.info("return non-natural WeChat echostr directly")
        return PlainTextResponse(echostr)

    try:
        return _handle_chat_request(ChatRequest(message=echostr))
    except Exception as e:
        log.exception("wechat validation chat dispatch error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    hist = [m.model_dump() for m in req.history[-MAX_HISTORY_MESSAGES:]]
    log.info(
        "chat_stream request use_rewrite=%s history=%d message_len=%d preview=%r",
        req.use_rewrite,
        len(hist),
        len(req.message),
        preview_text(req.message, 80),
    )

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _iter_events():
        try:
            for item in stream_answer_question(
                req.message.strip(),
                history=hist,
                use_rewrite=req.use_rewrite,
            ):
                yield _sse(item)
        except Exception as exc:
            log.exception("chat_stream endpoint error: %s", exc)
            yield _sse({"type": "error", "detail": str(exc)})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _iter_events(),
        media_type="text/event-stream; charset=utf-8",
        headers=headers,
    )


@app.get("/api/docs", summary="列出 data/docs 目录下的所有文件")
async def list_docs():
    files = sorted(
        [
            {"name": f.name, "size": f.stat().st_size, "suffix": f.suffix}
            for f in DOCS_DIR.iterdir()
            if f.is_file()
        ],
        key=lambda x: x["name"],
    )
    return {"files": files, "dir": str(DOCS_DIR)}


@app.post("/api/docs/upload", summary="上传文档到 data/docs 目录")
async def upload_doc(
    file: UploadFile = File(...),
    reindex: bool = False,
):
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix}，仅允许：{', '.join(_ALLOWED_SUFFIXES)}",
        )
    if not _SAFE_FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="文件名包含非法字符")

    dest = DOCS_DIR / filename
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB 上限
        raise HTTPException(status_code=400, detail="文件大小不能超过 10 MB")

    dest.write_bytes(content)
    log.info("uploaded doc: %s (%d bytes) reindex=%s", dest, len(content), reindex)

    result: dict = {"saved": filename, "bytes": len(content), "reindex": reindex}

    if reindex:
        try:
            from arg.xinshi.ingest import ingest as _ingest
            from arg.xinshi.application import reload_vectorstore

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _ingest)
            await loop.run_in_executor(None, reload_vectorstore)
            result["reindex_status"] = "ok"
            log.info("reindex + vectorstore reload triggered by upload of %s", filename)
        except Exception as exc:
            log.exception("reindex failed after upload of %s: %s", filename, exc)
            result["reindex_status"] = f"failed: {exc}"

    return JSONResponse(content=result)


# 全局重建任务状态，防止并发重复触发
_reindex_lock = asyncio.Lock()
_reindex_status: dict = {"running": False, "last_result": None}


@app.post("/api/docs/reindex", summary="触发全量重建向量索引")
async def trigger_reindex():
    if _reindex_status["running"]:
        raise HTTPException(status_code=409, detail="索引重建任务正在运行，请稍后再试")

    async def _run():
        _reindex_status["running"] = True
        t0 = time.perf_counter()
        try:
            from arg.xinshi.ingest import ingest as _ingest
            from arg.xinshi.application import reload_vectorstore

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _ingest)
            elapsed = round(time.perf_counter() - t0, 2)
            log.info("reindex finished in %.2fs, reloading vectorstore...", elapsed)

            # 关键：刷新内存中的 vectorstore，否则仍查旧数据
            await loop.run_in_executor(None, reload_vectorstore)

            _reindex_status["last_result"] = {"status": "ok", "elapsed_s": elapsed}
            log.info("vectorstore reloaded, reindex fully complete")
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            _reindex_status["last_result"] = {"status": "error", "detail": str(exc), "elapsed_s": elapsed}
            log.exception("reindex failed after %.2fs: %s", elapsed, exc)
        finally:
            _reindex_status["running"] = False

    asyncio.create_task(_run())
    return JSONResponse(content={"accepted": True, "message": "索引重建任务已启动，请稍后查询状态"})


@app.get("/api/docs/reindex/status", summary="查询索引重建任务状态")
async def reindex_status():
    return JSONResponse(content={
        "running": _reindex_status["running"],
        "last_result": _reindex_status["last_result"],
    })


@app.get("/")
async def index_page():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="static/index.html missing")
    return FileResponse(index, media_type="text/html; charset=utf-8")


if __name__ == "__main__":
    import os

    import uvicorn

    host = os.environ.get("XINSHI_HOST", "0.0.0.0")
    port = int(os.environ.get("XINSHI_PORT", "8765"))
    log.info("Starting uvicorn on http://%s:%s (XINSHI_LOG_LEVEL=%s)", host, port, os.environ.get("XINSHI_LOG_LEVEL", "INFO"))
    uvicorn.run(app, host=host, port=port)

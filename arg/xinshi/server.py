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
import logging
import re
import sys
import time
from pathlib import Path
from typing import Literal

# 保证可解析包 arg.xinshi
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 本地开发时自动加载 arg/xinshi/.env（Docker 环境中 .env 不存在，此行无副作用）
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from arg.xinshi.config import MAX_HISTORY_MESSAGES
from arg.xinshi.logutil import configure_logging, preview_text

configure_logging()
log = logging.getLogger(__name__)

from arg.xinshi.application import answer_question

STATIC_DIR = Path(__file__).resolve().parent / "static"
DOCS_DIR = Path(__file__).resolve().parent / "data" / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 允许上传的文件后缀
_ALLOWED_SUFFIXES = {".md", ".txt"}
# 安全文件名：只允许字母、数字、连字符、下划线、点
_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")

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


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
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
    except Exception as e:
        log.exception("chat endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


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

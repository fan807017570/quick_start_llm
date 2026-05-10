from __future__ import annotations

import logging
import time

from arg.xinshi.logutil import configure_logging, preview_text

configure_logging()

from arg.xinshi.config import MAX_HISTORY_MESSAGES, TOP_K_RETRIEVE, TOP_K_RERANK
from arg.xinshi.llm import get_llm
from arg.xinshi.reranker import BGEReranker
from arg.xinshi.retriever import get_vectorstore

log = logging.getLogger(__name__)

log.info("Loading vectorstore and reranker (cold start may take a while)...")
_t_load = time.perf_counter()
vectorstore = get_vectorstore()
reranker = BGEReranker()
log.info(
    "RAG backends ready in %.2fs (TOP_K_RETRIEVE=%d TOP_K_RERANK=%d)",
    time.perf_counter() - _t_load,
    TOP_K_RETRIEVE,
    TOP_K_RERANK,
)


def reload_vectorstore() -> None:
    """ingest 重建索引后调用，刷新内存中的 vectorstore 连接。"""
    global vectorstore
    log.info("reloading vectorstore after reindex...")
    t0 = time.perf_counter()
    vectorstore = get_vectorstore()
    log.info("vectorstore reloaded in %.2fs", time.perf_counter() - t0)


def retrieve(query: str):
    t0 = time.perf_counter()
    docs = vectorstore.similarity_search(query, k=TOP_K_RETRIEVE)
    t1 = time.perf_counter()
    docs = reranker.rerank(query, docs, TOP_K_RERANK)
    t2 = time.perf_counter()
    log.info(
        "retrieve: similarity_search=%.3fs rerank=%.3fs final_docs=%d",
        t1 - t0,
        t2 - t1,
        len(docs),
    )
    log.debug(
        "retrieve sections: %s",
        [d.metadata.get("section") for d in docs],
    )
    return docs


def _format_history_block(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for m in history[-MAX_HISTORY_MESSAGES:]:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "家长" if role == "user" else "顾问"
        lines.append(f"{label}：{content}")
    if not lines:
        return ""
    return "近期对话（供理解指代与语气）：\n" + "\n".join(lines) + "\n\n"


def build_prompt(
    user_message: str,
    docs,
    *,
    history: list[dict[str, str]] | None = None,
):
    context = "\n\n".join([
        f"[{d.metadata.get('section')}] {d.page_content}"
        for d in docs
    ])
    hist = _format_history_block(history)
    return f"""
    你是新实中学的资深招生顾问，像面对面跟家长、学生聊天一样回答：语气亲切、自然、有温度，可以适当用「咱们学校」「您」等称呼，但不要夸张承诺。
    回答必须基于下方「内部参考」中的事实，但**不要**在话里暴露你在查资料：禁止使用「根据资料显示」「据资料」「由上述内容可知」「综上所述」「从文档中可以看到」以及类似套话；直接像真人知道这些事一样说出来即可。
    若内部参考里能回答（含地址、面积、校长职务、师资、设施等），要说清楚、说完整；若确实没有相关内容，再坦诚说目前不了解或建议通过官方渠道核实，不要用生硬模板。
    若用户是在追问上一话题，请承接对话自然衔接，不要重复寒暄。
    内部参考：
    {context}
    {hist}当前用户这句话（请结合上文理解「那里、这个、还有吗」等指代）：
    {user_message}
    """


def _message_text(msg) -> str:
    """Chat models return AIMessage; embeddings and prompts need plain str."""
    if isinstance(msg, str):
        return msg
    return getattr(msg, "content", str(msg))


def contextualize_for_search(
    user_message: str,
    history: list[dict[str, str]] | None,
) -> str:
    """有多轮对话时，把追问改写成可单独检索的完整问句（消解指代）。"""
    text = user_message.strip()
    if not history:
        return text
    lines: list[str] = []
    for m in history[-MAX_HISTORY_MESSAGES:]:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "家长" if role == "user" else "顾问"
        lines.append(f"{label}：{content}")
    conv = "\n".join(lines)
    llm = get_llm()
    prompt = f"""下面是招生咨询对话节选。请把「用户最后一问」改写成一条**独立、完整**的中文句子，用于知识库向量检索。
要求：
- 把「那里、这边、这个、呢、还有吗、同上」等指代补全成具体事物（结合上文）；
- 保留用户真实意图，不要替用户改话题；
- 不要回答内容本身，不要解释。

对话节选：
{conv}

用户最后一问：{text}

只输出改写后的这一句："""
    out = _message_text(llm.invoke(prompt)).strip()
    return out or text


def query_rewrite(query: str) -> str:
    llm = get_llm()
    prompt = f"""
    将用户问题改写成更适合向量检索的查询（可补充同义词，但不要改变主题）。
    规则：
    - 问地址/在哪/校址/位置 → 保留并补充：学校地址、地理位置、位于、校址、广丰 等词。
    - 问面积/多大/占地 → 保留并补充：占地面积、建筑面积、平方米、亩。
    - 问校长/领导/负责人/谁主持 → 保留并补充：校长姓名、学校领导、领导班子、主持人、负责人 等词。
    - 问老师/教师/师资 → 保留并补充：教师团队、师资力量、专兼职教师、教职工 等词。
    - 问费用/学费/收费 → 保留并补充：收费标准、学费、住宿费、费用 等词。
    - 问招生/报名/入学 → 保留并补充：招生简章、报名条件、入学要求、招生政策 等词。
    - 不要改成与原文无关的主题（例如把校长问句改成招生政策）。
    只输出改写后的一句话，不要解释。
    用户问题：{query}
    """
    return _message_text(llm.invoke(prompt))


def answer_question(
    user_message: str,
    *,
    history: list[dict[str, str]] | None = None,
    use_rewrite: bool = True,
) -> dict:
    """供 CLI / HTTP 调用：可选多轮 history（不含当前句），检索 + 生成回答。"""
    t_all = time.perf_counter()
    hist = history or []
    hist = hist[-MAX_HISTORY_MESSAGES:]
    log.info(
        "answer_question start use_rewrite=%s history_len=%d user_len=%d preview=%r",
        use_rewrite,
        len(hist),
        len(user_message),
        preview_text(user_message, 100),
    )
    try:
        t_ctx = time.perf_counter()
        standalone = contextualize_for_search(user_message, hist if hist else None)
        log.info(
            "contextualize_for_search in %.3fs preview=%r",
            time.perf_counter() - t_ctx,
            preview_text(standalone, 100),
        )

        if use_rewrite:
            t_rw = time.perf_counter()
            q = query_rewrite(standalone)
            log.info(
                "query_rewrite done in %.3fs preview=%r",
                time.perf_counter() - t_rw,
                preview_text(q, 100),
            )
        else:
            q = standalone

        docs = retrieve(q)

        t_llm = time.perf_counter()
        prompt = build_prompt(user_message.strip(), docs, history=hist)
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = _message_text(response)
        log.info(
            "llm invoke done in %.3fs answer_len=%d",
            time.perf_counter() - t_llm,
            len(answer),
        )

        sources = [d.metadata.get("section") or "" for d in docs]
        log.info(
            "answer_question ok total=%.3fs sources=%s",
            time.perf_counter() - t_all,
            sources,
        )
        return {
            "answer": answer,
            "rewritten_query": q if use_rewrite else None,
            "standalone_query": standalone,
            "sources": sources,
        }
    except Exception:
        log.exception(
            "answer_question failed after %.3fs use_rewrite=%s",
            time.perf_counter() - t_all,
            use_rewrite,
        )
        raise


def main():
    history: list[dict[str, str]] = []
    while True:
        query = input("\n? 问题: ")
        out = answer_question(query, history=history, use_rewrite=True)
        print("\n回答：\n", out["answer"])
        for s in out["sources"]:
            print("-", s)
        history.append({"role": "user", "content": query.strip()})
        history.append({"role": "assistant", "content": out["answer"]})
        history = history[-MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    main()

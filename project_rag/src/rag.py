# -*- coding: utf-8 -*-
"""检索 + 生成模块 (RAG 的查询侧)。

流程::

    问题 -> [bge 文本向量] ---\
                              -> RRF 融合 -> top-k chunks -> 组装 Prompt
        -> [BM25 关键词召回] --/                              |
        -> [CLIP 文本向量 -> 图像索引] (可选)                  v
                                                    调用 LLM -> 带引用的回答

两个设计要点:

1. **Retriever 复用模型**: 向量模型加载一次要几十秒, 评测时要跑很多问题,
   所以把 embedder 缓存在 Retriever 对象里, 而不是每次查询都重新加载。

2. **引用溯源**: 拼 Prompt 时给每个片段编号并写上"出自哪个文件第几页",
   同时要求模型在回答里用 [编号] 标注来源。终端还会把命中的原文片段打出来,
   让用户能自己核对模型有没有胡说。
"""

from __future__ import annotations
# --- Windows 控制台兼容 ---
# 把输出重定向到文件/管道时，Python 在中文 Windows 上默认用 GBK 编码，
# 一旦 print 出 GBK 编不了的字符（emoji 等）就会抛 UnicodeEncodeError 中断脚本。
# 下面几行保证任何字符都不会把程序搞挂。
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
# ---------------------------


import re
from typing import List, Optional, Sequence

from . import config
from .embedding import ImageEmbedder, TextEmbedder
from .index import RetrievalHit, RagIndex

# --------------------------------------------------------------------------
# Prompt 模板
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一个严谨的文档问答助手。你只能依据用户提供的"参考资料"回答问题。

必须遵守的规则:
1. 每一条关键信息后面都要用方括号标出来源编号, 例如: 系统的输入维度是 784 [1]。
2. 如果参考资料不足以回答问题, 就直接说明"根据现有资料无法确定", 绝对不要编造。
3. 不要使用参考资料之外的任何知识。
4. 用简体中文回答, 语言简洁, 条理清晰, 适当分点。
5. 回答的最后另起一行, 用"参考来源:"列出你用到的编号及其文件页码。"""

USER_PROMPT_TEMPLATE = """以下是检索到的参考资料:

{context}

问题: {question}

请根据上面的参考资料回答。再次提醒: 必须用 [编号] 标注出处, 资料不足时如实说明。"""


# --------------------------------------------------------------------------
# 上下文组装
# --------------------------------------------------------------------------


def format_context(hits: Sequence[RetrievalHit], max_chars_per_chunk: int = 1500) -> str:
    """把检索结果拼成带编号的上下文块。"""
    blocks: List[str] = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        text = chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + " ...(已截断)"
        source = "[{}] 来源: {} 第 {} 页".format(number, chunk.doc, chunk.page)
        if chunk.kind == "image":
            source += " (图片)"
            if chunk.image_path:
                source += " 文件: {}".format(chunk.image_path)
        blocks.append("{}\n内容:\n{}".format(source, text))
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: Sequence[RetrievalHit]) -> str:
    return USER_PROMPT_TEMPLATE.format(context=format_context(hits), question=question.strip())


def format_hits_for_terminal(hits: Sequence[RetrievalHit], max_chars: int = 260) -> str:
    """把命中结果排版成终端可读的多行文本 (不交给 rich 渲染, 避免方括号被吞)。"""
    if not hits:
        return "  (没有检索到任何内容)"
    lines: List[str] = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        lines.append(
            "[{}] {} 第 {} 页  融合分={:.4f}  来源={}".format(
                number, chunk.doc, chunk.page, hit.score, hit.source_desc()
            )
        )
        lines.append("    chunk_id={}".format(chunk.chunk_id))
        if chunk.kind == "image" and chunk.image_path:
            lines.append("    图片路径: {}".format(chunk.image_path))
        body = re.sub(r"\s+", " ", chunk.text).strip()
        if len(body) > max_chars:
            body = body[:max_chars] + " ..."
        lines.append("    原文: {}".format(body))
        lines.append("")
    return "\n".join(lines).rstrip()


# --------------------------------------------------------------------------
# 检索器
# --------------------------------------------------------------------------


class Retriever:
    """封装"问题 -> 检索结果"的过程, 并缓存向量模型。

    参数:
        index:      已加载的 RagIndex
        use_bm25:   是否启用 BM25 关键词召回 (消融实验时关掉它)
        use_images: 是否启用 CLIP 图像召回; None 表示跟随索引自身的设置
        rrf_k:      RRF 平滑常数, 默认 60
    """

    def __init__(
        self,
        index: RagIndex,
        use_bm25: bool = True,
        use_images: Optional[bool] = None,
        rrf_k: int = config.DEFAULT_RRF_K,
        candidate_n: int = config.DEFAULT_CANDIDATE_N,
    ) -> None:
        self.index = index
        self.use_bm25 = bool(use_bm25)
        self.rrf_k = int(rrf_k)
        self.candidate_n = int(candidate_n)
        if use_images is None:
            use_images = index.with_images
        self.use_images = bool(use_images and index.image_index is not None)
        self._text_embedder: Optional[TextEmbedder] = None
        self._image_embedder: Optional[ImageEmbedder] = None

    # -- 模型懒加载 -----------------------------------------------------
    @property
    def text_embedder(self) -> TextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder(
                model_name=str(self.index.meta.get("text_model", config.DEFAULT_TEXT_MODEL))
            )
        return self._text_embedder

    @property
    def image_embedder(self) -> ImageEmbedder:
        if self._image_embedder is None:
            self._image_embedder = ImageEmbedder(
                model_name=str(self.index.meta.get("clip_model", config.DEFAULT_CLIP_MODEL))
            )
        return self._image_embedder

    # -- 检索 -----------------------------------------------------------
    def retrieve(self, question: str, top_k: int = config.DEFAULT_TOP_K) -> List[RetrievalHit]:
        """执行混合检索。向量模型缺失时自动退化为 BM25, 不会直接崩掉。"""
        question = (question or "").strip()
        if not question:
            raise config.ConfigError("问题不能为空。")

        query_vector = None
        if self.index.text_index is not None:
            query_vector = self.text_embedder.encode_query(question)
        elif not self.use_bm25:
            config.warn("该索引既没有向量索引也没有开启 BM25, 无法检索。")

        clip_query_vector = None
        if self.use_images:
            try:
                clip_query_vector = self.image_embedder.encode_query(question)
            except config.ConfigError as exc:
                config.warn("CLIP 查询编码失败, 本次只做文本检索: {}".format(exc))
                clip_query_vector = None

        return self.index.hybrid_search(
            query=question,
            query_vector=query_vector,
            clip_query_vector=clip_query_vector,
            top_k=top_k,
            use_bm25=self.use_bm25,
            rrf_k=self.rrf_k,
            candidate_n=self.candidate_n,
        )

    def describe(self) -> str:
        parts = []
        parts.append("向量检索" if self.index.text_index is not None else "向量检索(不可用)")
        parts.append("BM25 关键词检索" if (self.use_bm25 and self.index.bm25) else "BM25(关闭)")
        parts.append("CLIP 图像检索" if self.use_images else "图像检索(关闭)")
        return " + ".join(parts)


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------


def call_llm(
    messages: Sequence[dict],
    llm: Optional[config.LLMConfig] = None,
    temperature: Optional[float] = None,
) -> str:
    """调用 OpenAI 兼容接口。缺少 key / 网络错误 / 接口报错都给中文提示。"""
    if llm is None:
        llm = config.load_llm_config()
    llm.require()
    client = config.build_openai_client(llm)
    temp = llm.temperature if temperature is None else float(temperature)
    try:
        response = client.chat.completions.create(
            model=llm.model,
            messages=list(messages),
            temperature=temp,
        )
    except Exception as exc:
        raise config.ConfigError(_explain_llm_error(exc, llm)) from exc
    try:
        return (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError) as exc:
        raise config.ConfigError(
            "大模型返回的内容格式异常, 无法解析出回答文本。\n原始返回: {}".format(response)
        ) from exc


def _explain_llm_error(exc: Exception, llm: config.LLMConfig) -> str:
    """把底层 SDK 的报错翻译成用户能直接动手处理的中文提示。"""
    text = str(exc)
    lowered = text.lower()
    hints = [
        "调用大模型接口失败。",
        "  当前配置: {}".format(llm.describe()),
    ]
    if "401" in text or "invalid_api_key" in lowered or "authentication" in lowered:
        hints.append("  判断: API Key 无效或已过期。请打开 .env 检查 LLM_API_KEY 是否复制完整。")
    elif "404" in text or "not found" in lowered:
        hints.append(
            "  判断: 接口地址或模型名不对。请确认 LLM_BASE_URL 带了正确的后缀 "
            "(DeepSeek 是 https://api.deepseek.com/v1), 且 LLM_MODEL 是该服务商的模型名。"
        )
    elif "429" in text or "rate limit" in lowered or "quota" in lowered:
        hints.append("  判断: 触发限流或余额不足。稍后重试, 或检查账户余额。")
    elif "timeout" in lowered or "timed out" in lowered:
        hints.append("  判断: 请求超时。可以调大 .env 里的 LLM_TIMEOUT, 或检查网络/代理。")
    elif "connection" in lowered or "ssl" in lowered or "proxy" in lowered:
        hints.append("  判断: 网络无法连通。请检查网络, 或确认是否需要配置代理。")
    elif "insufficient" in lowered or "balance" in lowered:
        hints.append("  判断: 账户余额不足。")
    else:
        hints.append("  判断: 未识别的错误, 请对照上面的原始报错排查。")
    hints.append("  原始报错: {}".format(text))
    return "\n".join(hints)


def answer_question(
    index: RagIndex,
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    use_bm25: bool = True,
    use_images: Optional[bool] = None,
    rrf_k: int = config.DEFAULT_RRF_K,
    retriever: Optional[Retriever] = None,
    llm: Optional[config.LLMConfig] = None,
    temperature: Optional[float] = None,
) -> dict:
    """完整问答流程: 检索 -> 拼 Prompt -> 调 LLM -> 返回带引用的回答。

    返回字典::

        {"question":..., "answer":..., "hits":[...], "prompt":..., "context":...}
    """
    if retriever is None:
        retriever = Retriever(index, use_bm25=use_bm25, use_images=use_images, rrf_k=rrf_k)
    hits = retriever.retrieve(question, top_k=top_k)
    if not hits:
        return {
            "question": question,
            "answer": "",
            "hits": [],
            "prompt": "",
            "context": "（没有检索到任何内容）",
            "error": "没有检索到任何相关内容",
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, hits)},
    ]
    answer = call_llm(messages, llm=llm, temperature=temperature)
    return {
        "question": question,
        "answer": answer,
        "hits": hits,
        "prompt": messages[-1]["content"],
        "context": format_context(hits),
        "error": "",
    }


__all__ = [
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "Retriever",
    "answer_question",
    "build_user_prompt",
    "call_llm",
    "format_context",
    "format_hits_for_terminal",
]

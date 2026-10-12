# -*- coding: utf-8 -*-
"""评测模块。

回答两个问题:

1. **检索准不准** —— ``recall@k``: 正确答案所在的页/文档, 有没有出现在检索到的
   前 k 个 chunk 里。这是 RAG 的上限, 检索不到就一定答不对。
   顺带给出 ``MRR@k`` (第一个正确结果的排名倒数均值), 能区分"排第 1"和"排第 5"。

2. **答得对不对** —— 答案准确率。因为没有标准评测器, 本项目实现两种方式,
   用 ``--judge`` 切换:
     - ``keyword``: 关键词命中率。从参考答案里抽关键词, 看模型回答覆盖了多少。
       便宜、可复现、不花钱, 适合大规模跑消融实验。
     - ``llm``: 让大模型当裁判, 给 0 / 0.5 / 1 分。更接近人的判断, 但要花 token。
     - ``both``: 两种都算 (没有 API key 时会自动降级为只算关键词)。

关键词抽取复用 ``src/index.py`` 的 ``tokenize`` (单字 + 字符二元组),
所以本项目评测环节同样不需要 jieba 等分词库。

评测集格式 (JSONL, 每行一个 JSON 对象)::

    {"question": "...", "answer": "...", "source_page": 3, "source_doc": "a.pdf"}

    question    必填, 问题
    answer      必填, 参考答案 (用于算答案准确率)
    source_page 选填, 正确内容所在页码 (从 1 开始), 用于算 recall@k
    source_doc  选填, 正确内容所在文件名, 多文档索引时建议填
    source_page 与 source_doc 至少要有一个, 否则无法判断检索是否命中
    (以下划线开头的字段会被忽略, 可以写注释)
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


import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .index import RetrievalHit, RagIndex, tokenize
from .rag import Retriever, call_llm

# --------------------------------------------------------------------------
# 关键词抽取 (零依赖, 复用 tokenize 的单字 + bigram 切分)
# --------------------------------------------------------------------------

#: 极简中文停用词表, 只用于关键词命中率, 不影响 BM25
STOPWORDS = set(
    """的 了 是 在 和 与 及 或 也 都 而 但 但是 因为 所以 如果 就 这 那 有 我 你 他
她 它 我们 你们 他们 一个 一些 这个 那个 什么 怎么 如何 可以 能够 需要 应该 对于
关于 通过 由于 以及 并且 其中 进行 使用 采用 得到 结果 方法 问题 答案 请 以 为 于
上下 中内 外前 后时 时候 目前 主要 非常 比较 更 最 很 没有 会 要 把 被 让 使 从到
向 对 按 据 根据 例如 比如 如下 上述 该 其 此 等 等等 一个 一种 第一 第二 第三
其中 同时 因此 然后 另外 此外 总之 综上 我们 他们 它们 这个 这些 那些""".split()
)


def extract_keywords(text: str) -> List[str]:
    """从一段文本里抽关键词 (去重、保序)。

    规则: 长度 >= 2 (即中文 bigram 或英文单词), 去掉停用词与纯数字。
    """
    if not text:
        return []
    keywords: List[str] = []
    seen = set()
    for token in tokenize(text):
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def keyword_hit_rate(prediction: str, reference: str) -> Optional[float]:
    """关键词命中率 = 参考答案的关键词里, 有多少出现在模型回答中。

    返回 None 表示参考答案里抽不出关键词, 该题不计入平均分。
    """
    keywords = extract_keywords(reference)
    if not keywords:
        return None
    haystack = (prediction or "").lower()
    if not haystack.strip():
        return 0.0
    hit = sum(1 for word in keywords if word in haystack)
    return hit / float(len(keywords))


# --------------------------------------------------------------------------
# LLM 裁判
# --------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "你是一个严格的答案评测员。你会看到一个问题、一个参考答案和一个模型回答。"
    "请判断模型回答与参考答案在语义上是否一致。"
    "打分标准: 1 = 关键信息完全正确且完整; 0.5 = 部分正确或有重要遗漏; 0 = 错误、"
    "答非所问, 或凭空编造。"
    '只输出一个 JSON 对象, 不要输出别的任何内容, 格式为: '
    '{"score": 0 或 0.5 或 1, "reason": "一句话理由"}'
)


def parse_judge_output(text: str) -> Tuple[Optional[float], str]:
    """从裁判模型的输出里解析出分数。解析不了时返回 (None, 原因)。"""
    raw = (text or "").strip()
    if not raw:
        return None, "裁判返回了空内容"

    # 优先按 JSON 解析
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group())
            score = payload.get("score")
            reason = str(payload.get("reason", "")) or "裁判未给出理由"
            if isinstance(score, str):
                score = float(re.search(r"[0-9.]+", score).group())  # type: ignore[union-attr]
            score = float(score)
            return max(0.0, min(1.0, score)), reason
        except Exception:
            pass

    # 退化方案: 直接找 0 / 0.5 / 1 这样的数字
    number = re.search(r"(0(?:\.5)?|1(?:\.0)?)", raw)
    if number:
        try:
            return max(0.0, min(1.0, float(number.group()))), "从非 JSON 输出中解析出分数"
        except ValueError:
            pass
    return None, "无法解析裁判输出: {}".format(raw[:120])


def llm_judge(client, model: str, question: str, reference: str, prediction: str):
    """让大模型给模型回答打分, 返回 (分数, 理由)。"""
    user_prompt = (
        "问题:\n{}\n\n参考答案:\n{}\n\n模型回答:\n{}\n\n请按要求只输出 JSON。".format(
            question, reference, prediction if prediction else "(模型没有给出回答)"
        )
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
    except Exception as exc:
        return None, "裁判调用失败: {}".format(str(exc)[:160])
    return parse_judge_output(content)


# --------------------------------------------------------------------------
# 评测集读取
# --------------------------------------------------------------------------


@dataclass
class EvalItem:
    """一条评测样本。"""

    question: str
    answer: str
    source_pages: List[int] = field(default_factory=list)
    source_doc: str = ""
    line_no: int = 0

    @property
    def has_gold_location(self) -> bool:
        """是否有可用于判断检索命中的位置信息。"""
        return bool(self.source_pages) or bool(self.source_doc)


def _parse_pages(value) -> List[int]:
    """容忍 source_page 是 int / 字符串 / 列表等多种写法。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        pages: List[int] = []
        for item in value:
            pages.extend(_parse_pages(item))
        return pages
    try:
        page = int(str(value).strip())
    except (TypeError, ValueError):
        return []
    return [page] if page > 0 else []


def load_eval_set(path) -> List[EvalItem]:
    """读取 JSONL 评测集, 并对格式问题给出精确到行号的中文提示。"""
    eval_path = Path(path).expanduser().resolve()
    if not eval_path.exists():
        raise config.ConfigError(
            "找不到评测集文件: {}\n"
            "可以先复制示例文件再改:\n"
            "    copy data\\eval_set.example.jsonl data\\eval_set.jsonl\n"
            "示例文件在 project_rag/data/ 目录下。".format(eval_path)
        )

    items: List[EvalItem] = []
    try:
        with open(eval_path, "r", encoding="utf-8") as handle:
            raw_lines = handle.readlines()
    except OSError as exc:
        raise config.ConfigError("读取评测集失败: {}\n原始错误: {}".format(eval_path, exc)) from exc

    skipped: List[str] = []
    for line_no, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            skipped.append("第 {} 行不是合法 JSON ({})".format(line_no, exc.msg))
            continue
        if not isinstance(payload, dict):
            skipped.append("第 {} 行不是 JSON 对象".format(line_no))
            continue

        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not question or not answer:
            skipped.append("第 {} 行缺少 question 或 answer 字段".format(line_no))
            continue

        item = EvalItem(
            question=question,
            answer=answer,
            source_pages=_parse_pages(payload.get("source_page")),
            source_doc=str(payload.get("source_doc", "") or "").strip(),
            line_no=line_no,
        )
        if not item.has_gold_location:
            config.warn(
                "第 {} 行既没有 source_page 也没有 source_doc, 该题无法计算 recall@k, "
                "只会计入答案准确率。".format(line_no)
            )
        items.append(item)

    for message in skipped:
        config.warn("已跳过一条评测数据: {}".format(message))

    if not items:
        raise config.ConfigError(
            "评测集里没有可用的样本: {}\n"
            "每行必须是形如 {{\"question\": \"...\", \"answer\": \"...\", \"source_page\": 3}} 的 JSON 对象。".format(
                eval_path
            )
        )
    return items


# --------------------------------------------------------------------------
# 命中判断
# --------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    text = (name or "").strip().lower().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    for suffix in (".pdf", ".txt", ".md", ".docx"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def chunk_matches(chunk_doc: str, chunk_page: int, item: EvalItem) -> bool:
    """判断一个 chunk 是否就是该题的标准出处。

    规则:
      - 填了 source_doc: 文件名 (去掉扩展名) 必须一致
      - 填了 source_page: 页码必须命中其中之一
      - 两个都填: 必须同时满足
    """
    if item.source_doc:
        if _normalize_name(item.source_doc) != _normalize_name(chunk_doc):
            return False
    if item.source_pages:
        if int(chunk_page) not in item.source_pages:
            return False
    return True


def first_hit_rank(hits: Sequence[RetrievalHit], item: EvalItem) -> Optional[int]:
    """返回第一个命中 chunk 的排名 (从 1 开始), 没有命中返回 None。"""
    for rank, hit in enumerate(hits, start=1):
        if chunk_matches(hit.chunk.doc, hit.chunk.page, item):
            return rank
    return None


# --------------------------------------------------------------------------
# 评测主流程
# --------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """单题的评测明细。"""

    item: EvalItem
    hit: Optional[bool] = None
    rank: Optional[int] = None
    keyword_score: Optional[float] = None
    judge_score: Optional[float] = None
    judge_reason: str = ""
    prediction: str = ""
    error: str = ""
    top_chunks: List[str] = field(default_factory=list)


def _mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "待填" if value is None else ("{:.%df}" % digits).format(value)


def evaluate(
    index: RagIndex,
    items: Sequence[EvalItem],
    top_k: int = config.DEFAULT_TOP_K,
    judge: str = "keyword",
    use_bm25: bool = True,
    use_images: Optional[bool] = None,
    no_llm: bool = False,
    llm: Optional[config.LLMConfig] = None,
    show_details: bool = False,
) -> Dict[str, object]:
    """跑完整评测, 返回结果字典 (含 markdown 报告文本)。"""
    retriever = Retriever(index, use_bm25=use_bm25, use_images=use_images)
    config.info("检索方式: {}".format(retriever.describe()))

    # 准备 LLM 客户端
    client = None
    model_name = ""
    if not no_llm:
        if llm is None:
            llm = config.load_llm_config()
        client = config.build_openai_client(llm)  # 没有 key 会抛出中文 ConfigError
        model_name = llm.model
    else:
        config.warn("已开启 --no-llm: 不调用大模型, 答案准确率基于检索到的原文计算 (相当于检索上界)。")

    need_judge = judge in {"llm", "both"}
    judge_enabled = need_judge
    if need_judge and no_llm:
        config.warn("--no-llm 与 --judge {} 冲突, 已自动降级为关键词命中率。".format(judge))
        judge_enabled = False
    elif need_judge and client is None:
        config.warn("没有可用的 LLM 配置, 已自动降级为只算关键词命中率。")
        judge_enabled = False

    results: List[QuestionResult] = []
    started = time.time()

    for number, item in enumerate(items, start=1):
        config.emit("")
        config.emit("[{}/{}] {}".format(number, len(items), item.question), style="cyan")
        record = QuestionResult(item=item)
        try:
            hits = retriever.retrieve(item.question, top_k=top_k)
        except config.ConfigError as exc:
            record.error = str(exc)
            config.error("检索失败: {}".format(exc))
            results.append(record)
            continue

        record.top_chunks = [
            "{} 第 {} 页".format(hit.chunk.doc, hit.chunk.page) for hit in hits
        ]
        if item.has_gold_location:
            rank = first_hit_rank(hits, item)
            record.rank = rank
            record.hit = rank is not None
        else:
            record.hit = None

        if not hits:
            config.warn("没有检索到任何内容, 该题记 0 分。")
            record.hit = False if item.has_gold_location else None
            record.keyword_score = 0.0
            record.judge_score = 0.0 if judge_enabled else None
            results.append(record)
            continue

        # 生成回答
        if no_llm:
            # 不调用大模型时, 用检索到的原文拼接作为"回答", 相当于评测检索本身的质量
            prediction = "\n".join(hit.chunk.text for hit in hits)
            record.prediction = prediction
        else:
            from .rag import build_user_prompt, SYSTEM_PROMPT  # 避免顶层循环依赖

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(item.question, hits)},
            ]
            try:
                prediction = call_llm(messages, llm=llm)
            except config.ConfigError as exc:
                record.error = str(exc)
                config.error("生成回答失败: {}".format(exc))
                results.append(record)
                continue
            record.prediction = prediction

        record.keyword_score = keyword_hit_rate(prediction, item.answer)
        if judge_enabled and client is not None:
            score, reason = llm_judge(client, model_name, item.question, item.answer, prediction)
            record.judge_score = score
            record.judge_reason = reason

        config.emit(
            "  命中: {}   关键词命中率: {}   裁判分: {}".format(
                "是 (rank {})".format(record.rank) if record.hit else ("否" if record.hit is False else "无标准出处"),
                _fmt(record.keyword_score, 4),
                _fmt(record.judge_score, 2),
            )
        )
        results.append(record)

    elapsed = time.time() - started

    recall = _mean([1.0 if r.hit else 0.0 for r in results if r.hit is not None])
    mrr = _mean([1.0 / r.rank if r.rank else 0.0 for r in results if r.hit is not None])
    keyword_avg = _mean([r.keyword_score for r in results])
    judge_avg = _mean([r.judge_score for r in results])

    report = {
        "top_k": top_k,
        "judge": judge,
        "judge_enabled": judge_enabled,
        "no_llm": no_llm,
        "retriever": retriever.describe(),
        "index_dir": str(index.meta.get("index_dir", "")),
        "chunk_size": index.meta.get("chunk_size", "-"),
        "chunk_overlap": index.meta.get("chunk_overlap", "-"),
        "with_images": index.with_images,
        "num_samples": len(items),
        "num_location_samples": len([r for r in results if r.hit is not None]),
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "keyword_accuracy": keyword_avg,
        "judge_accuracy": judge_avg,
        "elapsed_sec": elapsed,
        "results": results,
    }
    report["markdown"] = render_markdown(report, show_details=show_details)
    return report


# --------------------------------------------------------------------------
# 报告渲染
# --------------------------------------------------------------------------


def render_markdown(report: Dict[str, object], show_details: bool = False) -> str:
    """生成可直接贴进 README 的 markdown 结果表。"""
    k = report.get("top_k")
    with_images = "是" if report.get("with_images") else "否"
    judge_enabled = bool(report.get("judge_enabled"))
    no_llm = bool(report.get("no_llm"))

    lines: List[str] = []
    lines.append("### RAG 评测结果")
    lines.append("")
    lines.append("| 运行配置 | 值 |")
    lines.append("| --- | --- |")
    lines.append("| 索引目录 | {} |".format(report.get("index_dir") or "-"))
    lines.append("| 检索方式 | {} |".format(report.get("retriever", "-")))
    lines.append("| 图像检索 (Stage 2) | {} |".format(with_images))
    lines.append("| chunk_size / chunk_overlap | {} / {} |".format(report.get("chunk_size"), report.get("chunk_overlap")))
    lines.append("| top_k | {} |".format(k))
    lines.append("| 评测样本数 (含标准出处) | {} / {} |".format(report.get("num_location_samples"), report.get("num_samples")))
    lines.append("| 答案生成 | {} |".format("仅检索原文 (--no-llm)" if no_llm else "大模型生成"))
    lines.append("| 裁判方式 | {} |".format(report.get("judge") if judge_enabled else "keyword (关键词命中率)"))
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append("| recall@{} | {} |".format(k, _fmt(report.get("recall_at_k"))))
    lines.append("| MRR@{} | {} |".format(k, _fmt(report.get("mrr_at_k"))))
    label = "关键词命中率 (基于检索原文)" if no_llm else "关键词命中率 (答案准确率)"
    lines.append("| {} | {} |".format(label, _fmt(report.get("keyword_accuracy"))))
    if judge_enabled:
        lines.append("| LLM 裁判平均分 (答案准确率) | {} |".format(_fmt(report.get("judge_accuracy"), 2)))
    lines.append("| 耗时 (秒) | {:.1f} |".format(float(report.get("elapsed_sec") or 0.0)))

    if show_details:
        lines.append("")
        lines.append("### 逐题明细")
        lines.append("")
        lines.append("| # | 问题 | 是否命中 | 命中排名 | 关键词命中率 | 裁判分 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for number, record in enumerate(report.get("results", []), start=1):  # type: ignore[arg-type]
            hit_text = "是" if record.hit else ("否" if record.hit is False else "-")
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    number,
                    _escape_cell(record.item.question),
                    hit_text,
                    record.rank if record.rank else "-",
                    _fmt(record.keyword_score),
                    _fmt(record.judge_score, 2) if judge_enabled else "-",
                )
            )
    return "\n".join(lines)


def _escape_cell(text: str, max_chars: int = 40) -> str:
    flat = re.sub(r"\s+", " ", text or "").replace("|", "/").strip()
    if len(flat) > max_chars:
        flat = flat[:max_chars] + "..."
    return flat


__all__ = [
    "EvalItem",
    "QuestionResult",
    "STOPWORDS",
    "evaluate",
    "extract_keywords",
    "keyword_hit_rate",
    "llm_judge",
    "load_eval_set",
    "parse_judge_output",
    "render_markdown",
    "chunk_matches",
    "first_hit_rank",
]

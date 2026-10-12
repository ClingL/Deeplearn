# -*- coding: utf-8 -*-
"""文本切块模块。

两个关键点:

1. **切块策略**: 不做粗暴的定长截断, 而是"段落 -> 句子 -> 按尺寸装箱"。
   优先在段落/句号边界切开, 保证每个 chunk 是语义完整的片段;
   只有遇到超长句 (例如一整页没有句号的表格) 才会硬切。
   ``chunk_size`` / ``chunk_overlap`` 都是 CLI 参数, 方便做消融实验。

2. **元数据溯源**: 每个 chunk 都带 ``{chunk_id, doc, page}``。
   检索命中后可以直接告诉用户"这句话来自 xxx.pdf 第 3 页", 用来算 recall@k。

本模块只依赖标准库 re, 不做分词 (分词与 BM25 打分统一放在 ``src/index.py``)。
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import config

# --------------------------------------------------------------------------
# 分句
# --------------------------------------------------------------------------

#: 中文句末标点
_CJK_TERMINATORS = "。！？；;…!?"
#: 句末标点后可能跟着的收尾符号 (引号、括号)
_TRAILING = "”’\"')）】》」』"
#: 按中文标点切句, 保留标点
_CJK_SENT_RE = re.compile(
    r"[^{term}\n]*[{term}]+[{trail}]*|[^{term}\n]+".format(
        term=re.escape(_CJK_TERMINATORS), trail=re.escape(_TRAILING)
    )
)
#: 英文句号切分 (只在没有中文标点的片段上使用, 避免切坏小数点和缩写)
_LATIN_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def split_sentences(text: str) -> List[str]:
    """把一段文本切成句子列表。先按中文标点切, 再对纯英文片段按 ". " 切。"""
    if not text:
        return []
    sentences: List[str] = []
    for block in _CJK_SENT_RE.findall(text):
        block = block.strip()
        if not block:
            continue
        if _has_cjk(block):
            sentences.append(block)
            continue
        # 纯英文片段: 按句号再切一次
        for piece in _LATIN_SENT_RE.split(block):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def split_paragraphs(text: str) -> List[str]:
    """按空行切段落; 没有空行的 PDF 会退化成按单行切, 仍然可用。"""
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in parts if p and p.strip()]
    if paragraphs:
        return paragraphs
    return [line.strip() for line in text.split("\n") if line.strip()]


def _hard_split(text: str, size: int, overlap: int) -> List[str]:
    """超长文本的兜底硬切 (带重叠)。正常情况下不应该走到这里。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    pieces: List[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):
            break
    return pieces


def _tail_overlap(sentences: Sequence[str], overlap: int) -> List[str]:
    """取上一块结尾的若干句作为下一块的重叠部分。

    只保留完整句子, 这样重叠部分也不会把句子切碎。最多保留 overlap 个字符。
    """
    if overlap <= 0 or not sentences:
        return []
    kept: List[str] = []
    total = 0
    for sent in reversed(sentences):
        if total + len(sent) > overlap:
            break
        kept.insert(0, sent)
        total += len(sent)
    # 至少给新块留出空间, 不允许把整块都搬过去
    if len(kept) >= len(sentences):
        kept = kept[1:]
    return kept


# --------------------------------------------------------------------------
# Chunk 数据结构
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    """一个检索单元。kind 为 "text" 或 "image"。"""

    chunk_id: str
    doc: str
    page: int
    text: str
    kind: str = "text"
    image_path: Optional[str] = None
    #: 命中的来源信息 (向量 / BM25 / 图像), 由检索阶段填充, 不落盘
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """人类可读的引用, 例如 ``report.pdf 第 3 页``。"""
        return "{} 第 {} 页".format(self.doc, self.page)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc": self.doc,
            "page": self.page,
            "kind": self.kind,
            "text": self.text,
            "image_path": self.image_path or "",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            chunk_id=str(data.get("chunk_id", "")),
            doc=str(data.get("doc", "")),
            page=int(data.get("page", 0) or 0),
            text=str(data.get("text", "")),
            kind=str(data.get("kind", "text")),
            image_path=(str(data["image_path"]) if data.get("image_path") else None),
        )

    def preview(self, max_chars: int = 120) -> str:
        """打印用的一行摘要。"""
        flat = re.sub(r"\s+", " ", self.text).strip()
        if len(flat) > max_chars:
            flat = flat[:max_chars] + " ..."
        return flat


def make_chunk_id(doc: str, page: int, index: int) -> str:
    """生成稳定且可读的 chunk id, 例如 ``report#p3#c0``。"""
    stem = doc.rsplit(".", 1)[0] if "." in doc else doc
    return "{}#p{}#c{}".format(stem, page, index)


# --------------------------------------------------------------------------
# 切块主逻辑
# --------------------------------------------------------------------------


def chunk_page_text(
    text: str,
    doc: str,
    page: int,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    start_index: int = 0,
) -> List[Chunk]:
    """把一页文本切成若干 chunk。

    参数:
        chunk_size:    每个 chunk 的目标最大字符数 (中文字符≈token 量级)
        chunk_overlap: 相邻 chunk 的重叠字符数, 用于避免答案正好落在切缝上
        start_index:   本页第一个 chunk 的编号 (同一页内从 0 开始递增)

    返回:
        Chunk 列表
    """
    if chunk_size <= 0:
        raise config.ConfigError("chunk_size 必须大于 0, 当前是 {}。".format(chunk_size))
    if chunk_overlap < 0:
        raise config.ConfigError("chunk_overlap 不能为负数, 当前是 {}。".format(chunk_overlap))
    if chunk_overlap >= chunk_size:
        # 自动收敛, 否则切块会原地踏步甚至死循环
        fixed = max(0, chunk_size // 4)
        config.warn(
            "chunk_overlap ({}) 不应大于等于 chunk_size ({}), 已自动调整为 {}。".format(
                chunk_overlap, chunk_size, fixed
            )
        )
        chunk_overlap = fixed

    if not text or not text.strip():
        return []

    chunks: List[Chunk] = []
    counter = start_index

    def flush(buffer: List[str]) -> None:
        """把当前缓冲的句子合成一个 chunk。"""
        nonlocal counter
        if not buffer:
            return
        content = "".join(buffer) if _has_cjk("".join(buffer)) else " ".join(buffer)
        content = content.strip()
        if not content:
            return
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc, page, counter),
                doc=doc,
                page=page,
                text=content,
                kind="text",
            )
        )
        counter += 1

    for paragraph in split_paragraphs(text):
        sentences = split_sentences(paragraph)
        if not sentences:
            continue

        buffer: List[str] = []
        buffer_len = 0

        for sentence in sentences:
            # 情况一: 单句就超过 chunk_size, 无法装箱, 只能硬切
            if len(sentence) > chunk_size:
                flush(buffer)
                buffer = []
                buffer_len = 0
                for piece in _hard_split(sentence, chunk_size, chunk_overlap):
                    chunks.append(
                        Chunk(
                            chunk_id=make_chunk_id(doc, page, counter),
                            doc=doc,
                            page=page,
                            text=piece,
                            kind="text",
                        )
                    )
                    counter += 1
                continue

            # 情况二: 装不下了, 先结块, 再把结尾若干句作为重叠
            if buffer and buffer_len + len(sentence) > chunk_size:
                flush(buffer)
                buffer = _tail_overlap(buffer, chunk_overlap)
                buffer_len = sum(len(s) for s in buffer)

            buffer.append(sentence)
            buffer_len += len(sentence)

        flush(buffer)

    return chunks


def build_chunks(
    pages,
    doc_name: str,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """把 LoadedDocument 的所有页切成一个扁平的 chunk 列表。

    参数:
        pages: PageContent 列表 (来自 loaders.load_pdf)
    """
    all_chunks: List[Chunk] = []
    for page in pages:
        page_chunks = chunk_page_text(
            text=page.text,
            doc=doc_name,
            page=page.page,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_index=0,
        )
        all_chunks.extend(page_chunks)
    return all_chunks


def build_image_chunks(images, doc_name: str) -> List[Chunk]:
    """把 ImageRecord 列表转成 image 类型的 chunk (Stage 2 使用)。

    图片 chunk 的 text 是它的 caption, 只用于终端展示与引用, 不参与 BM25。
    """
    chunks: List[Chunk] = []
    for idx, image in enumerate(images):
        caption = image.caption or "[图片] 出自 {} 第 {} 页".format(doc_name, image.page)
        chunks.append(
            Chunk(
                chunk_id="{}#p{}#img{}".format(doc_name.rsplit(".", 1)[0], image.page, idx),
                doc=doc_name,
                page=image.page,
                text=caption,
                kind="image",
                image_path=str(image.image_path) if image.image_path else None,
            )
        )
    return chunks


# --------------------------------------------------------------------------
# 对外导出
# --------------------------------------------------------------------------


__all__ = [
    "Chunk",
    "split_sentences",
    "split_paragraphs",
    "chunk_page_text",
    "build_chunks",
    "build_image_chunks",
    "make_chunk_id",
]

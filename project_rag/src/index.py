# -*- coding: utf-8 -*-
"""索引与检索模块。

包含四部分:

1. ``tokenize``  中文/英文混合分词 (正则实现, 零额外依赖)
                 中文用"单字 + 字符二元组(bigram)"切分, 例如
                 "深度学习" -> ["深","度","学","习","深度","度学","学习"]。
                 这是没装专业分词器时中文检索的标准近似做法。

2. ``BM25``      自己实现的 Okapi BM25 (k1=1.5, b=0.75), 约 40 行, 无外部依赖。

3. ``rrf_fuse``  Reciprocal Rank Fusion, 把多路召回结果按"排名"融合。
                 用排名而不是分数, 因为向量相似度和 BM25 分数根本不在一个量纲上。

4. ``RagIndex``  FAISS 向量索引 (IndexFlatIP, 向量已归一化, 内积=余弦相似度)
                 + BM25 关键词索引 + 可选的图像索引, 统一负责建立/保存/加载/检索。

为什么用混合检索: 向量召回擅长语义相近但用词不同的情况, BM25 擅长人名、
编号、专有名词这类"必须精确命中"的情况, 两者互补。
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
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .chunking import Chunk

# --------------------------------------------------------------------------
# 索引目录中的文件名
# --------------------------------------------------------------------------

INDEX_META_FILE = "meta.json"
INDEX_CHUNKS_FILE = "chunks.jsonl"
INDEX_TEXT_VECTORS_FILE = "text.faiss"
INDEX_IMAGE_VECTORS_FILE = "image.faiss"
INDEX_BM25_FILE = "bm25_tokens.json"

#: 检索来源名称, 用于终端展示
SOURCE_VECTOR = "向量"
SOURCE_BM25 = "BM25"
SOURCE_IMAGE = "图像"


# --------------------------------------------------------------------------
# 1. 分词 (零依赖)
# --------------------------------------------------------------------------

#: 连续的中日韩字符
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
#: 英文单词 / 数字 (含小数与百分号前的数字)
_LATIN_RE = re.compile(r"[a-z]+|[0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> List[str]:
    """把一段文本切成检索用的 token, 只用标准库 re, 不依赖任何分词器。

    规则:
      - 中文连续片段: 既输出每个单字, 也输出相邻两个字组成的 bigram。
        单字保证召回 (不会漏), bigram 保证区分度 (比单字更有信息量)。
      - 英文与数字: 按非字母数字边界切开, 并统一转小写。
      - 标点与空白直接丢弃。

    示例::

        >>> tokenize("深度学习 Deep Learning 3.5")
        ['深', '度', '学', '习', '深度', '度学', '学习', 'deep', 'learning', '3.5']
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: List[str] = []
    for match in _CJK_RUN_RE.finditer(lowered):
        run = match.group()
        tokens.extend(run)
        if len(run) > 1:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(_LATIN_RE.findall(lowered))
    return tokens


# --------------------------------------------------------------------------
# 2. BM25 (自己实现, 无第三方依赖)
# --------------------------------------------------------------------------


class BM25:
    """Okapi BM25 打分器。

    公式::

        score(q, d) = sum_t IDF(t) * f(t,d) * (k1 + 1)
                      / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

        IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

    其中 N 是文档总数, n(t) 是包含 t 的文档数, f(t,d) 是 t 在 d 中的词频,
    |d| 是文档长度, avgdl 是平均文档长度。IDF 用 ln(1 + x) 形式, 保证非负。

    参数:
        corpus_tokens: 每篇文档的 token 列表
        k1: 词频饱和参数, 1.2 ~ 2.0 之间较常用
        b:  文档长度归一化强度, 0 表示不归一化, 1 表示完全归一化
    """

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus_size = len(corpus_tokens)
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}

        doc_freq: Dict[str, int] = {}
        for tokens in corpus_tokens:
            term_freq: Dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self.doc_freqs.append(term_freq)
            self.doc_lengths.append(len(tokens))
            for token in term_freq:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.avgdl = (sum(self.doc_lengths) / self.corpus_size) if self.corpus_size else 0.0
        n_docs = self.corpus_size
        for token, freq in doc_freq.items():
            self.idf[token] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        """返回每篇文档的 BM25 分数 (长度 = 文档数)。"""
        scores = np.zeros(self.corpus_size, dtype="float32")
        if self.corpus_size == 0 or self.avgdl <= 0:
            return scores

        for token in query_tokens:
            idf = self.idf.get(token)
            if not idf:
                continue
            for doc_index, term_freq in enumerate(self.doc_freqs):
                freq = term_freq.get(token)
                if not freq:
                    continue
                length_norm = 1.0 - self.b + self.b * (self.doc_lengths[doc_index] / self.avgdl)
                scores[doc_index] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * length_norm)
        return scores


# --------------------------------------------------------------------------
# 3. RRF 融合
# --------------------------------------------------------------------------


def rrf_fuse(
    rankings: Sequence[Tuple[str, Sequence[Tuple[str, float]]]],
    k: int = config.DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float, Dict[str, int]]]:
    """Reciprocal Rank Fusion。

    参数:
        rankings: [(来源名, [(chunk_id, 原始分数), ...]), ...],
                  每个来源内部必须已经按分数从高到低排好序
        k:        RRF 平滑常数, 论文推荐 60
        weights:  各来源的权重, 默认全部为 1

    返回:
        [(chunk_id, 融合分数, {来源名: 排名}), ...], 按融合分数降序

    公式: score(chunk) = sum_over_sources weight / (k + rank), rank 从 1 开始。
    只用排名不用原始分数, 因此不需要对向量相似度和 BM25 分数做任何量纲对齐。
    """
    fused_scores: Dict[str, float] = {}
    fused_ranks: Dict[str, Dict[str, int]] = {}

    for source_index, (source_name, items) in enumerate(rankings):
        weight = 1.0
        if weights and source_index < len(weights):
            weight = float(weights[source_index])
        for rank, (chunk_id, _score) in enumerate(items, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + weight / (k + rank)
            fused_ranks.setdefault(chunk_id, {})[source_name] = rank

    ordered = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [(chunk_id, score, fused_ranks.get(chunk_id, {})) for chunk_id, score in ordered]


# --------------------------------------------------------------------------
# 4. 检索结果
# --------------------------------------------------------------------------


@dataclass
class RetrievalHit:
    """一条检索命中结果。"""

    chunk: Chunk
    score: float
    sources: Dict[str, int] = field(default_factory=dict)

    def source_desc(self) -> str:
        """例如: "向量 rank1 / BM25 rank3"。"""
        if not self.sources:
            return "未知来源"
        return " / ".join("{} rank{}".format(name, rank) for name, rank in self.sources.items())


# --------------------------------------------------------------------------
# FAISS 导入 (延迟)
# --------------------------------------------------------------------------


def _import_faiss():
    """导入 faiss, 失败时给出中文提示。"""
    try:
        import faiss

        return faiss
    except ImportError as exc:
        raise config.ConfigError(
            "缺少 faiss-cpu 依赖, 无法建立/查询向量索引。\n"
            "请用项目虚拟环境安装依赖:\n"
            "    .venv\\Scripts\\python.exe -m pip install -r project_rag/requirements-rag.txt\n"
            "原始错误: {}".format(exc)
        ) from exc


# --------------------------------------------------------------------------
# 索引本体
# --------------------------------------------------------------------------


class RagIndex:
    """文本向量索引 + BM25 索引 (+ 可选图像向量索引)。

    目录结构::

        index_dir/
            meta.json         索引元信息 (模型名、切块参数、文档列表等)
            chunks.jsonl      所有 chunk 的元数据 (含 doc/page, 用于溯源)
            text.faiss        文本向量索引 (IndexFlatIP)
            image.faiss       图像向量索引 (开启 --with-images 时才有)
            bm25_tokens.json  BM25 的文档 token 序列 (加载时重建打分器, 避免 pickle 兼容问题)
    """

    def __init__(self) -> None:
        self.meta: Dict[str, object] = {}
        self.chunks: List[Chunk] = []
        self.text_ids: List[str] = []  # 与 text_index 的行号一一对应
        self.image_ids: List[str] = []  # 与 image_index 的行号一一对应
        self.text_index = None
        self.image_index = None
        self.bm25: Optional[BM25] = None
        self._chunk_map: Dict[str, Chunk] = {}

    # ---------------------------------------------------------------- 基本信息
    @property
    def with_images(self) -> bool:
        return bool(self.meta.get("with_images", False))

    @property
    def num_text_chunks(self) -> int:
        return len([c for c in self.chunks if c.kind == "text"])

    @property
    def num_image_chunks(self) -> int:
        return len([c for c in self.chunks if c.kind == "image"])

    @property
    def docs(self) -> List[str]:
        return list(dict.fromkeys(c.doc for c in self.chunks))

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunk_map.get(chunk_id)

    def summary(self) -> str:
        lines = [
            "文档数: {} ({})".format(len(self.docs), ", ".join(self.docs[:5])),
            "文本 chunk 数: {}".format(self.num_text_chunks),
            "图像 chunk 数: {}".format(self.num_image_chunks),
            "切块参数: chunk_size={} chunk_overlap={}".format(
                self.meta.get("chunk_size", "-"), self.meta.get("chunk_overlap", "-")
            ),
            "文本向量模型: {} (dim={})".format(
                self.meta.get("text_model", "-"), self.meta.get("text_dim", "-")
            ),
            "图像检索: {}".format("已开启 (CLIP)" if self.with_images else "未开启"),
        ]
        return "\n".join("  " + line for line in lines)

    # ---------------------------------------------------------------- 建立索引
    def set_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._chunk_map = {c.chunk_id: c for c in self.chunks}

    def build_text_index(self, vectors: np.ndarray, chunk_ids: Sequence[str]) -> None:
        """建立文本向量索引。向量会再归一化一次, 配合 IndexFlatIP 得到余弦相似度。"""
        faiss = _import_faiss()
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            self.text_index = None
            self.text_ids = []
            return
        if vectors.shape[0] != len(chunk_ids):
            raise config.ConfigError(
                "文本向量数量 ({}) 与 chunk 数量 ({}) 不一致, 无法建立索引。".format(
                    vectors.shape[0], len(chunk_ids)
                )
            )
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)
        self.text_index = index
        self.text_ids = list(chunk_ids)

    def build_image_index(self, vectors: np.ndarray, chunk_ids: Sequence[str]) -> None:
        """建立图像向量索引 (Stage 2)。"""
        faiss = _import_faiss()
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            self.image_index = None
            self.image_ids = []
            return
        if vectors.shape[0] != len(chunk_ids):
            raise config.ConfigError(
                "图像向量数量 ({}) 与图像 chunk 数量 ({}) 不一致, 无法建立索引。".format(
                    vectors.shape[0], len(chunk_ids)
                )
            )
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)
        self.image_index = index
        self.image_ids = list(chunk_ids)

    def build_bm25(self, text_chunks: Sequence[Chunk]) -> None:
        """建立 BM25 索引。中文分词用本模块的 tokenize (单字 + bigram)。"""
        corpus_tokens = [tokenize(chunk.text) for chunk in text_chunks]
        if not corpus_tokens:
            self.bm25 = None
            return
        self.bm25 = BM25(corpus_tokens)

    # ---------------------------------------------------------------- 持久化
    def save(self, index_dir) -> Path:
        """把索引写入目录 (目录不存在会自动创建)。"""
        faiss = _import_faiss()
        out_dir = Path(index_dir).expanduser().resolve()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise config.ConfigError(
                "无法创建索引目录: {}\n请检查路径是否存在、是否有写权限。\n原始错误: {}".format(
                    out_dir, exc
                )
            ) from exc

        self.meta.setdefault("version", config.INDEX_FORMAT_VERSION)
        self.meta["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.meta["index_dir"] = str(out_dir)
        self.meta["num_chunks"] = len(self.chunks)
        self.meta["num_text_chunks"] = self.num_text_chunks
        self.meta["num_image_chunks"] = self.num_image_chunks
        self.meta["docs"] = self.docs
        self.meta["use_bm25"] = self.bm25 is not None

        _write_json(out_dir / INDEX_META_FILE, self.meta)

        with open(out_dir / INDEX_CHUNKS_FILE, "w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        if self.text_index is not None:
            faiss.write_index(self.text_index, str(out_dir / INDEX_TEXT_VECTORS_FILE))
        if self.image_index is not None:
            faiss.write_index(self.image_index, str(out_dir / INDEX_IMAGE_VECTORS_FILE))

        # BM25 只存 token 序列, 加载时重建 (JSON 比 pickle 稳定, 也不怕版本差异)
        bm25_payload = {
            "chunk_ids": list(self.text_ids),
            "tokens": [tokenize(self._chunk_map[cid].text) for cid in self.text_ids if cid in self._chunk_map],
        }
        _write_json(out_dir / INDEX_BM25_FILE, bm25_payload)
        return out_dir

    @classmethod
    def load(cls, index_dir) -> "RagIndex":
        """从目录加载索引。目录不存在或缺少文件时给出"先跑 ingest"的中文提示。"""
        faiss = _import_faiss()
        root = Path(index_dir).expanduser().resolve()
        if not root.exists():
            raise config.ConfigError(_missing_index_message(root, "索引目录不存在"))
        if not root.is_dir():
            raise config.ConfigError(
                "{}\n给的是文件而不是索引目录。".format(_missing_index_message(root, "路径不是目录"))
            )

        meta_path = root / INDEX_META_FILE
        chunks_path = root / INDEX_CHUNKS_FILE
        if not meta_path.exists() or not chunks_path.exists():
            raise config.ConfigError(
                _missing_index_message(root, "目录里缺少 meta.json 或 chunks.jsonl")
            )

        index = cls()
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                index.meta = json.load(handle)
        except Exception as exc:
            raise config.ConfigError(
                "读取 {} 失败, 文件可能损坏, 建议删掉索引目录重新 ingest。\n原始错误: {}".format(
                    meta_path, exc
                )
            ) from exc

        chunks: List[Chunk] = []
        try:
            with open(chunks_path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunks.append(Chunk.from_dict(json.loads(line)))
                    except Exception as exc:
                        config.warn("chunks.jsonl 第 {} 行解析失败, 已跳过: {}".format(line_no, exc))
        except OSError as exc:
            raise config.ConfigError(
                "读取 {} 失败: {}".format(chunks_path, exc)
            ) from exc
        if not chunks:
            raise config.ConfigError(
                "{}\nchunks.jsonl 里没有任何 chunk, 说明上次 ingest 没有成功抽取到文本。\n"
                "请检查 PDF 是否为扫描版 (没有文字层)。".format(_missing_index_message(root, "索引为空"))
            )
        index.set_chunks(chunks)

        text_path = root / INDEX_TEXT_VECTORS_FILE
        if text_path.exists():
            try:
                index.text_index = faiss.read_index(str(text_path))
            except Exception as exc:
                raise config.ConfigError(
                    "读取 FAISS 文本索引失败 ({})。建议删除索引目录后重新 ingest。\n"
                    "原始错误: {}".format(text_path, exc)
                ) from exc
            index.text_ids = [c.chunk_id for c in chunks if c.kind == "text"]
            if index.text_index.ntotal != len(index.text_ids):
                config.warn(
                    "文本索引行数 ({}) 与文本 chunk 数 ({}) 不一致, 可能索引目录是旧版本产生的, "
                    "建议重新 ingest。".format(index.text_index.ntotal, len(index.text_ids))
                )
        else:
            config.warn("索引目录里没有 text.faiss, 只能使用 BM25 关键词检索。")

        image_path = root / INDEX_IMAGE_VECTORS_FILE
        if image_path.exists():
            try:
                index.image_index = faiss.read_index(str(image_path))
                index.image_ids = [c.chunk_id for c in chunks if c.kind == "image"]
            except Exception as exc:
                config.warn("读取 FAISS 图像索引失败, 已忽略图像检索: {}".format(exc))
                index.image_index = None
                index.image_ids = []

        bm25_path = root / INDEX_BM25_FILE
        if bm25_path.exists():
            try:
                with open(bm25_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                tokens = payload.get("tokens") or []
                if tokens:
                    index.bm25 = BM25(tokens)
            except Exception as exc:
                config.warn("读取 BM25 索引失败, 已退化为纯向量检索: {}".format(exc))
                index.bm25 = None
        else:
            config.warn("索引目录里没有 bm25_tokens.json, 只能使用向量检索。")

        return index

    # ---------------------------------------------------------------- 检索
    def search_vector(self, query_vector: np.ndarray, top_n: int = config.DEFAULT_CANDIDATE_N):
        """向量召回, 返回 [(chunk_id, 余弦相似度), ...]。"""
        if self.text_index is None or not self.text_ids:
            return []
        faiss = _import_faiss()
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.text_index.d:
            raise config.ConfigError(
                "查询向量维度 ({}) 与索引维度 ({}) 不一致。\n"
                "通常是换过向量模型或索引是旧版本产生的, 请重新 ingest。".format(
                    query.shape[1], self.text_index.d
                )
            )
        faiss.normalize_L2(query)
        top_n = max(1, min(int(top_n), len(self.text_ids)))
        scores, indices = self.text_index.search(query, top_n)
        results: List[Tuple[str, float]] = []
        for score, position in zip(scores[0], indices[0]):
            position = int(position)
            if position < 0 or position >= len(self.text_ids):
                continue
            results.append((self.text_ids[position], float(score)))
        return results

    def search_image(self, clip_query_vector: np.ndarray, top_n: int = 5):
        """跨模态召回: 用 CLIP 文本向量在图像索引里检索。"""
        if self.image_index is None or not self.image_ids:
            return []
        faiss = _import_faiss()
        query = np.asarray(clip_query_vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.image_index.d:
            config.warn(
                "CLIP 查询向量维度 ({}) 与图像索引维度 ({}) 不一致, 已跳过图像检索。".format(
                    query.shape[1], self.image_index.d
                )
            )
            return []
        faiss.normalize_L2(query)
        top_n = max(1, min(int(top_n), len(self.image_ids)))
        scores, indices = self.image_index.search(query, top_n)
        results: List[Tuple[str, float]] = []
        for score, position in zip(scores[0], indices[0]):
            position = int(position)
            if position < 0 or position >= len(self.image_ids):
                continue
            results.append((self.image_ids[position], float(score)))
        return results

    def search_bm25(self, query: str, top_n: int = config.DEFAULT_CANDIDATE_N):
        """BM25 关键词召回, 返回 [(chunk_id, BM25 分数), ...] (只保留分数大于 0 的)。"""
        if self.bm25 is None or not self.text_ids:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        if scores.size == 0:
            return []
        top_n = max(1, min(int(top_n), len(scores)))
        order = np.argsort(-scores)[:top_n]
        results: List[Tuple[str, float]] = []
        for position in order:
            position = int(position)
            score = float(scores[position])
            if score <= 0:
                continue
            if position >= len(self.text_ids):
                continue
            results.append((self.text_ids[position], score))
        return results

    def hybrid_search(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        clip_query_vector: Optional[np.ndarray] = None,
        top_k: int = config.DEFAULT_TOP_K,
        use_bm25: bool = True,
        rrf_k: int = config.DEFAULT_RRF_K,
        candidate_n: int = config.DEFAULT_CANDIDATE_N,
    ) -> List[RetrievalHit]:
        """混合检索主入口: 各路召回 -> RRF 融合 -> 取前 top_k。

        参数:
            query:            用户问题原文 (BM25 用)
            query_vector:     bge 文本查询向量 (向量召回用)
            clip_query_vector: CLIP 文本查询向量 (图像召回用, Stage 2)
            candidate_n:      每路召回各取多少条候选再融合 (取大一点召回更全)
        """
        rankings: List[Tuple[str, Sequence[Tuple[str, float]]]] = []

        if query_vector is not None:
            vector_hits = self.search_vector(query_vector, candidate_n)
            if vector_hits:
                rankings.append((SOURCE_VECTOR, vector_hits))

        if use_bm25 and self.bm25 is not None:
            bm25_hits = self.search_bm25(query, candidate_n)
            if bm25_hits:
                rankings.append((SOURCE_BM25, bm25_hits))

        if clip_query_vector is not None and self.image_index is not None:
            image_hits = self.search_image(clip_query_vector, max(3, top_k))
            if image_hits:
                rankings.append((SOURCE_IMAGE, image_hits))

        if not rankings:
            return []

        fused = rrf_fuse(rankings, k=rrf_k)
        hits: List[RetrievalHit] = []
        for chunk_id, score, sources in fused[: max(1, int(top_k))]:
            chunk = self.get_chunk(chunk_id)
            if chunk is None:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=score, sources=sources))
        return hits


# --------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------


def _write_json(path: Path, payload) -> None:
    """写 JSON, 统一 UTF-8 且不转义中文。"""
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise config.ConfigError("写入文件失败: {}\n原始错误: {}".format(path, exc)) from exc


def _missing_index_message(root: Path, reason: str) -> str:
    """拼装"索引不存在"的友好提示。"""
    lines = [
        "无法加载索引。",
        "  原因: {}".format(reason),
        "  索引目录: {}".format(root),
        "",
        "请先建立索引, 例如:",
        "    python main.py ingest --pdf data/你的文档.pdf --out {}".format(root.name),
        "如果已经建过索引, 请确认 --index 参数指向的目录是否正确。",
    ]
    return "\n".join(lines)


__all__ = [
    "BM25",
    "RagIndex",
    "RetrievalHit",
    "tokenize",
    "rrf_fuse",
    "SOURCE_VECTOR",
    "SOURCE_BM25",
    "SOURCE_IMAGE",
    "INDEX_META_FILE",
    "INDEX_CHUNKS_FILE",
    "INDEX_TEXT_VECTORS_FILE",
    "INDEX_IMAGE_VECTORS_FILE",
    "INDEX_BM25_FILE",
]

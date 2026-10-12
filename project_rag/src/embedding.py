# -*- coding: utf-8 -*-
"""向量化模块。

两条独立的编码链路:

    TextEmbedder  : BAAI/bge-small-zh-v1.5 (约 100MB, 512 维) -> 编码文本
    ImageEmbedder : openai/clip-vit-base-patch32 (约 600MB, 512 维) -> 编码图像
                    CLIP 的文本塔同时用于把"查询文字"编码到图像空间,
                    从而实现"用文字搜图片"的跨模态检索。

工程注意:
    1. 两个模型都在首次使用时自动下载, 缓存目录被 config.apply_hf_env()
       固定到项目内的 hf_cache/, 不会污染用户主目录。
    2. 所有向量都做 L2 归一化, 配合 FAISS 的 IndexFlatIP,
       内积 (inner product) 就等于余弦相似度。
    3. 模型是懒加载的: 只有在真正需要时才 import torch / sentence-transformers,
       这样 `--help`、`eval --judge keyword` 之类的操作不会被拖慢。
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


from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import config


# --------------------------------------------------------------------------
# 依赖导入
# --------------------------------------------------------------------------


def _require_numpy() -> None:
    if np is None:  # pragma: no cover
        raise config.ConfigError("缺少 numpy 依赖, 请先安装项目依赖。")


def _import_sentence_transformers():
    """导入 sentence-transformers, 失败时给出中文提示。"""
    config.apply_hf_env()  # 必须在导入前设置缓存目录
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise config.ConfigError(
            "缺少 sentence-transformers 依赖, 无法生成文本向量。\n"
            "请用项目虚拟环境安装依赖:\n"
            "    .venv\\Scripts\\python.exe -m pip install -r project_rag/requirements-rag.txt\n"
            "原始错误: {}".format(exc)
        ) from exc
    return SentenceTransformer


def _import_clip():
    """导入 transformers 的 CLIP 模型, 失败时给出中文提示。"""
    config.apply_hf_env()
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise config.ConfigError(
            "缺少 transformers / torch 依赖, 无法生成图像向量 (CLIP)。\n"
            "sentence-transformers 会一并安装 transformers; 请确认已安装项目依赖:\n"
            "    .venv\\Scripts\\python.exe -m pip install -r project_rag/requirements-rag.txt\n"
            "原始错误: {}".format(exc)
        ) from exc
    return torch, CLIPModel, CLIPProcessor


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化。零向量保持不变 (避免除零)。"""
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """对外暴露的归一化函数, 索引模块也会用到。"""
    return _l2_normalize(matrix)


# --------------------------------------------------------------------------
# 文本向量
# --------------------------------------------------------------------------


class TextEmbedder:
    """基于 sentence-transformers 的中文文本向量模型。"""

    def __init__(self, model_name: str = config.DEFAULT_TEXT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dim: Optional[int] = None

    # -- 懒加载 ---------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        SentenceTransformer = _import_sentence_transformers()
        config.info(
            "正在加载文本向量模型: {} (首次运行需要联网下载约 100MB)。\n"
            "       下载缓存目录: {}".format(self.model_name, config.HF_CACHE_DIR)
        )
        try:
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except Exception as exc:
            raise config.ConfigError(
                "加载文本向量模型失败: {}\n"
                "常见原因: 1) 网络无法访问 HuggingFace; 2) 缓存目录被占用; "
                "3) 模型名写错。\n"
                "如需走镜像, 可以先设置环境变量 HF_ENDPOINT=https://hf-mirror.com 再运行。\n"
                "原始错误: {}".format(self.model_name, exc)
            ) from exc
        return self._model

    # -- 属性 -----------------------------------------------------------
    @property
    def dim(self) -> int:
        """向量维度。bge-small-zh-v1.5 为 512。"""
        if self._dim is None:
            model = self._ensure_model()
            try:
                self._dim = int(model.get_sentence_embedding_dimension())
            except Exception:
                self._dim = int(model.encode(["测试"], normalize_embeddings=False).shape[1])
        return int(self._dim)

    # -- 编码 -----------------------------------------------------------
    def encode_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """编码文档 (入库用)。返回 shape=(n, dim) 的归一化 float32 矩阵。"""
        _require_numpy()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        model = self._ensure_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise config.ConfigError(
                "文本向量编码失败 (共 {} 条)。\n"
                "可以尝试把 batch_size 调小, 或检查是否有超长文本 (超过模型最大长度 512)。\n"
                "原始错误: {}".format(len(texts), exc)
            ) from exc
        return _l2_normalize(np.asarray(vectors, dtype="float32"))

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询。bge 系列官方建议给"查询"加指令前缀, 文档侧不加。"""
        _require_numpy()
        model = self._ensure_model()
        text = config.BGE_QUERY_INSTRUCTION + (query or "")
        try:
            vector = model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise config.ConfigError("查询向量编码失败。\n原始错误: {}".format(exc)) from exc
        return _l2_normalize(np.asarray(vector, dtype="float32"))[0]


# --------------------------------------------------------------------------
# 图像向量 (CLIP)
# --------------------------------------------------------------------------


class ImageEmbedder:
    """基于 CLIP 的图像/文本向量模型, 用于跨模态检索 (Stage 2)。

    简易实现的说明: 图像向量与文本向量各自的维度空间不同 (虽然这里都是 512 维,
    但语义空间是两套), 不能直接放进同一个 FAISS 索引比较。本项目采用的是
    "文本索引 + 图像索引分开建, 检索时两边都查, 再用 RRF 合并结果"的简易方案。
    """

    def __init__(self, model_name: str = config.DEFAULT_CLIP_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._torch = None
        self._dim: Optional[int] = None

    # -- 懒加载 ---------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model, self._processor, self._torch
        torch, CLIPModel, CLIPProcessor = _import_clip()
        config.info(
            "正在加载 CLIP 图像向量模型: {} (首次运行需要联网下载约 600MB)。\n"
            "       下载缓存目录: {}".format(self.model_name, config.HF_CACHE_DIR)
        )
        try:
            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained(self.model_name)
            processor = CLIPProcessor.from_pretrained(self.model_name)
            model = model.to(device)
            model.eval()
        except Exception as exc:
            raise config.ConfigError(
                "加载 CLIP 模型失败: {}\n"
                "常见原因: 1) 网络无法访问 HuggingFace; 2) 磁盘空间不足。\n"
                "如需走镜像, 可以先设置环境变量 HF_ENDPOINT=https://hf-mirror.com 再运行。\n"
                "原始错误: {}".format(self.model_name, exc)
            ) from exc
        self._model, self._processor, self._torch = model, processor, torch
        self._device = device
        return self._model, self._processor, self._torch

    @property
    def device(self) -> str:  # type: ignore[override]
        self._ensure_model()
        return getattr(self, "_device", "cpu")

    @property
    def dim(self) -> int:
        """图像向量维度。clip-vit-base-patch32 的投影维度为 512。"""
        if self._dim is None:
            model, _, _ = self._ensure_model()
            try:
                self._dim = int(model.config.projection_dim)
            except Exception:
                self._dim = 512
        return int(self._dim)

    # -- 编码 -----------------------------------------------------------
    def encode_images(self, image_paths: Sequence[str], batch_size: int = 8) -> np.ndarray:
        """批量编码图片文件, 返回 shape=(n, dim) 的归一化 float32 矩阵。

        读取失败的图片会被跳过, 并用零向量占位, 保证返回行数与输入一一对应。
        """
        _require_numpy()
        model, processor, torch = self._ensure_model()
        device = getattr(self, "_device", "cpu")
        paths = [Path(p) for p in image_paths]
        if not paths:
            return np.zeros((0, self.dim), dtype="float32")

        try:
            from PIL import Image
        except ImportError as exc:
            raise config.ConfigError(
                "缺少 Pillow 依赖, 无法读取图片。\n"
                "请用项目虚拟环境安装: pip install Pillow\n原始错误: {}".format(exc)
            ) from exc

        results: List[Optional[np.ndarray]] = []
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            images = []
            valid_flags: List[bool] = []
            for path in batch:
                try:
                    with Image.open(str(path)) as handle:
                        images.append(handle.convert("RGB").copy())
                    valid_flags.append(True)
                except Exception as exc:
                    config.warn("图片读取失败, 已用零向量占位: {} ({})".format(path, exc))
                    images.append(None)
                    valid_flags.append(False)

            valid_images = [img for img in images if img is not None]
            if valid_images:
                try:
                    inputs = processor(images=valid_images, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        features = model.get_image_features(**inputs)
                    features = features.detach().cpu().numpy().astype("float32")
                    features = _l2_normalize(features)
                except Exception as exc:
                    raise config.ConfigError(
                        "CLIP 图像编码失败。\n"
                        "如果显存/内存不足, 可以用 --device cpu 或者减少图片数量后重试。\n"
                        "原始错误: {}".format(exc)
                    ) from exc
            else:
                features = np.zeros((0, self.dim), dtype="float32")

            cursor = 0
            for ok in valid_flags:
                if ok:
                    results.append(features[cursor])
                    cursor += 1
                else:
                    results.append(np.zeros(self.dim, dtype="float32"))

        matrix = np.asarray(results, dtype="float32")
        return _l2_normalize(matrix)

    def encode_query(self, query: str) -> np.ndarray:
        """用 CLIP 的文本塔把问题编码到图像空间, 实现"用文字搜图"。"""
        _require_numpy()
        model, processor, torch = self._ensure_model()
        device = getattr(self, "_device", "cpu")
        try:
            inputs = processor(text=[query or ""], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = model.get_text_features(**inputs)
            features = features.detach().cpu().numpy().astype("float32")
        except Exception as exc:
            raise config.ConfigError("CLIP 文本编码失败。\n原始错误: {}".format(exc)) from exc
        return _l2_normalize(features)[0]


__all__ = ["TextEmbedder", "ImageEmbedder", "l2_normalize"]

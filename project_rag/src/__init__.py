# -*- coding: utf-8 -*-
"""多模态文档问答系统 (RAG) 的核心包。

模块划分::

    config     全局配置 (.env / 模型缓存目录 / 统一终端输出)
    loaders    PDF 解析: 抽文本 + 抽图片 (PyMuPDF)
    chunking   文本切块 + 中文分词工具, 保留 (doc, page) 元数据用于溯源
    embedding  文本向量 (bge-small-zh-v1.5) 与图像向量 (CLIP)
    index      FAISS 向量索引 + BM25 关键词索引 + RRF 混合检索
    rag        检索 -> 组装 Prompt -> 调用 LLM -> 带引用回答
    evaluate   recall@k 与答案准确率评测

说明: 本包内模块之间使用相对导入, 请通过项目根目录的 ``main.py`` 调用,
不要直接执行 ``python src/xxx.py``。
"""
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


__all__ = [
    "config",
    "loaders",
    "chunking",
    "embedding",
    "index",
    "rag",
    "evaluate",
]

__version__ = "0.1.0"

# -*- coding: utf-8 -*-
"""多模态文档问答系统 (RAG) —— 命令行入口。

三个子命令::

    python main.py ingest --pdf data/xxx.pdf --out index_dir [--with-images]
    python main.py ask    --index index_dir "你的问题" [-k 5] [--show-context]
    python main.py eval   --index index_dir --eval-set data/eval_set.jsonl [-k 5] [--judge keyword|llm|both]

Stage 1 (主线, 不带 --with-images): 纯文本 RAG
    PDF -> 抽文本 -> 切块 -> bge 向量 -> FAISS -> 混合检索 -> LLM 带引用回答

Stage 2 (可选, 加 --with-images): 跨模态检索
    额外用 CLIP 把 PDF 内嵌图片编码成向量, 建一个并行的图像索引,
    检索时文本索引和图像索引都查, 再用 RRF 合并结果。

关于模型下载: 首次运行会自动下载 bge-small-zh-v1.5 (约 100MB),
开启 --with-images 时还会下载 CLIP (约 600MB)。缓存目录已固定在项目内的
hf_cache/, 不会污染用户主目录。
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


import argparse
import sys
from pathlib import Path

# 保证从任何目录执行 `python main.py` 都能 import 到 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import chunking, config, embedding, evaluate, loaders, rag  # noqa: E402
from src.index import RagIndex  # noqa: E402

EXAMPLES = """\
使用示例:
  # 第一步: 建立索引 (纯文本 RAG, Stage 1)
  python main.py ingest --pdf data/your_doc.pdf --out index_dir

  # 如果 PDF 里有关键图表, 加上 --with-images (Stage 2, 需要下载 CLIP)
  python main.py ingest --pdf data/your_doc.pdf --out index_dir --with-images

  # 消融实验: 换切块大小重新建索引 (注意输出到不同目录, 不要覆盖)
  python main.py ingest --pdf data/your_doc.pdf --out index_cs256 --chunk-size 256

  # 第二步: 提问
  python main.py ask --index index_dir "这篇文档的核心结论是什么?"
  python main.py ask --index index_dir "图 3 说明了什么?" --with-images --show-context

  # 第三步: 评测 (需要 data/eval_set.jsonl)
  python main.py eval --index index_dir --eval-set data/eval_set.jsonl -k 5
  python main.py eval --index index_dir --eval-set data/eval_set.jsonl --judge llm
  python main.py eval --index index_dir --eval-set data/eval_set.jsonl --no-llm

注意: 需要联网下载向量模型; 调用 ask / eval 需要先在 .env 里配置 LLM_API_KEY。
"""


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------


def _add_retrieval_options(parser: argparse.ArgumentParser, with_images_default: bool) -> None:
    """ask 和 eval 共用的检索选项 (也是消融实验的开关)。"""
    parser.add_argument("-k", "--top-k", type=int, default=config.DEFAULT_TOP_K,
                        help="检索返回的 chunk 数, 默认 {}".format(config.DEFAULT_TOP_K))
    parser.add_argument("--no-bm25", action="store_true",
                        help="关闭 BM25 关键词召回, 只用向量检索 (用于消融实验)")
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--with-images", dest="with_images", action="store_true",
                             default=with_images_default,
                             help="启用 CLIP 跨模态图像检索 (索引需在 ingest 时开启过)")
    image_group.add_argument("--no-images", dest="with_images", action="store_false",
                             help="关闭图像检索")
    parser.add_argument("--rrf-k", type=int, default=config.DEFAULT_RRF_K,
                        help="RRF 融合的平滑常数, 默认 {}".format(config.DEFAULT_RRF_K))
    parser.add_argument("--candidate-n", type=int, default=config.DEFAULT_CANDIDATE_N,
                        help="每路召回各取多少条候选再融合, 默认 {}".format(config.DEFAULT_CANDIDATE_N))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="多模态文档问答系统 (RAG): PDF 解析 -> 切块 -> 混合检索 -> 带引用的回答",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{ingest,ask,eval}")

    # ---------------- ingest ----------------
    p_ingest = subparsers.add_parser(
        "ingest", help="解析 PDF 并建立索引 (向量 + BM25, 可选图像索引)",
        description="解析 PDF, 切块, 向量化, 建立 FAISS + BM25 索引。",
    )
    p_ingest.add_argument("--pdf", required=True, help="要处理的 PDF 文件路径")
    p_ingest.add_argument("--out", required=True, help="索引输出目录 (不存在会自动创建)")
    p_ingest.add_argument("--with-images", action="store_true",
                          help="同时抽取 PDF 内嵌图片并用 CLIP 建立图像索引 (Stage 2, 首次会下载约 600MB 模型)")
    p_ingest.add_argument("--chunk-size", type=int, default=config.DEFAULT_CHUNK_SIZE,
                          help="切块的最大字符数, 默认 {}".format(config.DEFAULT_CHUNK_SIZE))
    p_ingest.add_argument("--chunk-overlap", type=int, default=config.DEFAULT_CHUNK_OVERLAP,
                          help="相邻块的重叠字符数, 默认 {}".format(config.DEFAULT_CHUNK_OVERLAP))
    p_ingest.add_argument("--min-image-size", type=int, default=config.DEFAULT_MIN_IMAGE_SIZE,
                          help="图片最小边长 (像素), 小于该值的图当作图标忽略, 默认 {}".format(config.DEFAULT_MIN_IMAGE_SIZE))
    p_ingest.add_argument("--text-model", default=config.DEFAULT_TEXT_MODEL,
                          help="文本向量模型, 默认 {}".format(config.DEFAULT_TEXT_MODEL))
    p_ingest.add_argument("--clip-model", default=config.DEFAULT_CLIP_MODEL,
                          help="图像向量模型, 默认 {}".format(config.DEFAULT_CLIP_MODEL))
    p_ingest.add_argument("--device", default=None, choices=["cpu", "cuda"],
                          help="强制指定推理设备, 默认自动选择")
    p_ingest.add_argument("--batch-size", type=int, default=32, help="文本向量编码的批大小, 默认 32")

    # ---------------- ask ----------------
    p_ask = subparsers.add_parser(
        "ask", help="对已有索引提问, 输出带引用来源的回答",
        description="检索相关片段, 组装 Prompt 调用大模型, 输出带页码引用的回答。",
    )
    p_ask.add_argument("--index", required=True, help="ingest 生成的索引目录")
    p_ask.add_argument("question", nargs="+", help="你的问题 (建议用引号包起来)")
    _add_retrieval_options(p_ask, with_images_default=False)
    p_ask.add_argument("--show-context", action="store_true",
                       help="把送给大模型的完整上下文和 Prompt 打印出来 (排查问题用)")
    p_ask.add_argument("--temperature", type=float, default=None,
                       help="生成温度, 默认读 .env 里的 LLM_TEMPERATURE (0.2)")

    # ---------------- eval ----------------
    p_eval = subparsers.add_parser(
        "eval", help="在评测集上计算 recall@k 与答案准确率, 输出 markdown 结果表",
        description="读取 JSONL 评测集, 计算 recall@k / MRR@k / 答案准确率。",
    )
    p_eval.add_argument("--index", required=True, help="ingest 生成的索引目录")
    p_eval.add_argument("--eval-set", required=True, help="评测集 JSONL 文件路径")
    _add_retrieval_options(p_eval, with_images_default=False)
    p_eval.add_argument("--judge", default="keyword", choices=["keyword", "llm", "both"],
                        help="答案准确率的评测方式: keyword=关键词命中率(默认, 便宜), llm=大模型裁判, both=两者都算")
    p_eval.add_argument("--no-llm", action="store_true",
                        help="完全不调用大模型, 只用检索到的原文算关键词命中率 (相当于检索上界)")
    p_eval.add_argument("--details", action="store_true", help="在报告里附上逐题明细表")
    p_eval.add_argument("--out-md", default=None,
                        help="把 markdown 结果表同时写入该文件, 方便贴进 README")

    return parser


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    config.title("阶段一: 解析 PDF 并建立索引")
    config.info("模型缓存目录: {}".format(config.apply_hf_env()))
    config.info("向量模型首次使用会自动下载 (bge-small-zh-v1.5 约 100MB), 请保持网络畅通。")
    if args.with_images:
        config.info("已开启 --with-images, 还会下载 CLIP 模型 (约 600MB) 并把图片编码进索引。")

    out_dir = Path(args.out).expanduser().resolve()
    image_dir = out_dir / "images"

    # 1. 解析 PDF
    config.title("步骤 1/4: 解析 PDF")
    config.info("文件: {}".format(args.pdf))
    if args.with_images:
        config.info("图片输出目录: {}".format(image_dir))
    document = loaders.load_pdf(
        pdf_path=args.pdf,
        image_out_dir=image_dir,
        extract_images=args.with_images,
        min_image_size=args.min_image_size,
    )
    config.success(document.summary())

    # 2. 切块
    config.title("步骤 2/4: 文本切块")
    config.info(
        "切块参数: chunk_size={} chunk_overlap={}".format(args.chunk_size, args.chunk_overlap)
    )
    text_chunks = chunking.build_chunks(
        pages=document.pages,
        doc_name=document.doc_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not text_chunks:
        raise config.ConfigError(
            "没有从 PDF 中切出任何文本块。\n"
            "最常见的原因: 这是一份扫描版 PDF (整页都是图片, 没有文字层)。\n"
            "解决办法: 先用 OCR 工具 (OCRmyPDF / PaddleOCR / Adobe 导出可搜索 PDF) "
            "生成带文字层的 PDF, 再重新运行 ingest。\n"
            "也可以只跑图像检索: python main.py ingest --pdf <文件> --out {} --with-images".format(out_dir.name)
        )
    config.success(
        "共切出 {} 个文本块, 平均每块约 {:.0f} 字".format(
            len(text_chunks), sum(len(c.text) for c in text_chunks) / float(len(text_chunks))
        )
    )

    image_chunks = chunking.build_image_chunks(document.images, document.doc_name)
    if args.with_images:
        config.success("共抽到 {} 张有效图片".format(len(image_chunks)))

    all_chunks = list(text_chunks) + list(image_chunks)

    # 3. 文本向量化
    config.title("步骤 3/4: 文本向量化 (bge-small-zh-v1.5)")
    text_embedder = embedding.TextEmbedder(model_name=args.text_model, device=args.device)
    text_vectors = text_embedder.encode_documents(
        [c.text for c in text_chunks], batch_size=args.batch_size, show_progress=True
    )
    config.success("文本向量维度: {}, 数量: {}".format(text_vectors.shape[1], text_vectors.shape[0]))

    # 4. 建立索引
    config.title("步骤 4/4: 建立 FAISS + BM25 索引")
    index = RagIndex()
    index.set_chunks(all_chunks)
    index.build_text_index(text_vectors, [c.chunk_id for c in text_chunks])
    config.success("FAISS 文本索引已建立 (IndexFlatIP, 向量已归一化, 内积即余弦相似度)")

    index.build_bm25(text_chunks)
    config.success("BM25 关键词索引已建立 (自己实现, 无第三方依赖; 中文用单字 + 字符二元组切分)")

    image_dim = 0
    if args.with_images and image_chunks:
        config.info("正在用 CLIP 编码 {} 张图片 (首次需要下载模型)...".format(len(image_chunks)))
        image_embedder = embedding.ImageEmbedder(model_name=args.clip_model, device=args.device)
        image_paths = [c.image_path for c in image_chunks if c.image_path]
        if len(image_paths) != len(image_chunks):
            config.warn("有图片缺少本地文件路径, 这些图片不会进入索引。")
            image_chunks = [c for c in image_chunks if c.image_path]
        if image_chunks:
            image_vectors = image_embedder.encode_images(image_paths)
            index.set_chunks(list(text_chunks) + list(image_chunks))
            index.build_image_index(image_vectors, [c.chunk_id for c in image_chunks])
            image_dim = int(image_vectors.shape[1])
            config.success("FAISS 图像索引已建立, 维度 {}".format(image_dim))
    elif args.with_images:
        config.warn("没有抽到符合条件的图片, 跳过图像索引 (可调小 --min-image-size 再试)。")

    index.meta.update(
        {
            "version": config.INDEX_FORMAT_VERSION,
            "with_images": bool(args.with_images and image_dim > 0),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "text_model": args.text_model,
            "text_dim": int(text_vectors.shape[1]),
            "clip_model": args.clip_model if image_dim else "",
            "image_dim": image_dim,
            "source_pdf": str(document.pdf_path),
            "min_image_size": args.min_image_size,
        }
    )

    saved_dir = index.save(out_dir)

    config.title("索引建立完成")
    config.emit(index.summary())
    config.emit("  索引目录: {}".format(saved_dir))
    config.emit("")
    config.emit("下一步可以这样提问:")
    config.emit('    python main.py ask --index {} "你的问题"'.format(args.out))
    return 0


# --------------------------------------------------------------------------
# ask
# --------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> int:
    question = " ".join(args.question).strip()
    if not question:
        raise config.ConfigError('问题不能为空, 用法: python main.py ask --index index_dir "你的问题"')

    config.apply_hf_env()
    index = RagIndex.load(args.index)

    config.title("问题")
    config.emit(question)
    config.info("索引信息: {} 个文本块, {} 张图片".format(index.num_text_chunks, index.num_image_chunks))

    llm = config.load_llm_config()
    llm.require()  # 没有配置 key 时在这里给出清晰的中文指引

    retriever = rag.Retriever(
        index,
        use_bm25=not args.no_bm25,
        use_images=args.with_images,
        rrf_k=args.rrf_k,
        candidate_n=args.candidate_n,
    )
    config.info("检索方式: {}".format(retriever.describe()))

    config.title("检索结果 (共 {} 条)".format(args.top_k))
    hits = retriever.retrieve(question, top_k=args.top_k)
    if not hits:
        config.warn("没有检索到任何相关内容。")
        config.warn("建议: 换一种问法; 或调大 -k; 或确认索引和问题是否属于同一份文档。")
        return 0
    config.emit(rag.format_hits_for_terminal(hits))

    if args.show_context:
        config.title("送给大模型的上下文")
        config.emit(rag.format_context(hits))

    config.title("大模型回答")
    config.info("模型: {} ({})".format(llm.model, llm.base_url))
    messages = [
        {"role": "system", "content": rag.SYSTEM_PROMPT},
        {"role": "user", "content": rag.build_user_prompt(question, hits)},
    ]
    if args.show_context:
        config.title("完整 Prompt (user)")
        config.emit(messages[-1]["content"])
    answer = rag.call_llm(messages, llm=llm, temperature=args.temperature)
    config.emit(answer)

    config.title("命中来源 (可自行核对回答是否有依据)")
    for number, hit in enumerate(hits, start=1):
        config.emit(
            "[{}] {} 第 {} 页   {}".format(number, hit.chunk.doc, hit.chunk.page, hit.source_desc())
        )
    return 0


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    config.apply_hf_env()
    index = RagIndex.load(args.index)

    config.title("评测")
    config.info("索引: {} ({} 个文本块)".format(args.index, index.num_text_chunks))
    items = evaluate.load_eval_set(args.eval_set)
    config.success("评测集读取完成, 共 {} 条样本".format(len(items)))

    report = evaluate.evaluate(
        index=index,
        items=items,
        top_k=args.top_k,
        judge=args.judge,
        use_bm25=not args.no_bm25,
        use_images=args.with_images,
        no_llm=args.no_llm,
        show_details=args.details,
    )

    config.title("评测结果 (markdown, 可直接贴进 README)")
    config.emit(report["markdown"])

    if args.out_md:
        out_path = Path(args.out_md).expanduser().resolve()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(str(report["markdown"]) + "\n")
        except OSError as exc:
            raise config.ConfigError("写入结果文件失败: {}\n原始错误: {}".format(out_path, exc)) from exc
        config.success("结果已写入: {}".format(out_path))

    config.emit("")
    config.info(
        "做消融实验时, 建议每次只改一个变量 (chunk_size / --no-bm25 / --with-images), "
        "把上面的表格贴进 README 的试验记录表。"
    )
    return 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    handlers = {"ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval}
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover
        parser.print_help()
        return 2

    try:
        return handler(args)
    except config.ConfigError as exc:
        config.emit("")
        config.error(str(exc))
        return 2
    except KeyboardInterrupt:
        config.emit("")
        config.warn("已被用户中断 (Ctrl+C)。已建立的索引如果没有保存完, 请重新运行 ingest。")
        return 130
    except Exception:
        config.emit("")
        config.error(
            "出现了未预期的错误, 下面是完整调用栈。\n"
            "请把报错信息连同你运行的命令一起记录下来, 便于定位问题。"
        )
        raise


if __name__ == "__main__":
    sys.exit(main())

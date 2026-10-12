# -*- coding: utf-8 -*-
"""PDF 解析模块 (Stage 1 的入口)。

职责:
    把一份 PDF 拆成两部分:
      1. 每页的纯文本 (交给 chunking 切块)
      2. 每页内嵌的图片 (Stage 2 用 CLIP 编码, 可选)

依赖 PyMuPDF (导入名 ``pymupdf``)。

关于扫描版 PDF: PyMuPDF 只能抽取 PDF 里"文字层"的内容。如果一份 PDF 是
扫描件 (整页就是一张图), 抽出来会是空的。本模块会检测这种情况并给出明确
提示, 让用户先去做 OCR, 而不是丢一堆空 chunk 下去。
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
from pathlib import Path
from typing import List, Optional

from . import config

# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass
class PageContent:
    """单页文本。page 从 1 开始计数, 与 PDF 阅读器显示的页码一致, 便于引用溯源。"""

    page: int
    text: str


@dataclass
class ImageRecord:
    """单张图片。image_path 为 None 表示抽取失败或未开启图片抽取。"""

    page: int
    image_path: Optional[Path]
    width: int
    height: int
    caption: str = ""

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "image_path": str(self.image_path) if self.image_path else "",
            "width": self.width,
            "height": self.height,
            "caption": self.caption,
        }


@dataclass
class LoadedDocument:
    """一份 PDF 的解析结果。"""

    doc_name: str
    pdf_path: Path
    pages: List[PageContent] = field(default_factory=list)
    images: List[ImageRecord] = field(default_factory=list)
    page_count: int = 0

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def pages_with_text(self) -> int:
        return sum(1 for p in self.pages if p.text.strip())

    def summary(self) -> str:
        return "{}: 共 {} 页, 有文字的页 {} 页, 提取正文约 {} 字, 图片 {} 张".format(
            self.doc_name,
            self.page_count,
            self.pages_with_text,
            self.total_chars,
            len(self.images),
        )


# --------------------------------------------------------------------------
# 依赖导入 (延迟导入以便给出中文报错)
# --------------------------------------------------------------------------


def _import_pymupdf():
    """导入 PyMuPDF, 失败时给出可照做的中文提示。

    注意: 统一使用新版包名 ``import pymupdf``。旧的 ``import fitz`` 别名已经废弃,
    在 PyMuPDF 1.24+ 上会打印 deprecation 警告, 所以本项目不再使用该别名。
    """
    try:
        import pymupdf

        return pymupdf
    except ImportError as exc:
        raise config.ConfigError(
            "缺少 PyMuPDF 依赖, 无法解析 PDF。\n"
            "请用项目虚拟环境安装依赖:\n"
            "    .venv\\Scripts\\python.exe -m pip install -r project_rag/requirements-rag.txt\n"
            "原始错误: {}".format(exc)
        ) from exc


# --------------------------------------------------------------------------
# 文本清洗
# --------------------------------------------------------------------------

#: 中日韩统一表意文字 + 常见中文标点, 用于判断是否需要无空格拼接
_CJK_RE = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef\u3040-\u30ff]")


def _is_cjk(ch: str) -> bool:
    return bool(ch) and bool(_CJK_RE.match(ch))


def _join_lines(lines: List[str]) -> str:
    """把 PDF 的硬换行还原成正常段落。

    PDF 里每一行都是换行, 直接拼起来会把句子切碎, 严重伤害检索质量。
    规则:
      - 空行 -> 段落分隔
      - 前后都是中日韩字符 -> 直接拼接 (中文没有词间空格)
      - 英文行尾连字符 -> 去掉连字符拼接
      - 其它情况 -> 用一个空格拼接
    """
    out: List[str] = []
    buf = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buf:
                out.append(buf)
                buf = ""
            out.append("")
            continue
        if not buf:
            buf = line
            continue
        if _is_cjk(buf[-1]) or _is_cjk(line[0]):
            buf += line
        elif buf.endswith("-") and len(buf) > 1 and line[0].islower():
            buf = buf[:-1] + line
        else:
            buf += " " + line
    if buf:
        out.append(buf)
    return "\n".join(out)


def clean_page_text(raw: str) -> str:
    """清洗单页原始文本: 统一换行符、去掉软连字符、还原硬换行、压缩空白。"""
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")  # 软连字符
    text = text.replace("\ufeff", "")
    text = _join_lines(text.split("\n"))
    # 压缩连续空格与 3 个以上换行
    text = re.sub(r"[ \t\u3000]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# 图片抽取
# --------------------------------------------------------------------------


def _extract_images_from_page(
    doc,
    page,
    page_no: int,
    doc_stem: str,
    image_out_dir: Path,
    min_image_size: int,
    seen_xrefs: set,
) -> List[ImageRecord]:
    """抽取单页内的图片。

    说明: PDF 中同一张图可能被多页引用 (同一个 xref), 这里用 seen_xrefs 去重,
    避免同一张图被编码多次。同时过滤掉图标、logo 之类的小图。
    """
    records: List[ImageRecord] = []
    try:
        image_list = page.get_images(full=True)
    except Exception as exc:
        config.warn("第 {} 页读取图片列表失败, 已跳过: {}".format(page_no, exc))
        return records

    for img_info in image_list:
        try:
            xref = int(img_info[0])
        except (IndexError, TypeError, ValueError):
            continue
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            config.warn("第 {} 页 xref={} 图片解码失败, 已跳过: {}".format(page_no, xref, exc))
            continue
        if not base_image:
            continue

        width = int(base_image.get("width", 0) or 0)
        height = int(base_image.get("height", 0) or 0)
        if width < min_image_size or height < min_image_size:
            # 小图基本是 logo / 项目符号, 过滤掉
            continue

        ext = str(base_image.get("ext", "png") or "png").lower()
        if ext in {"jpx", "jp2", "jbig2", "jb2"}:
            # 这些格式 Pillow 不一定能读, 统一跳过并提示
            config.warn("第 {} 页有 {} 格式图片, 当前实现不支持, 已跳过。".format(page_no, ext))
            continue
        if ext == "jpeg":
            ext = "jpg"

        try:
            image_out_dir.mkdir(parents=True, exist_ok=True)
            file_name = "{}_p{}_x{}.{}".format(doc_stem, page_no, xref, ext)
            file_path = image_out_dir / file_name
            file_path.write_bytes(base_image["image"])
        except Exception as exc:
            config.warn("第 {} 页图片保存失败, 已跳过: {}".format(page_no, exc))
            continue

        caption = "[图片] 出自 {} 第 {} 页, 尺寸 {}x{}".format(doc_stem, page_no, width, height)
        records.append(
            ImageRecord(
                page=page_no,
                image_path=file_path,
                width=width,
                height=height,
                caption=caption,
            )
        )
    return records


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def load_pdf(
    pdf_path,
    image_out_dir: Optional[Path] = None,
    extract_images: bool = False,
    min_image_size: int = config.DEFAULT_MIN_IMAGE_SIZE,
) -> LoadedDocument:
    """解析一份 PDF。

    参数:
        pdf_path:       PDF 文件路径
        image_out_dir:  图片输出目录 (extract_images=True 时必填)
        extract_images: 是否抽取图片 (Stage 2 才需要, 默认关闭以节省时间)
        min_image_size: 图片最小边长 (像素), 小于该值的图会被当作图标过滤

    返回:
        LoadedDocument

    异常:
        config.ConfigError: 文件不存在 / 不是 PDF / 依赖缺失, 均为中文提示
    """
    pymupdf = _import_pymupdf()

    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise config.ConfigError(
            "找不到 PDF 文件: {}\n请检查路径是否正确 (可以用绝对路径, 例如 D:\\docs\\a.pdf)。".format(path)
        )
    if path.is_dir():
        raise config.ConfigError("给的是目录而不是 PDF 文件: {}\n请指定具体某个 .pdf 文件。".format(path))
    if path.suffix.lower() not in {".pdf"}:
        config.warn("文件后缀不是 .pdf ({}), 仍尝试按 PDF 解析。".format(path.suffix))

    if extract_images and image_out_dir is None:
        raise config.ConfigError("开启了图片抽取, 但没有指定图片输出目录 (image_out_dir)。")

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise config.ConfigError(
            "打开 PDF 失败: {}\n可能原因: 文件损坏、被加密、或不是真正的 PDF。\n原始错误: {}".format(
                path, exc
            )
        ) from exc

    doc_stem = path.stem
    pages: List[PageContent] = []
    images: List[ImageRecord] = []
    seen_xrefs: set = set()
    page_count = 0

    try:
        page_count = doc.page_count
        if page_count == 0:
            raise config.ConfigError("这份 PDF 的页数是 0, 文件可能是空的或损坏: {}".format(path))

        for page_index in range(page_count):
            page_no = page_index + 1  # 对外统一用 1 起始页码
            try:
                page = doc.load_page(page_index)
            except Exception as exc:
                config.warn("第 {} 页加载失败, 已跳过: {}".format(page_no, exc))
                pages.append(PageContent(page=page_no, text=""))
                continue

            try:
                raw_text = page.get_text("text") or ""
            except Exception as exc:
                config.warn("第 {} 页文字抽取失败, 已跳过: {}".format(page_no, exc))
                raw_text = ""
            pages.append(PageContent(page=page_no, text=clean_page_text(raw_text)))

            if extract_images:
                images.extend(
                    _extract_images_from_page(
                        doc=doc,
                        page=page,
                        page_no=page_no,
                        doc_stem=doc_stem,
                        image_out_dir=Path(image_out_dir),
                        min_image_size=min_image_size,
                        seen_xrefs=seen_xrefs,
                    )
                )
    finally:
        try:
            doc.close()
        except Exception:
            pass

    loaded = LoadedDocument(
        doc_name=path.name,
        pdf_path=path,
        pages=pages,
        images=images,
        page_count=page_count,
    )

    # 扫描版检测: 给了页数但几乎没有文字
    if loaded.total_chars < 50:
        config.warn(
            "这份 PDF 几乎抽不到文字 ({} 字 / {} 页), 很可能是扫描版或图片型 PDF。\n"
            "        本项目的文本 RAG 流水线依赖 PDF 的文字层, 请先用 OCR 工具\n"
            "        (例如 OCRmyPDF / PaddleOCR / Adobe 导出可搜索 PDF) 处理后再放进来。".format(
                loaded.total_chars, page_count
            )
        )
    elif loaded.pages_with_text < page_count:
        missing = page_count - loaded.pages_with_text
        config.warn("有 {} 页没有抽到文字 (可能是整页插图或扫描页), 这些页不会产生文本 chunk。".format(missing))

    return loaded


__all__ = [
    "PageContent",
    "ImageRecord",
    "LoadedDocument",
    "clean_page_text",
    "load_pdf",
]

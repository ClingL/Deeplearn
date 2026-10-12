# -*- coding: utf-8 -*-
"""全局配置模块。

本模块负责三件事:

1. 把 HuggingFace / PyTorch 的模型缓存目录固定到项目内的 ``hf_cache``,
   避免首次运行时把几百 MB 的模型下载到用户主目录, 污染系统盘。
2. 读取项目根目录下的 ``.env``, 拿到 LLM 的 api_key / base_url / model。
   本项目统一使用 OpenAI 兼容接口, 因此 DeepSeek / 智谱 / 阿里百炼 / OpenAI
   都只需要改这三个环境变量。
3. 提供统一的终端输出函数 (优先使用 rich, 未安装则退化为普通 print)
   以及集中管理的默认超参数。

重要: 本模块必须在导入 sentence-transformers / torch 之前被导入,
否则上面第 1 条的缓存目录设置不会生效。
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


import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------
# 路径常量
# --------------------------------------------------------------------------

#: 项目根目录 (即 project_rag/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 数据目录 (放 PDF、评测集)
DATA_DIR = PROJECT_ROOT / "data"

#: 默认索引输出目录
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index_dir"

#: 模型缓存目录, 固定在项目内
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

#: 环境变量文件
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"


# --------------------------------------------------------------------------
# 默认超参数 (集中管理, 便于做消融实验)
# --------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_N = 20
DEFAULT_MIN_IMAGE_SIZE = 64
DEFAULT_TEMPERATURE = 0.2

DEFAULT_TEXT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

#: bge 系列模型官方建议: 检索时为"查询"加指令前缀, 文档侧不加。
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

#: 当前索引格式版本, 将来结构变化时用于给出提示
INDEX_FORMAT_VERSION = 1


# --------------------------------------------------------------------------
# 统一异常
# --------------------------------------------------------------------------


class ConfigError(RuntimeError):
    """配置或环境缺失时抛出, 异常信息统一为可直接操作的中文提示。"""


# --------------------------------------------------------------------------
# 终端输出
# --------------------------------------------------------------------------

_CONSOLE = None
try:  # rich 是可选依赖, 没装也能跑
    from rich.console import Console as _RichConsole

    _CONSOLE = _RichConsole()
except Exception:  # pragma: no cover - 仅在没装 rich 时触发
    _CONSOLE = None


def _safe_streams() -> None:
    """避免在 GBK 控制台打印中文时抛出 UnicodeEncodeError 直接崩掉。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


_safe_streams()


def emit(message: str, style: Optional[str] = None) -> None:
    """输出一行文本。

    统一关闭 rich 的 markup 与 highlight, 因为文档原文里可能包含
    ``[xxx]`` 这种方括号内容, 会被 rich 当成样式标记吞掉。
    """
    if _CONSOLE is not None:
        _CONSOLE.print(message, style=style, markup=False, highlight=False)
    else:  # pragma: no cover
        print(message)


def info(message: str) -> None:
    emit("[提示] " + message)


def warn(message: str) -> None:
    emit("[警告] " + message, style="yellow")


def error(message: str) -> None:
    emit("[错误] " + message, style="bold red")


def success(message: str) -> None:
    emit("[完成] " + message, style="green")


def title(message: str) -> None:
    emit("")
    emit("=" * 68, style="cyan")
    emit(message, style="bold cyan")
    emit("=" * 68, style="cyan")


def rule() -> None:
    emit("-" * 68)


# --------------------------------------------------------------------------
# 模型缓存目录
# --------------------------------------------------------------------------


def apply_hf_env() -> Path:
    """把 HuggingFace / torch 的缓存目录全部指向项目内的 ``hf_cache``。

    返回缓存目录路径。此函数应在任何模型库被导入前调用。
    """
    cache_dir = HF_CACHE_DIR
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warn("无法创建缓存目录 {}: {}".format(cache_dir, exc))

    cache_str = str(cache_dir)
    os.environ["HF_HOME"] = cache_str
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir / "sentence_transformers")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    # 关闭软链接警告 (Windows 上没开开发者模式时会有无害告警刷屏)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return cache_dir


# --------------------------------------------------------------------------
# .env 读取
# --------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """大模型接口配置。"""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = 60.0

    @property
    def configured(self) -> bool:
        """是否已经填好了可用的 api_key。"""
        key = (self.api_key or "").strip()
        return bool(key) and key.lower() not in {"sk-替换成你自己的key", "your-api-key"}

    def describe(self) -> str:
        """用于打印的简短描述, 不泄露 key 的内容。"""
        masked = "未设置"
        if self.configured:
            key = self.api_key.strip()
            masked = key[:6] + "****" + key[-4:] if len(key) > 12 else "已设置"
        return "base_url={} model={} api_key={}".format(self.base_url, self.model, masked)

    def require(self) -> None:
        """没有配置好 key 时给出可直接照做的中文报错。"""
        if self.configured:
            return
        raise ConfigError(build_missing_key_message())


def build_missing_key_message() -> str:
    """拼装"缺少 API Key"的中文引导信息。"""
    lines = [
        "未检测到可用的 LLM_API_KEY, 无法调用大模型接口。",
        "",
        "请按下面三步配置 (只需做一次):",
        "  1. 进入项目目录 project_rag, 把示例配置复制成真正的配置文件:",
        "       Windows     :  copy .env.example .env",
        "       macOS/Linux :  cp .env.example .env",
        "  2. 用编辑器打开 .env, 填写三项:",
        "       LLM_API_KEY  = 你自己的 API Key",
        "       LLM_BASE_URL = 服务商的 OpenAI 兼容地址 (DeepSeek 用 https://api.deepseek.com/v1)",
        "       LLM_MODEL    = 模型名 (DeepSeek 用 deepseek-chat)",
        "     文件里已经写好了 DeepSeek / 智谱 / 阿里百炼 / OpenAI / Ollama 的示例值。",
        "  3. 保存后重新运行命令。",
        "",
        "如果只是想做纯检索评测 (不调用大模型), 可以用:",
        "    python main.py eval --index index_dir --eval-set data/eval_set.jsonl --judge keyword",
        "",
        "配置文件位置: {}".format(ENV_FILE),
        "示例文件位置: {}".format(ENV_EXAMPLE_FILE),
    ]
    return "\n".join(lines)


def load_env_file() -> None:
    """读取项目根目录的 .env (不存在时静默跳过, 让后续的报错来提示用户)。"""
    if not ENV_FILE.exists():
        warn(".env 不存在, 将只使用系统环境变量。可执行 copy .env.example .env 后填写。")
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ConfigError(
            "缺少 python-dotenv 依赖, 无法读取 .env 文件。\n"
            "请用项目虚拟环境安装依赖 (不要用系统 Python):\n"
            "    .venv\\Scripts\\python.exe -m pip install -r project_rag/requirements-rag.txt\n"
            "原始错误: {}".format(exc)
        ) from exc
    load_dotenv(dotenv_path=str(ENV_FILE), override=False)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        warn("环境变量 {} 不是合法数字 ({}) , 已忽略并使用默认值 {}".format(name, raw, default))
        return default


def load_llm_config() -> LLMConfig:
    """读取 .env + 系统环境变量, 组装 LLMConfig。"""
    load_env_file()
    return LLMConfig(
        api_key=os.environ.get("LLM_API_KEY", "").strip(),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").strip()
        or "https://api.deepseek.com/v1",
        model=os.environ.get("LLM_MODEL", "deepseek-chat").strip() or "deepseek-chat",
        temperature=_env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        timeout=_env_float("LLM_TIMEOUT", 60.0),
    )


def build_openai_client(llm: LLMConfig):
    """按 OpenAI 兼容协议创建客户端。缺少 key 时抛出中文 ConfigError。

    说明: 这里使用 openai 新版官方 SDK 的写法::

        from openai import OpenAI
        client = OpenAI(api_key=..., base_url=...)
        resp = client.chat.completions.create(model=..., messages=[...])
        text = resp.choices[0].message.content

    不使用任何已移除的旧接口 (如 openai.ChatCompletion.create)。
    """
    llm.require()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ConfigError(
            "缺少 openai 依赖, 无法调用大模型。\n"
            "请用项目虚拟环境安装依赖:\n"
            "    .venv\\Scripts\\python.exe -m pip install -r project_rag/requirements-rag.txt\n"
            "原始错误: {}".format(exc)
        ) from exc
    try:
        # timeout 是新版 SDK 支持的参数; 万一某个版本不认, 这里退化为不带 timeout 再建一次
        try:
            return OpenAI(api_key=llm.api_key, base_url=llm.base_url, timeout=llm.timeout)
        except TypeError:
            return OpenAI(api_key=llm.api_key, base_url=llm.base_url)
    except Exception as exc:  # 例如 base_url 写错
        raise ConfigError(
            "创建 OpenAI 客户端失败, 请检查 .env 里的 LLM_BASE_URL 是否写对。\n"
            "当前值: {}\n原始错误: {}".format(llm.base_url, exc)
        ) from exc


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "DEFAULT_INDEX_DIR",
    "HF_CACHE_DIR",
    "ENV_FILE",
    "ENV_EXAMPLE_FILE",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_TOP_K",
    "DEFAULT_RRF_K",
    "DEFAULT_CANDIDATE_N",
    "DEFAULT_MIN_IMAGE_SIZE",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TEXT_MODEL",
    "DEFAULT_CLIP_MODEL",
    "BGE_QUERY_INSTRUCTION",
    "INDEX_FORMAT_VERSION",
    "ConfigError",
    "LLMConfig",
    "apply_hf_env",
    "load_llm_config",
    "build_openai_client",
    "build_missing_key_message",
    "emit",
    "info",
    "warn",
    "error",
    "success",
    "title",
    "rule",
]

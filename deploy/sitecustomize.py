# -*- coding: utf-8 -*-
"""把模型缓存重定向到本项目目录内（由 Python 启动时自动导入）。

由 setup.sh / 本地安装脚本复制到虚拟环境的 site-packages 下。
仓库里的这份是**唯一真实来源**，改这里，别直接改 venv 里的副本。

为什么需要它
------------
PyTorch 默认把预训练权重缓存到 `~/.cache/torch`，HuggingFace 模型缓存在
`~/.cache/huggingface`，matplotlib 字体缓存在 `~/.matplotlib`。

在受限环境（比如 DSH 的 workspace-write 沙箱）里，往工作区之外写文件会被拒绝：

    PermissionError: [WinError 5] 拒绝访问。: 'C:\\Users\\xxx/.cache\\torch'

把缓存放在项目内还顺带解决了服务器上的另一个常见问题：
配额小、或者 home 目录被清空导致重下模型。

缓存位置
--------
    <项目根>/.cache/torch          预训练权重（ResNet / EfficientNet / MobileNet 等）
    <项目根>/.cache/huggingface    HuggingFace 模型（bge 向量模型 / CLIP 等）
    <项目根>/.cache/matplotlib     matplotlib 字体缓存

这些目录可以随时整个删掉，下次运行会自动重新下载/重建。

注意：本文件必须同时兼容 Windows 和 Linux 的 venv 目录结构
    Windows:  <root>/.venv/Lib/site-packages/sitecustomize.py
    Linux:    <root>/.venv/lib/python3.12/site-packages/sitecustomize.py
两者的层级数不同，所以下面用「向上找名为 .venv 的目录」来定位项目根，
而不是写死 parents[N]。
"""
import os
from pathlib import Path


def _find_project_root():
    """从本文件位置向上找 .venv 目录，其父目录就是项目根。"""
    try:
        here = Path(__file__).resolve()
    except NameError:  # 极端情况下 __file__ 不存在
        return None

    for parent in here.parents:
        if parent.name in (".venv", "venv"):
            return parent.parent

    # 没找到 .venv（比如被手动复制到别处），再退一步：
    # 往上找一个同时含 README.md 的目录
    for parent in here.parents:
        if (parent / "README.md").is_file() and (parent / "lessons").is_dir():
            return parent

    return None


_ROOT = _find_project_root()

if _ROOT is not None:
    _CACHE = _ROOT / ".cache"
    # setdefault：如果你自己已经设过环境变量，就尊重你的设置
    os.environ.setdefault("TORCH_HOME", str(_CACHE / "torch"))
    os.environ.setdefault("HF_HOME", str(_CACHE / "huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
    # 国内下载 HuggingFace 模型慢的话，取消下面这行的注释走镜像站
    # os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

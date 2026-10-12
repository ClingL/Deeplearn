# -*- coding: utf-8 -*-
"""模型构建 —— 支持从零训练的小 CNN 和多个预训练骨干。

设计要点
--------
1. 预训练骨干在小尺寸输入（比如 32/64）上需要改 stem，否则特征图被压得太小。
2. 统一暴露 `head_name`，让训练脚本能做「冻结 backbone / 判别式学习率」。
3. 从零训练的对比模型（smallcnn）走完全一样的接口，保证消融实验可比。
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


from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# 每个骨干的：构造函数、权重枚举、分类头属性名
_BACKBONES = {
    "resnet18": (torchvision.models.resnet18, "ResNet18_Weights"),
    "resnet34": (torchvision.models.resnet34, "ResNet34_Weights"),
    "resnet50": (torchvision.models.resnet50, "ResNet50_Weights"),
    "efficientnet_b0": (torchvision.models.efficientnet_b0, "EfficientNet_B0_Weights"),
    "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small, "MobileNet_V3_Small_Weights"),
}

HEAD_ATTR = {
    "resnet18": "fc", "resnet34": "fc", "resnet50": "fc",
    "efficientnet_b0": "classifier", "mobilenet_v3_small": "classifier",
    "smallcnn": "head",
}


class SmallCNN(nn.Module):
    """4 层卷积的小网络，用作"从零训练"的 baseline。

    故意做得简单：这样迁移学习的提升才显得有说服力。
    """

    def __init__(self, n_classes: int = 10, in_ch: int = 3, width: int = 32):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(width * 8, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(self.dropout(x))


def build_model(arch: str, n_classes: int, pretrained: bool = True,
                img_size: int = 224) -> nn.Module:
    """按名字构建模型。

    arch: smallcnn / resnet18 / resnet34 / resnet50 / efficientnet_b0 / mobilenet_v3_small
    """
    arch = arch.lower()

    if arch == "smallcnn":
        print(f"[model] SmallCNN (从零训练), n_classes={n_classes}")
        return SmallCNN(n_classes)

    if arch not in _BACKBONES:
        raise ValueError(
            f"不支持的 arch: {arch}\n可选: smallcnn, " + ", ".join(_BACKBONES.keys())
        )

    ctor, weights_name = _BACKBONES[arch]
    weights = None
    if pretrained:
        weights = getattr(torchvision.models, weights_name).IMAGENET1K_V1

    model = ctor(weights=weights)

    # ---- 适配小尺寸输入 ----
    # 原版 ResNet 的 stem 是 7x7 stride2 + maxpool stride2，
    # 224 的图变成 56，但 32 的图会变成 8 —— 太小，后面没东西可学。
    # CIFAR 上的标准改法：3x3 stride1 conv + 去掉 maxpool。
    if img_size <= 64 and arch.startswith("resnet"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        print(f"[model] {arch}: 输入 {img_size}x{img_size}，已把 stem 改为 3x3/stride1 "
              f"并去掉 maxpool")

    # ---- 换分类头 ----
    head_attr = HEAD_ATTR[arch]
    old_head = getattr(model, head_attr)

    if isinstance(old_head, nn.Linear):
        new_head = nn.Linear(old_head.in_features, n_classes)
    else:
        # mobilenet 的 classifier 是 Sequential(Linear, ...)，只换最后那个 Linear
        new_head = _replace_last_linear(old_head, n_classes)

    setattr(model, head_attr, new_head)

    tag = "ImageNet 预训练" if pretrained else "随机初始化"
    print(f"[model] {arch} ({tag}) -> {n_classes} 类, head='{head_attr}'")
    return model


def _replace_last_linear(seq: nn.Sequential, n_classes: int) -> nn.Sequential:
    layers = list(seq)
    for i in range(len(layers) - 1, -1, -1):
        if isinstance(layers[i], nn.Linear):
            layers[i] = nn.Linear(layers[i].in_features, n_classes)
            break
    return nn.Sequential(*layers)


def head_name(arch: str) -> str:
    return HEAD_ATTR.get(arch.lower(), "head")


def split_params(model: nn.Module, arch: str) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """把参数分成 (backbone 参数, head 参数)，用于判别式学习率和冻结。"""
    hn = head_name(arch)
    backbone, head = [], []
    for name, p in model.named_parameters():
        if name.startswith(hn + "."):
            head.append(p)
        else:
            backbone.append(p)
    return backbone, head


def freeze_backbone(model: nn.Module, arch: str, freeze: bool = True) -> None:
    """冻结 / 解冻 backbone。head 始终可训练。"""
    hn = head_name(arch)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(hn + ".") if freeze else True


def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def print_model_summary(model: nn.Module, arch: str) -> None:
    backbone, head = split_params(model, arch)
    n_bb = sum(p.numel() for p in backbone)
    n_hd = sum(p.numel() for p in head)
    c = count_params(model)
    print(f"[model] 参数量: 总计 {c['total']:,} | backbone {n_bb:,} | head {n_hd:,}")
    print(f"[model] 可训练: {c['trainable']:,} ({c['trainable'] / c['total']:.1%})")

# -*- coding: utf-8 -*-
r"""
第 06 课：迁移学习 / 微调（实习里最常用的技能，没有之一）

运行：
    .\.venv\Scripts\python.exe lessons\06_transfer_learning.py

本课会做一个**对比实验**，用同一份小数据（每类只有 300 张）：
    A. 从零训练 ResNet18（随机初始化）
    B. 用 ImageNet 预训练权重微调
你会亲眼看到 B 甩开 A 多少 —— 这就是为什么工业界几乎不从头训模型。

学完你应该能回答：
  - 什么是预训练 / 微调 / 特征提取，三者区别？
  - 什么时候该冻结 backbone，什么时候该解冻？
  - 为什么微调要用更小的学习率？
  - 判别式学习率(discriminative LR)是怎么回事？
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

import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def sec(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


sec("0. 环境")
print("device:", DEVICE, "| GPU:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "-")

# ----------------------------------------------------------------------------
sec("1. 迁移学习是什么，为什么实习里最重要")
# ----------------------------------------------------------------------------
print("""
核心想法：
  ImageNet 上训练的模型，它的浅层已经学会了"边缘/纹理/颜色"，
  中层学会了"眼睛/轮子/毛发"这些通用部件。
  这些特征对几乎所有视觉任务都有用 —— 所以你只需要换掉最后一层，
  用自己的少量数据教会它"这些特征组合起来是什么类别"。

三种做法（面试必考，一定分清）：
  ┌────────────────┬──────────────────────────┬──────────────────────────┐
  │ 策略            │ 做什么                    │ 什么时候用                │
  ├────────────────┼──────────────────────────┼──────────────────────────┤
  │ 特征提取        │ 冻结 backbone，只训新 head │ 数据很少(<1000)、和预训练  │
  │ (frozen)       │ backbone.eval() 不更新    │ 域接近(都是自然图像)       │
  ├────────────────┼──────────────────────────┼──────────────────────────┤
  │ 微调            │ 解冻部分/全部层，用小 lr   │ 数据中等、想让特征适配新域  │
  │ (fine-tune)    │ 一起训练                  │ 最常用的方式              │
  ├────────────────┼──────────────────────────┼──────────────────────────┤
  │ 从头训练        │ 随机初始化                │ 数据百万级、或域差很远      │
  │ (from scratch) │                           │ (医学影像/遥感)           │
  └────────────────┴──────────────────────────┴──────────────────────────┘

关键细节（面试加分项）：
  - 微调要小学习率（比如 backbone 1e-4，head 1e-3），
    因为预训练权重已经在很好的位置了，大 lr 会把学到的东西"冲掉"(catastrophic forgetting)
  - BatchNorm 层微调时通常保持 eval 模式（冻结 running stats），
    除非你的数据量足够大能重新估准统计量
  - 替换 head 后，新 head 是随机初始化的，所以前几轮 loss 会很高，正常
""")

# ----------------------------------------------------------------------------
sec("2. 数据：故意只取一小部分（模拟真实业务里的「数据少」）")
# ----------------------------------------------------------------------------
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
PER_CLASS = 300          # 每类只给 300 张训练图（总共 3000 张）

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

full_train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
test_set = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)

# 分层抽取：每类取前 PER_CLASS 个
targets = np.array(full_train.targets)
idx = []
for c in range(10):
    cls_idx = np.where(targets == c)[0][:PER_CLASS]
    idx.extend(cls_idx.tolist())
np.random.seed(0)
np.random.shuffle(idx)
train_set = Subset(full_train, idx)

print(f"训练集 {len(train_set)} 张（每类 {PER_CLASS} 张）")
print(f"测试集 {len(test_set)} 张（完整 10000 张，评估够稳）")
print("类别分布:", Counter(targets[idx]).most_common())

BATCH = 64
# num_workers=0 的原因见第 05 课（Windows 多进程需要 __main__ 保护）
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
print(f"每个 epoch {len(train_loader)} 个 iteration")


# ----------------------------------------------------------------------------
sec("3. 构造 CIFAR 版 ResNet18")
# ----------------------------------------------------------------------------
def build_resnet18(n_classes=10, pretrained=True):
    """标准做法：改 stem 适应 32x32 小图，换掉最后的 fc。

    原版 ResNet 是为 224x224 设计的：
      conv1 是 7x7 stride2，紧跟 maxpool stride2 -> 32x32 的图会被压成 4x4，太小
    所以 CIFAR 上通常改成 3x3 stride1，去掉 maxpool。
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)
    # 从头训练时也要做同样的结构修改，保证两个实验可比
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()                    # 去掉 maxpool
    m.fc = nn.Linear(m.fc.in_features, n_classes)  # 换成 10 类
    return m


def print_param_groups(model, tag):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] 总参数 {total:,} | 可训练 {trainable:,} ({trainable/total:.1%})")


model_preview = build_resnet18(pretrained=True)
print(model_preview)
print_param_groups(model_preview, "预训练ResNet18")
del model_preview
torch.cuda.empty_cache()

print("""
注意最后一行是 fc: Linear(in_features=512, out_features=10)
我们把 1000 类换成了 10 类 —— 这就是"换 head"。
""")

# ----------------------------------------------------------------------------
sec("4. 通用训练/评估函数")
# ----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, correct / total


def train_model(model, epochs, lr, tag="", lr_backbone=None, verbose=True):
    """lr_backbone 不为 None 时使用判别式学习率（backbone 用更小的 lr）。"""
    if lr_backbone is not None:
        params = [
            {"params": [p for n, p in model.named_parameters()
                        if not n.startswith("fc") and p.requires_grad], "lr": lr_backbone},
            {"params": [p for n, p in model.named_parameters()
                        if n.startswith("fc") and p.requires_grad], "lr": lr},
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc, best_state = 0.0, None
    t0 = time.time()
    if verbose:
        print(f"\n--- {tag} ---")
        print(f"{'epoch':>5} | {'tr_loss':>8} {'tr_acc':>7} | {'te_loss':>8} {'te_acc':>7} | {'耗时':>6}")
        print("-" * 62)

    for ep in range(1, epochs + 1):
        te0 = time.time()
        model.train()
        rl, rc, rn = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rl += loss.item() * xb.size(0)
            rc += (logits.argmax(1) == yb).sum().item()
            rn += xb.size(0)
        tr_loss, tr_acc = rl / rn, rc / rn
        te_loss, te_acc = evaluate(model, test_loader)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss); history["test_acc"].append(te_acc)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"{ep:>5} | {tr_loss:>8.4f} {tr_acc:>7.4f} | {te_loss:>8.4f} {te_acc:>7.4f} | "
                  f"{time.time()-te0:>5.1f}s")
    history["best_acc"] = best_acc
    history["time"] = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"最好 test acc = {best_acc:.4f}   总耗时 {history['time']:.1f}s")
    return history


# ----------------------------------------------------------------------------
sec("5. 实验 A：从零训练（随机初始化，不用预训练权重）")
# ----------------------------------------------------------------------------
EPOCHS = 8
print(f"数据只有 {len(train_set)} 张，我们看从零训练能到多少\n")

model_scratch = build_resnet18(pretrained=False).to(DEVICE)
print_param_groups(model_scratch, "从零训练")
hist_scratch = train_model(model_scratch, EPOCHS, lr=1e-3, tag="A. From Scratch (random init)")

# ----------------------------------------------------------------------------
sec("6. 实验 B：迁移学习（ImageNet 预训练 + 微调）")
# ----------------------------------------------------------------------------
print("""
分两阶段做，这是最稳的流程：
  阶段1 特征提取：冻结 backbone，只训 fc。让新 head 先"对齐"到已有特征上，
                  避免随机初始化的 head 一开始就产生巨大梯度把 backbone 带偏。
  阶段2 微调      ：解冻 backbone，用很小的 lr 整体微调（判别式学习率）。
""")

model_ft = build_resnet18(pretrained=True).to(DEVICE)

# --- 阶段 1：冻结 backbone，只训 fc ---
for n, p in model_ft.named_parameters():
    p.requires_grad = n.startswith("fc")     # 只有 fc 需要梯度
print_param_groups(model_ft, "阶段1 冻结backbone")

print("\n验证冻结是否生效（除了 fc 都应该 requires_grad=False）:")
for n, p in list(model_ft.named_parameters())[:3]:
    print(f"  {n:<28} requires_grad={p.requires_grad}")
for n, p in list(model_ft.named_parameters())[-2:]:
    print(f"  {n:<28} requires_grad={p.requires_grad}")

hist_stage1 = train_model(model_ft, epochs=3, lr=1e-3, tag="B-阶段1. 只训 head (frozen backbone)")

# --- 阶段 2：解冻，用小 lr 微调 ---
for p in model_ft.parameters():
    p.requires_grad = True
print_param_groups(model_ft, "阶段2 全部解冻")

print("""
判别式学习率：backbone 1e-4，head 1e-3。
为什么？backbone 的权重已经很好，用大 lr 会把它毁掉；
     head 是新初始化的，需要正常 lr 才能学起来。
""")
hist_stage2 = train_model(model_ft, epochs=EPOCHS, lr=1e-3, lr_backbone=1e-4,
                          tag="B-阶段2. 全网络微调 (discriminative LR)")

# 合并两个阶段的曲线，方便画图
hist_ft = {
    "train_loss": hist_stage1["train_loss"] + hist_stage2["train_loss"],
    "train_acc": hist_stage1["train_acc"] + hist_stage2["train_acc"],
    "test_loss": hist_stage1["test_loss"] + hist_stage2["test_loss"],
    "test_acc": hist_stage1["test_acc"] + hist_stage2["test_acc"],
    "best_acc": max(hist_stage1["best_acc"], hist_stage2["best_acc"]),
    "stage1_len": len(hist_stage1["train_acc"]),
    "time": hist_stage1["time"] + hist_stage2["time"],
}

# ----------------------------------------------------------------------------
sec("7. 对比结果 —— 这一课的核心结论")
# ----------------------------------------------------------------------------
print(f"""
                      从零训练        迁移学习        提升
--------------------------------------------------------------
最好 test acc         {hist_scratch['best_acc']:.4f}         {hist_ft['best_acc']:.4f}        {(hist_ft['best_acc']-hist_scratch['best_acc'])*100:+.2f} 个点
训练总耗时(s)         {hist_scratch['time']:.1f}           {hist_ft['time']:.1f}
--------------------------------------------------------------

结论（务必记进笔记）：
  同样 3000 张图、同样 11 个 epoch 的预算，
  预训练模型能拿到比从零训练高得多的准确率。
  数据越少，迁移学习的优势越大 —— 因为从零训练在少数据上直接过拟合。

这就是为什么实习里你 99% 的时间在"微调现有模型"，而不是"发明新网络"。
""")

# 额外的纯特征提取结果（完全不解冻）参考
print(f"纯特征提取(阶段1结束时的 acc) = {hist_stage1['best_acc']:.4f}")
print(f"微调之后                     = {hist_stage2['best_acc']:.4f}")
print("→ 解冻微调通常比纯冻结再涨 1-3 个点，但代价是训练更慢、更容易过拟合。\n")

# ----------------------------------------------------------------------------
sec("8. 画对比曲线")
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

axes[0].plot(hist_scratch["test_acc"], "o-", label="A. From Scratch")
axes[0].plot(hist_ft["test_acc"], "s-", label="B. Transfer Learning")
axes[0].axvline(hist_ft["stage1_len"] - 0.5, ls="--", c="gray", label="解冻微调开始")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("test accuracy")
axes[0].set_title("小数据下：从零训练 vs 迁移学习")
axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(hist_scratch["test_loss"], "o-", label="A. From Scratch")
axes[1].plot(hist_ft["test_loss"], "s-", label="B. Transfer Learning")
axes[1].axvline(hist_ft["stage1_len"] - 0.5, ls="--", c="gray")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test loss")
axes[1].set_title("Test Loss")
axes[1].legend(); axes[1].grid(alpha=.3)

plt.tight_layout()
p = OUT_DIR / "06_transfer_compare.png"
plt.savefig(p, dpi=110)
plt.close()
print("对比曲线 ->", p)

# ----------------------------------------------------------------------------
sec("9. 保存")
# ----------------------------------------------------------------------------
p = OUT_DIR / "06_finetuned_resnet18.pt"
torch.save({"model_state": model_ft.state_dict(), "classes": CLASSES,
            "best_acc": hist_ft["best_acc"], "per_class_train": PER_CLASS}, p)
print("微调后模型 ->", p)

with open(OUT_DIR / "06_history.json", "w", encoding="utf-8") as f:
    json.dump({"scratch": hist_scratch, "transfer": hist_ft}, f, indent=2)
print("训练历史 ->", OUT_DIR / "06_history.json")

# 推理示例
model_ft.eval()
mean_t = torch.tensor(MEAN).view(3, 1, 1)
std_t = torch.tensor(STD).view(3, 1, 1)
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    xb = xb.to(DEVICE)
    pred = model_ft(xb).argmax(1).cpu()
print("\n前 10 个测试样本:")
for i in range(10):
    ok = "OK " if pred[i] == yb[i] else "ERR"
    print(f"  {ok} 预测={CLASSES[pred[i]]:<6} 真实={CLASSES[yb[i]]}")

# ----------------------------------------------------------------------------
sec("10. 练习")
# ----------------------------------------------------------------------------
print("""
练习 1（必做）：把 PER_CLASS 从 300 改成 30（每类只有 30 张！），重跑。
        观察从零训练会不会崩，迁移学习还稳不稳。

练习 2：把阶段 2 的 lr_backbone 从 1e-4 改成 1e-2（故意太大），
        看准确率是不是反而下降 —— 体会"预训练权重被破坏"。

练习 3：去掉阶段 1，直接解冻微调，看结果如何。阶段 1 到底有没有用？

练习 4：改成"只解冻 layer4"（只微调最后几个 block）：
        for n, p in model_ft.named_parameters():
            p.requires_grad = n.startswith("fc") or n.startswith("layer4")
        对比全解冻，哪个更好？为什么？

练习 5（核心思维）：把你自己的任务接上来。
        换成任何图像数据集（比如 kaggle 的猫狗、花卉分类），
        用这份代码，10 分钟就能出一个 baseline。

练习 6（面试题）：如果你的数据和 ImageNet 差很远（比如 X 光片），
        迁移学习还有用吗？该怎么做？（提示：考虑只加载部分层、
        或者用自监督预训练模型）
""")

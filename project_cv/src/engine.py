# -*- coding: utf-8 -*-
"""训练引擎：一个 epoch 的训练/评估、指标、早停、画曲线。

训练脚本（train.py）只负责配置和流程，真正的循环在这里。
这样你想换任务时，只改这个文件就行。
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
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class AverageMeter:
    """滑动平均，用来累计 loss / acc，避免自己算加权平均时写错。"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __str__(self):
        return f"{self.name}={self.avg:.4f}"


# ------------------------------------------------------------------ mixup
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Mixup：把两张图按随机比例线性混合，标签也按同样比例混合。

    作用：强正则化，小数据集上常能涨 1-3 个点。
    代价：训练变慢（loss 要按两组标签加权），且需要更多 epoch。
    典型场景：数据少、过拟合明显时。
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], lam


# ------------------------------------------------------------------ train
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[object] = None,
    mixup_alpha: float = 0.0,
    grad_clip: float = 0.0,
    epoch: int = 0,
    log_every: int = 0,
) -> Dict[str, float]:
    """跑一个 epoch 的训练，返回 {"loss":..., "acc":...}。

    五步骨架：前向 -> 损失 -> zero_grad -> backward -> step
    """
    model.train()
    loss_meter, acc_meter = AverageMeter("loss"), AverageMeter("acc")
    t0 = time.time()

    for i, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if mixup_alpha > 0:
            xb, ya, yb2, lam = mixup_batch(xb, yb, mixup_alpha)
            logits = model(xb)
            loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb2)
            pred = logits.argmax(1)
            # mixup 下准确率只是参考值，用主标签算
            correct = (pred == ya).float().mean().item()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            correct = (logits.argmax(1) == yb).float().mean().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), xb.size(0))
        acc_meter.update(correct, xb.size(0))

        if log_every and (i + 1) % log_every == 0:
            print(f"    [epoch {epoch}] iter {i+1}/{len(loader)}  "
                  f"loss={loss_meter.avg:.4f}  acc={acc_meter.avg:.4f}  "
                  f"用时 {time.time() - t0:.0f}s")

    return {"loss": loss_meter.avg, "acc": acc_meter.avg, "time": time.time() - t0}


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module,
             device: torch.device) -> Dict:
    """在验证/测试集上评估。

    三个必须记住的点：
      model.eval()      -> 关掉 dropout、BN 用 running stats
      torch.no_grad()   -> 不建计算图，省显存
      shuffle=False     -> 数据集本身要保证，否则指标会飘
    """
    model.eval()
    loss_meter = AverageMeter("loss")
    preds, trues = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss_meter.update(criterion(logits, yb).item(), xb.size(0))
        preds.append(logits.argmax(1).cpu())
        trues.append(yb.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()

    return {
        "loss": loss_meter.avg,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ------------------------------------------------------------------ early stop
class EarlyStopping:
    """验证指标不涨就停，并记住最好的一轮。

    mode="max" 用准确率（越大越好），mode="min" 用 loss。
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        assert mode in ("max", "min")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        """返回 True 表示刷新了最好成绩。"""
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ------------------------------------------------------------------ plotting
def plot_curves(history: Dict[str, List[float]], out_path: str | Path,
                best_epoch: int = -1, stage_boundary: int = -1) -> None:
    """画 train/val 的 loss 和 acc 曲线。判断过拟合/欠拟合全靠这张图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="train")
    axes[1].plot(ep, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.3)

    for ax in axes:
        if best_epoch > 0:
            ax.axvline(best_epoch, ls="--", c="gray", alpha=0.8,
                       label=f"best epoch={best_epoch}")
        if stage_boundary > 0:
            ax.axvline(stage_boundary, ls=":", c="red", alpha=0.8,
                       label=f"unfreeze@{stage_boundary}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


# ------------------------------------------------------------------ io helpers
def save_json(obj, path: str | Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def set_seed(seed: int) -> None:
    """固定所有随机源。做对比实验时，不同 run 之间只有你改的那个变量不同。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn 确定性会变慢，但能让结果可复现；对比实验时开着更公平
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_dataset(ds, class_names: List[str]) -> str:
    from collections import Counter
    from torch.utils.data import Subset
    targets = ds.dataset.targets if isinstance(ds, Subset) else ds.targets
    if isinstance(ds, Subset):
        targets = [targets[i] for i in ds.indices]
    cnt = Counter(targets)
    parts = [f"{class_names[k]}:{cnt.get(k, 0)}" for k in sorted(cnt)]
    return "  ".join(parts)

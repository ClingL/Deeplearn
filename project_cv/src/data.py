# -*- coding: utf-8 -*-
"""数据集构建与数据体检。

支持两种目录结构
-----------------
结构 A（数据集已经给了划分）::

    data/flowers/train/<类别名>/*.jpg
    data/flowers/val/<类别名>/*.jpg
    data/flowers/test/<类别名>/*.jpg   # 可选，没有就从 val 里再切一半

结构 B（只有按类别分好的文件夹）::

    data/flowers/<类别名>/*.jpg

结构 B 会按类别**分层**切分成 train / val / test。

为什么要分三份
--------------
  train: 更新参数
  val  : 选超参数、早停、选"最好的一轮"   <- 你只能看这个
  test : 最后评一次，只评一次             <- 这才是能写进简历的数字

如果拿 test 来调参，test 就"泄漏"进你的决策了，指标会虚高。
这是面试常问的点，也是很多人实际会犯的错。
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


import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ImageNet 统计量。用预训练权重时**必须**用它，否则输入分布和预训练时不匹配。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 各家数据集的验证集目录命名不统一：有的 val，有的 valid，有的 validation。
# （Kaggle 的 Oxford Flower 102 用的是 valid/）三种都要认。
VAL_DIR_NAMES = ("val", "valid", "validation")


# ---------------------------------------------------------------- transforms
def build_transforms(img_size: int, augment: str = "basic"):
    """返回 (train_tf, eval_tf)。

    augment:
        none   - 只 resize，做消融实验的对照组
        basic  - 随机裁剪 + 水平翻转
        strong - 再加 RandAugment + ColorJitter + RandomErasing

    验证/测试**绝对不能**用随机变换，否则指标会飘。
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if augment == "none":
        train_ops = [transforms.Resize((img_size, img_size))]
    elif augment == "basic":
        train_ops = [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
        ]
    elif augment == "strong":
        train_ops = [
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        raise ValueError(f"未知的 augment 选项: {augment}（只能是 none / basic / strong）")

    train_tf = transforms.Compose(
        train_ops
        + [transforms.ToTensor(), normalize]
        + ([transforms.RandomErasing(p=0.25)] if augment == "strong" else [])
    )

    crop = int(round(img_size * 1.14))
    eval_tf = transforms.Compose([
        transforms.Resize(crop),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------- data audit
SPLIT_NAMES = ("train", "training", "val", "valid", "validation", "test", "testing")


def _iter_class_dirs(data_dir: str | Path):
    """找出所有 (划分名, 类别目录)。

    支持三种布局：
      A. data/train/<类别>/*.jpg + data/val/...   （已经划分好）
      B. data/<类别>/*.jpg                        （未划分）
      C. data/<某一层>/<类别>/*.jpg                （多套了一层目录）

    返回 (布局说明, [(划分名或None, 类别目录 Path)])。

    为什么必须区分：如果数据已经划分成 train/val/test，
    而你还把 train、val、test 当成三个「类别」去统计，得到的
    类别分布和均衡性结论就完全是错的。
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {root}")

    # --- 布局 A ---
    split_dirs = [
        d for d in sorted(root.iterdir())
        if d.is_dir() and d.name.lower() in SPLIT_NAMES
        and any(c.is_dir() for c in d.iterdir())
    ]
    if split_dirs:
        pairs = []
        for sp in split_dirs:
            for cd in sorted(d for d in sp.iterdir() if d.is_dir()):
                pairs.append((sp.name, cd))
        if pairs:
            return "已划分 (train/val/test)", pairs

    # --- 布局 C：只套了一层 ---
    base = root
    subs = [d for d in root.iterdir() if d.is_dir()]
    if len(subs) == 1 and any(c.is_dir() for c in subs[0].iterdir()):
        base = subs[0]

    # --- 布局 B ---
    pairs = [(None, cd) for cd in sorted(d for d in base.iterdir() if d.is_dir())]
    return "未划分 (只有类别文件夹)", pairs


def audit_dataset(data_dir: str | Path) -> Dict:
    """数据体检：类别分布、不均衡比例、重复图（含跨划分泄漏）、损坏图。

    这就是简历上「数据体检发现 XX 张重复图」那句话的来源。
    """
    layout, pairs = _iter_class_dirs(data_dir)
    if not pairs:
        raise FileNotFoundError(f"{data_dir} 下没有找到任何类别文件夹")

    counts: Dict[str, int] = {}          # 类别 -> 图片数（跨划分合并）
    split_counts: Dict[str, int] = {}    # 划分 -> 图片数
    hashes: Dict[str, List] = {}         # md5 -> [(路径, 划分)]
    corrupt: List[str] = []
    sizes: Counter = Counter()

    for split, class_dir in pairs:
        files = sorted(f for f in class_dir.rglob("*") if f.suffix.lower() in IMG_EXTS)
        counts[class_dir.name] = counts.get(class_dir.name, 0) + len(files)
        if split:
            split_counts[split] = split_counts.get(split, 0) + len(files)

        for f in files:
            try:
                h = hashlib.md5(f.read_bytes()).hexdigest()
            except OSError:
                corrupt.append(str(f))
                continue
            hashes.setdefault(h, []).append((str(f), split))
            try:
                from PIL import Image
                with Image.open(f) as im:
                    sizes[im.size] += 1
            except Exception:
                corrupt.append(str(f))

    dup_groups = [v for v in hashes.values() if len(v) > 1]
    n_dup = sum(len(g) - 1 for g in dup_groups)

    # 跨划分重复 = 数据泄漏：同一张图既在 train 又在 val/test
    leak_groups = [
        g for g in dup_groups
        if len({sp for _, sp in g if sp}) > 1
    ]

    counts_sorted = sorted(counts.items(), key=lambda kv: kv[1])
    imbalance = (max(counts.values()) / max(min(counts.values()), 1)) if counts else 0.0

    return {
        "布局": layout,
        "类别数": len(counts),
        "图片总数": sum(counts.values()),
        "每类数量_升序": counts_sorted,
        "最少类": counts_sorted[0] if counts_sorted else None,
        "最多类": counts_sorted[-1] if counts_sorted else None,
        "不均衡比例": round(imbalance, 2),
        "各划分数量": split_counts,
        "重复图片数": n_dup,
        "重复组数": len(dup_groups),
        "跨划分重复组数": len(leak_groups),
        "跨划分重复示例": [[(p, sp) for p, sp in g[:4]] for g in leak_groups[:3]],
        "损坏图片数": len(corrupt),
        "损坏图片列表": corrupt[:20],
        "尺寸分布_top5": sizes.most_common(5),
        "重复示例": [[p for p, _ in g[:4]] for g in dup_groups[:5]],
    }


def print_audit(report: Dict) -> None:
    print("=" * 72)
    print("数据体检报告")
    print("=" * 72)
    print(f"{'目录布局':<10}: {report['布局']}")
    for k in ["类别数", "图片总数", "不均衡比例", "重复图片数", "重复组数", "损坏图片数"]:
        print(f"{k:<10}: {report[k]}")

    if report.get("各划分数量"):
        parts = "  ".join(f"{k}={v}" for k, v in sorted(report["各划分数量"].items()))
        print(f"{'各划分':<10}: {parts}")

    items = report["每类数量_升序"]
    print(f"\n类别分布（升序，共 {report['类别数']} 类，数量为各划分之和）:")
    # 类别多的时候只显示头尾，类别少就全显示（否则会重复打印）
    if len(items) > 20:
        show = items[:15] + [("...", 0)] + items[-5:]
    else:
        show = items
    for name, n in show:
        if name == "...":
            print("  ...")
        else:
            bar = "#" * min(int(n / max(1, report["最多类"][1]) * 40), 40)
            print(f"  {name:<24} {n:>7}  {bar}")

    if report.get("跨划分重复组数"):
        print(f"\n[严重] 发现 {report['跨划分重复组数']} 组**跨划分重复** —— 数据泄漏！")
        print("  同一张图同时出现在不同划分里，会让验证/测试指标虚高。")
        print("  示例（路径, 划分）:")
        for g in report["跨划分重复示例"]:
            for p, sp in g:
                print(f"    [{sp}] {p}")
        print("  → 处理: python project_cv/scripts/dedup_data.py --data-dir <dir> --apply")

    if report["重复图片数"]:
        print(f"\n发现 {report['重复图片数']} 张重复图（内容完全相同）。示例:")
        for g in report["重复示例"]:
            for p in g:
                print("   ", p)
        print("  → 去重: python project_cv/scripts/dedup_data.py --data-dir <dir> --apply")

    if report["损坏图片数"]:
        print(f"\n发现 {report['损坏图片数']} 张损坏图片:")
        for p in report["损坏图片列表"]:
            print("   ", p)

    print(f"\n图片尺寸分布 top5: {report['尺寸分布_top5']}")
    print("\n把以上内容整理进 README 的「数据」一节。")


# ---------------------------------------------------------------- class names
def maybe_display_names(class_names: List[str], data_dir: str | Path) -> List[str]:
    """如果数据集带 cat_to_name.json，把数字类别名换成可读名称。

    Oxford Flower 102 的类别目录是 1..102，cat_to_name.json 里是花名。
    换了之后混淆矩阵、per-class 准确率、README 里显示的都是花名。

    找不到映射文件就原样返回，不影响任何流程。
    """
    root = Path(data_dir)
    candidates = [
        root / "cat_to_name.json",
        root.parent / "cat_to_name.json",
        root.parent.parent / "cat_to_name.json",
    ]
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            mapping = json.loads(cand.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[data] 读取 {cand} 失败，忽略类别名映射: {e}")
            continue

        mapped = [str(mapping.get(str(c), str(c))) for c in class_names]
        # 名字有重复就别换了，否则混淆矩阵会看不出是哪个类
        if len(set(mapped)) != len(mapped):
            print(f"[data] {cand.name} 映射后有重复名称，保持原类别名")
            continue

        print(f"[data] 已用 {cand.name} 映射类别名，例如 "
              f"'{class_names[0]}' -> '{mapped[0]}'")
        return mapped
    return class_names


# ---------------------------------------------------------------- splitting
def _three_way_split(targets: np.ndarray, val_ratio: float, test_ratio: float,
                     seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """分层三分。保证每份里各类别比例一致。"""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(targets))
    test_size = test_ratio
    tmp_size = val_ratio + test_ratio
    if tmp_size <= 0 or tmp_size >= 1:
        raise ValueError(f"val_ratio + test_ratio 必须在 (0,1) 内，当前 {tmp_size}")

    train_idx, tmp_idx = train_test_split(
        idx, test_size=tmp_size, random_state=seed, stratify=targets
    )
    # 在剩下的里面再切 val / test
    rel_test = test_ratio / tmp_size
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=rel_test, random_state=seed, stratify=targets[tmp_idx]
    )
    return train_idx, val_idx, test_idx


def build_datasets(
    data_dir: str | Path,
    img_size: int = 224,
    augment: str = "basic",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """构建 (train_ds, val_ds, test_ds, class_names, info)。"""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")

    train_tf, eval_tf = build_transforms(img_size, augment)
    info: Dict = {"结构": None, "train数量": 0, "val数量": 0, "test数量": 0}

    # --- 结构 A：已有 train/val ---
    # 注意 val 目录的名字各家数据集不一样：有的叫 val，有的叫 valid，
    # 有的叫 validation（Kaggle 的 Oxford Flower 102 就是 valid/）。
    # 只认 "val" 的话会被误判成「未划分」，然后 train/valid/test 会被
    # 当成三个类别去训练 —— 结果完全错，而且不会报错，最难查。
    val_dir = next((root / n for n in VAL_DIR_NAMES if (root / n).is_dir()), None)
    if (root / "train").is_dir() and val_dir is not None:
        train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
        val_full = datasets.ImageFolder(val_dir, transform=eval_tf)
        if train_ds.classes != val_full.classes:
            raise ValueError(
                "train 和 val 的类别不一致！\n"
                f"train: {train_ds.classes}\nval:   {val_full.classes}"
            )
        class_names = train_ds.classes

        if (root / "test").is_dir():
            test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)
            val_ds = val_full
            info["结构"] = f"A (已有 train/{val_dir.name}/test)"
        else:
            # 没有 test 目录，就把 val 对半分成 val / test
            tg = np.array(val_full.targets)
            v_idx, t_idx = _three_way_split(tg, val_ratio=0.5, test_ratio=0.5, seed=seed)[1:]
            val_ds = Subset(val_full, v_idx)
            test_ds = Subset(val_full, t_idx)
            info["结构"] = f"A' (已有 train/{val_dir.name}，对半切成 val/test)"
            print(f"[data] 未找到 test 目录，把 {val_dir.name} 对半切分: "
                  f"val {len(v_idx)} / test {len(t_idx)}")

    # --- 结构 B：只有一个目录 ---
    else:
        scan_root = root
        subs = [d for d in root.iterdir() if d.is_dir()]
        if len(subs) == 1 and any(c.is_dir() for c in subs[0].iterdir()):
            scan_root = subs[0]
            print(f"[data] 只发现一个子目录，自动下钻到: {scan_root}")

        base = datasets.ImageFolder(scan_root)
        targets = np.array(base.targets)
        if len(targets) < 30:
            raise ValueError(f"图片太少（{len(targets)} 张），无法划分数据集")

        tr_idx, v_idx, t_idx = _three_way_split(targets, val_ratio, test_ratio, seed)
        train_ds = Subset(datasets.ImageFolder(scan_root, transform=train_tf), tr_idx)
        val_ds = Subset(datasets.ImageFolder(scan_root, transform=eval_tf), v_idx)
        test_ds = Subset(datasets.ImageFolder(scan_root, transform=eval_tf), t_idx)
        class_names = base.classes
        info["结构"] = "B (自动分层切分)"
        print(f"[data] 分层三分: train {len(tr_idx)} / val {len(v_idx)} / "
              f"test {len(t_idx)}  (seed={seed})")

    # 有些数据集（如 Oxford Flower 102）的类别目录是数字 ID，旁边放了一个
    # cat_to_name.json 做映射。有就换成可读名字，混淆矩阵、per-class 报告
    # 和 README 都会好看很多 —— 「雏菊 vs 蒲公英」比「23 vs 45」有说服力。
    class_names = maybe_display_names(class_names, root)

    info["train数量"] = len(train_ds)
    info["val数量"] = len(val_ds)
    info["test数量"] = len(test_ds)
    info["类别数"] = len(class_names)
    info["类别列表"] = class_names
    return train_ds, val_ds, test_ds, class_names, info


def build_loaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 64,
    augment: str = "basic",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
    class_weights: bool = False,
):
    """目录 -> DataLoader。返回 (train_loader, val_loader, test_loader, class_names, info)。"""
    train_ds, val_ds, test_ds, class_names, info = build_datasets(
        data_dir, img_size=img_size, augment=augment,
        val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
    )

    weights = None
    if class_weights:
        targets = _get_targets(train_ds)
        cnt = np.bincount(targets, minlength=len(class_names)).astype(np.float64)
        cnt[cnt == 0] = 1.0
        w = cnt.sum() / (len(class_names) * cnt)
        weights = torch.tensor(w, dtype=torch.float32)
        print(f"[data] 类别权重: {np.round(w, 3).tolist()}")

    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 64), shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=max(batch_size, 64), shuffle=False, **kw)

    info["类别权重"] = None if weights is None else weights.tolist()
    return train_loader, val_loader, test_loader, class_names, info


def _get_targets(ds) -> np.ndarray:
    if isinstance(ds, Subset):
        base_targets = np.array(ds.dataset.targets)
        return base_targets[ds.indices]
    return np.array(ds.targets)


def class_distribution(ds) -> Dict[str, int]:
    return dict(Counter(_get_targets(ds).tolist()))

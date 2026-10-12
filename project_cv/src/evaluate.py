# -*- coding: utf-8 -*-
"""错误分析（面试加分项）。

用法::

    python project_cv/src/evaluate.py --ckpt outputs/03_resnet_finetune/best.pt \\
        --data-dir data/flowers_split

产出（在 ckpt 同级的 eval/ 目录下）:
    confusion_matrix.png   混淆矩阵热力图
    confusion_matrix.txt   原始计数矩阵
    per_class_acc.txt      每个类别的准确率（找最差的类）
    top_confusions.txt     最容易混淆的类别对
    misclassified.png      预测错误的样本拼图
    summary.md             可直接粘进 README 的结论

为什么值得做：面试官问「你的模型哪里不好」，只答一个准确率是不够的。
能说清「A 类和 B 类混得最多，因为它们长得像 / 训练样本少」才是加分项。
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

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as data_mod    # noqa: E402
from src import engine              # noqa: E402
from src import models as models_mod  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="模型错误分析",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True, help="train.py 产出的 best.pt")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", default="test", choices=["test", "val", "train"],
                   help="在哪个划分上做分析（默认 test）")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--top-n", type=int, default=15, help="展示前 N 组易混类别")
    p.add_argument("--n-error-images", type=int, default=25, help="错误样本拼图数量")
    p.add_argument("--seed", type=int, default=None,
                   help="默认用 checkpoint 里记录的 seed，保证划分一致")
    return p.parse_args()


def main():
    args = parse_args()
    device = engine.get_device()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"找不到 checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    class_names = ckpt["class_names"]
    arch = ckpt["arch"]
    img_size = ckpt.get("img_size", 224)
    seed = args.seed if args.seed is not None else train_args.get("seed", 42)

    print("=" * 72)
    print(f"错误分析: {ckpt_path.name}")
    print("=" * 72)
    print(f"arch={arch}  类别数={len(class_names)}  img_size={img_size}  seed={seed}")
    if ckpt.get("test_metrics"):
        tm = ckpt["test_metrics"]
        print(f"训练时记录的 test acc={tm.get('acc', float('nan')):.4f}  "
              f"f1_macro={tm.get('f1_macro', float('nan')):.4f}")

    # ---- 重建数据（seed 一致 -> 划分完全一致，这是关键）----
    try:
        _, val_loader, test_loader, cn, info = data_mod.build_loaders(
            args.data_dir, img_size=img_size, batch_size=args.batch_size,
            augment="none",                       # 分析时不要随机增强
            val_ratio=train_args.get("val_ratio", 0.15),
            test_ratio=train_args.get("test_ratio", 0.15),
            num_workers=args.num_workers, seed=seed,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"错误: {e}\n请用 --data-dir 指向训练时用的那个数据集目录。")
    if cn != class_names:
        if len(cn) == len(class_names):
            # 类别数一样、只是名字不同：最常见的情况是数据集旁边多了
            # cat_to_name.json，把数字 ID 换成了可读名称。
            # 用新名字，报告里显示「daisy vs dandelion」比「23 vs 45」有用得多。
            print("[i] 类别数一致但名称不同，使用数据集当前的类别名（更可读）")
            print(f"    ckpt: {class_names[:3]}  ->  数据: {cn[:3]}")
            class_names = cn
        else:
            print(f"[!]  类别数不一致！\n  ckpt: {len(class_names)} 类\n  数据: {len(cn)} 类")
            print("   请确认 --data-dir 和训练时是同一个数据集。结果可能不可信。")
    loader = {"test": test_loader, "val": val_loader, "train": None}[args.split] or test_loader
    print(f"分析划分: {args.split}  样本数: {len(loader.dataset)}")

    # ---- 重建模型 ----
    model = models_mod.build_model(arch, n_classes=len(class_names),
                                   pretrained=False, img_size=img_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    res = engine.evaluate(model, loader, criterion, device)
    y_true, y_pred = res["y_true"], res["y_pred"]

    out_dir = ckpt_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n整体指标: acc={res['acc']:.4f}  f1_macro={res['f1_macro']:.4f}  "
          f"loss={res['loss']:.4f}")

    # ---- 混淆矩阵 ----
    n_cls = len(class_names)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    with open(out_dir / "confusion_matrix.txt", "w", encoding="utf-8") as f:
        f.write("行=真实类别, 列=预测类别\n")
        f.write("true\\pred\t" + "\t".join(class_names) + "\n")
        for i, name in enumerate(class_names):
            f.write(name + "\t" + "\t".join(str(v) for v in cm[i]) + "\n")

    # ---- 每类准确率 ----
    per_class = []
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        acc = cm[i, i] / total if total else 0.0
        per_class.append((name, acc, int(total), int(cm[i, i])))

    with open(out_dir / "per_class_acc.txt", "w", encoding="utf-8") as f:
        f.write(f"{'类别':<26}{'准确率':>9}{'样本数':>9}{'正确数':>9}\n")
        f.write("-" * 55 + "\n")
        for name, acc, total, correct in sorted(per_class, key=lambda x: x[1]):
            f.write(f"{name:<26}{acc:>9.4f}{total:>9}{correct:>9}\n")

    print(f"\n最差的 10 个类别（面试时重点讲这些）:")
    print(f"  {'类别':<24}{'准确率':>9}{'样本数':>9}")
    for name, acc, total, _ in sorted(per_class, key=lambda x: x[1])[:10]:
        bar = "." * int(acc * 30)
        print(f"  {name:<24}{acc:>9.4f}{total:>9}  {bar}")

    # ---- 易混类别对 ----
    pairs = []
    for i in range(n_cls):
        for j in range(n_cls):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)

    with open(out_dir / "top_confusions.txt", "w", encoding="utf-8") as f:
        f.write(f"{'次数':>7}  {'真实':<24} -> {'预测':<24}\n")
        f.write("-" * 62 + "\n")
        for cnt, a, b in pairs[:args.top_n]:
            f.write(f"{cnt:>7}  {a:<24} -> {b:<24}\n")

    print(f"\n最容易混淆的 {min(args.top_n, len(pairs))} 组:")
    for cnt, a, b in pairs[:args.top_n]:
        print(f"  {cnt:>5} 次   {a}  ->  {b}")

    # ---- 混淆矩阵热力图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        side = max(6, min(20, n_cls * 0.35))
        fig, ax = plt.subplots(figsize=(side, side * 0.9))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        if n_cls <= 30:
            ax.set_xticklabels(class_names, rotation=90, fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(f"混淆矩阵（按行归一化）  acc={res['acc']:.4f}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=120)
        plt.close()
        print(f"\n混淆矩阵图 -> {out_dir / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"画混淆矩阵失败: {e}")

    # ---- 错误样本拼图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wrong = np.where(y_true != y_pred)[0]
        if len(wrong) == 0:
            print("没有错分样本，无法生成拼图。")
        else:
            sel = wrong[:args.n_error_images]
            n_show = len(sel)
            cols = 5
            rows = int(np.ceil(n_show / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
            axes = np.atleast_1d(axes).ravel()

            ds = loader.dataset
            mean = torch.tensor(data_mod.IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(data_mod.IMAGENET_STD).view(3, 1, 1)

            for ax, k in zip(axes, sel):
                img, _ = ds[int(k)]
                img = (img * std + mean).clamp(0, 1)
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.set_title(f"真:{class_names[y_true[k]]}\n预:{class_names[y_pred[k]]}",
                             fontsize=7, color="red")
                ax.axis("off")
            for ax in axes[n_show:]:
                ax.axis("off")

            plt.suptitle(f"错分样本（共 {len(wrong)} 个，展示前 {n_show} 个）")
            plt.tight_layout()
            plt.savefig(out_dir / "misclassified.png", dpi=120)
            plt.close()
            print(f"错分样本拼图 -> {out_dir / 'misclassified.png'}")
    except Exception as e:
        print(f"画错误样本失败: {e}")

    # ---- 汇总 markdown ----
    worst3 = sorted(per_class, key=lambda x: x[1])[:3]
    lines = [
        f"# 错误分析: {ckpt_path.parent.name}",
        "",
        f"- 划分: {args.split}，样本数 {len(y_true)}",
        f"- 准确率: **{res['acc']:.4f}**，F1(macro): {res['f1_macro']:.4f}",
        f"- 错分样本数: {int((y_true != y_pred).sum())}",
        "",
        "## 最差的 3 个类别",
        "",
        "| 类别 | 准确率 | 样本数 |",
        "|---|---|---|",
    ]
    for name, acc, total, _ in worst3:
        lines.append(f"| {name} | {acc:.4f} | {total} |")

    lines += ["", f"## 最容易混淆的 {min(5, len(pairs))} 组", "",
              "| 次数 | 真实 | 预测 |", "|---|---|---|"]
    for cnt, a, b in pairs[:5]:
        lines.append(f"| {cnt} | {a} | {b} |")

    lines += ["", "## 我的分析（必须自己写，面试会问）", "",
              "- 为什么这几类容易混？（长得像 / 训练样本少 / 标注有歧义 / 分辨率不够）",
              "- 你打算怎么改？（补数据 / 针对性增强 / 换更大输入 / 类别合并）", ""]

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"分析汇总 -> {out_dir / 'summary.md'}")
    print("\n把 summary.md 里「我的分析」那部分自己写完，然后贴进项目 README。")


if __name__ == "__main__":
    main()

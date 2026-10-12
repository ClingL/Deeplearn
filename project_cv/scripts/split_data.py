# -*- coding: utf-8 -*-
"""把「按类别分好的单个目录」分层切分成 train / val / test。

用法::

    # 预览（不写文件）
    python project_cv/scripts/split_data.py --data-dir data/flowers --dry-run

    # 实际生成 data/flowers_split/{train,val,test}
    python project_cv/scripts/split_data.py --data-dir data/flowers

为什么需要它：`train.py` 也能自动切分，但把切分结果落盘的好处是
**之后每次实验用的都是同一份划分**，对比才公平，也方便你检查数据。

分层切分（stratify）的意思是：每个类别都按同样比例分到三份里，
不会出现「某类全在训练集、验证集里一个都没有」的情况。
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
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as data_mod  # noqa: E402

IMG_EXTS = data_mod.IMG_EXTS


def parse_args():
    p = argparse.ArgumentParser(description="分层切分图像数据集")
    p.add_argument("--data-dir", required=True, help="源目录（每个类别一个子文件夹）")
    p.add_argument("--out-dir", default=None, help="输出目录，默认 <data-dir>_split")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="copy", choices=["copy", "hardlink"],
                   help="copy 最稳；hardlink 更快且不占额外空间（同一磁盘才行）")
    p.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.data_dir)
    if not src.is_dir():
        raise SystemExit(f"源目录不存在: {src}")

    # 找类别目录
    scan = src
    subs = [d for d in src.iterdir() if d.is_dir()]
    if len(subs) == 1 and any(c.is_dir() for c in subs[0].iterdir()):
        scan = subs[0]
        print(f"[split] 自动下钻到 {scan}")

    class_dirs = sorted(d for d in scan.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"{scan} 下没有类别子文件夹")

    out = Path(args.out_dir) if args.out_dir else src.parent / (src.name + "_split")

    # 收集并分层切分
    plan = {"train": [], "val": [], "test": []}
    print(f"[split] 源: {scan}")
    print(f"[split] 输出: {out}")
    print(f"[split] 比例 train/val/test = "
          f"{1 - args.val_ratio - args.test_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}")
    print(f"{'类别':<24}{'总数':>7}{'train':>8}{'val':>7}{'test':>7}")
    print("-" * 55)

    total = 0
    rng = np.random.default_rng(args.seed)
    for cd in class_dirs:
        files = sorted(f for f in cd.rglob("*") if f.suffix.lower() in IMG_EXTS)
        if not files:
            print(f"{cd.name:<24}{'0':>7}  (空目录，跳过)")
            continue
        idx = np.arange(len(files))
        rng.shuffle(idx)
        n = len(files)
        n_test = int(round(n * args.test_ratio))
        n_val = int(round(n * args.val_ratio))
        # 保证每类 train 至少 1 张
        if n - n_val - n_test < 1:
            n_val = max(0, min(n_val, n - 1))
            n_test = max(0, min(n_test, n - 1 - n_val))

        test_i = idx[:n_test]
        val_i = idx[n_test:n_test + n_val]
        train_i = idx[n_test + n_val:]

        for split, sel in [("train", train_i), ("val", val_i), ("test", test_i)]:
            for i in sel:
                plan[split].append((files[int(i)], cd.name))

        total += n
        print(f"{cd.name:<24}{n:>7}{len(train_i):>8}{len(val_i):>7}{len(test_i):>7}")

    print("-" * 55)
    print(f"{'合计':<24}{total:>7}{len(plan['train']):>8}"
          f"{len(plan['val']):>7}{len(plan['test']):>7}")

    if args.dry_run:
        print("\n[dry-run] 未写任何文件。去掉 --dry-run 实际执行。")
        return

    if out.exists() and any(out.iterdir()):
        print(f"\n[!]  输出目录已存在且非空: {out}")
        resp = input("继续会覆盖同名文件，确定吗？输入 yes 继续: ").strip().lower()
        if resp != "yes":
            print("已取消。")
            return

    print(f"\n[split] 开始{'硬链接' if args.mode == 'hardlink' else '复制'}文件...")
    for split, items in plan.items():
        for i, (f, cls) in enumerate(items, 1):
            dst_dir = out / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f.name
            if dst.exists():
                # 同名文件加后缀，避免覆盖
                dst = dst_dir / f"{f.stem}_{abs(hash(str(f))) % 100000}{f.suffix}"
            if args.mode == "hardlink":
                try:
                    import os
                    os.link(f, dst)
                    continue
                except Exception:
                    pass  # 跨盘等情况退回复制
            shutil.copy2(f, dst)
        print(f"  {split}: {len(items)} 张 -> {out / split}")

    print(f"\n完成。下一步:")
    print(f"  python project_cv/src/train.py --data-dir {out} --run-name 01_baseline "
          f"--arch smallcnn --no-pretrained --epochs 40")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""生成一个假数据集，用来 30 秒验证 project_cv 的流程是否正常。

用法::

    # 1. 造假数据（3 类，每类 40 张，32x32 彩色噪声图）
    python project_cv/scripts/make_dummy_data.py

    # 2. 用 2 个 epoch 跑一遍，验证训练/验证/测试/画图/存盘全通
    python project_cv/src/train.py --data-dir data/_dummy --run-name smoke_test \\
        --arch smallcnn --no-pretrained --img-size 32 --epochs 2 --batch-size 16

    # 3. 验证错误分析脚本
    python project_cv/src/evaluate.py --ckpt outputs/smoke_test/best.pt --data-dir data/_dummy

    # 4. 验证表格聚合
    python project_cv/scripts/make_report.py --outputs-dir outputs

    # 5. 用完删掉
    #    Windows:  Remove-Item -Recurse -Force data\\_dummy, outputs\\smoke_test

为什么要做这一步：先确认「代码没问题」，再去查「数据有没有问题」。
不然一旦报错，你分不清是工程的 bug 还是数据集的坑。

注意：图是纯噪声，所以模型学不到东西（准确率约等于 1/类别数）。
这是**预期的**，说明流程正确；重点是脚本能从头跑到尾不报错。
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description="生成用于烟雾测试的假图像数据集")
    p.add_argument("--out-dir", default=str(ROOT.parent / "data" / "_dummy"),
                   help="输出目录，默认 <项目根>/data/_dummy")
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("需要 Pillow：请用项目虚拟环境运行（见 run.cmd）")

    out = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    # 分层地把每类切成 train / val / test
    n_val = max(1, args.per_class // 5)
    n_test = max(1, args.per_class // 5)
    n_train = args.per_class - n_val - n_test
    if n_train < 1:
        raise SystemExit("每类图片太少，请把 --per-class 调大（至少 10）")

    total = 0
    print(f"生成到: {out}")
    print(f"{'类别':<12}{'train':>8}{'val':>6}{'test':>6}")
    print("-" * 32)

    for c in range(args.classes):
        cls = f"class_{c}"
        # 每一类给一个略微不同的基色，让类别间有点差别（但仍然基本是噪声）
        base = rng.integers(60, 200, size=3)

        for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                noise = rng.integers(0, 80, size=(args.img_size, args.img_size, 3))
                arr = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{cls}_{split}_{i:03d}.jpg", quality=85)
                total += 1
        print(f"{cls:<12}{n_train:>8}{n_val:>6}{n_test:>6}")

    print("-" * 32)
    print(f"共 {total} 张图，{args.classes} 类")

    print(f"""
下一步（在项目根目录执行）:

  .\\run.cmd project_cv\\src\\train.py --data-dir data\\_dummy --run-name smoke_test ^
      --arch smallcnn --no-pretrained --img-size {args.img_size} --epochs 2 --batch-size 16

  .\\run.cmd project_cv\\src\\evaluate.py --ckpt outputs\\smoke_test\\best.pt --data-dir data\\_dummy

  .\\run.cmd project_cv\\scripts\\make_report.py --outputs-dir outputs

预期：能一路跑完不报错。准确率约 {1.0 / args.classes:.2f}（纯噪声，学不到东西是正常的）。
""")


if __name__ == "__main__":
    main()

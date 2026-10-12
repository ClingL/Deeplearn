# -*- coding: utf-8 -*-
"""把所有 run 的 metrics.json 聚合成一张 markdown 消融实验表。

用法::

    python project_cv/scripts/make_report.py --outputs-dir outputs

这张表就是你简历上「消融实验表明……」那一行的来源。
把它粘进项目 README。
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
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="聚合消融实验表格")
    p.add_argument("--outputs-dir", default="outputs", help="train.py 的输出根目录")
    p.add_argument("--out", default=None, help="把表格写到文件（默认只打印）")
    p.add_argument("--sort-by", default="test_acc", help="按哪个字段排序")
    p.add_argument("--exclude", default="smoke_test,verify,_*",
                   help="不参与统计的实验名，逗号分隔；支持 * 后缀通配。"
                        "默认排除烟雾测试，否则它会变成 baseline 把对比搞乱")
    p.add_argument("--baseline", default=None,
                   help="作为基准（提升列的参照）的实验名；默认取 test_acc 最低的那个")
    return p.parse_args()


def make_matcher(patterns):
    pats = [x.strip() for x in patterns.split(",") if x.strip()]

    def excluded(name: str) -> bool:
        for pat in pats:
            if pat.endswith("*"):
                if name.startswith(pat[:-1]):
                    return True
            elif name == pat:
                return True
        return False

    return excluded


def yn(v) -> str:
    if v is True or v == "True":
        return "是"
    if v is False or v == "False":
        return "否"
    return str(v)


def main():
    args = parse_args()
    root = Path(args.outputs_dir)
    if not root.is_dir():
        raise SystemExit(f"找不到目录: {root}\n先跑几个 train.py 实验。")

    is_excluded = make_matcher(args.exclude)

    rows = []
    for d in sorted(root.iterdir()):
        if is_excluded(d.name):
            continue
        mf = d / "metrics.json"
        if mf.is_file():
            try:
                rows.append(json.loads(mf.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"跳过 {mf}: {e}")

    if not rows:
        raise SystemExit(f"{root} 下没有找到任何 metrics.json。")

    key = args.sort_by
    rows.sort(key=lambda r: r.get(key, 0) or 0, reverse=True)

    # 参考行：默认取 test_acc 最低的那个实验（它就是你的 baseline）
    if args.baseline:
        matches = [r for r in rows if r.get("run_name") == args.baseline]
        if not matches:
            raise SystemExit(f"--baseline {args.baseline} 不在这批实验里。"
                             f"可选: {[r.get('run_name') for r in rows]}")
        baseline = matches[0]
    else:
        baseline = min(rows, key=lambda r: r.get(key, 0) or 0)
    base_acc = baseline.get("test_acc", 0) or 0

    headers = ["实验", "骨干", "预训练", "增强", "冻结轮数", "backbone lr",
               "参数量", "最好轮", "val acc", "test acc", "test F1", "test acc 提升", "训练(分)"]
    widths = [max(len(h), 6) for h in headers]

    def fmt_row(cells):
        out = []
        for c, w in zip(cells, widths):
            s = str(c)
            out.append(s + " " * max(0, w - len(s)) if not _is_num(s) else " " * max(0, w - len(s)) + s)
        return "| " + " | ".join(out) + " |"

    def _is_num(s):
        try:
            float(s)
            return True
        except Exception:
            return False

    body = []
    for r in rows:
        acc = r.get("test_acc", 0) or 0
        delta = acc - base_acc
        body.append([
            r.get("run_name", "?"),
            r.get("arch", "?"),
            yn(r.get("pretrained")),
            r.get("augment", "?"),
            r.get("freeze_epochs", 0),
            f"{r.get('backbone_lr', 0):.1e}" if r.get("backbone_lr") else "-",
            f"{r.get('params_total', 0):,}",
            r.get("best_epoch", "?"),
            f"{r.get('val_acc', 0):.4f}",
            f"{acc:.4f}",
            f"{r.get('test_f1_macro', 0):.4f}",
            f"{delta:+.4f}" if delta else "-",
            f"{r.get('train_minutes', 0):.1f}",
        ])

    # 重新计算宽度
    widths = [len(h) for h in headers]
    for row in body:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"

    def render(cells, right_align_num=True):
        parts = []
        for c, w in zip(cells, widths):
            s = str(c)
            pad = w - len(s)
            if right_align_num and _is_num(s.replace(",", "").replace("%", "")):
                parts.append(" " + " " * pad + s + " ")
            else:
                parts.append(" " + s + " " * pad + " ")
        return "|" + "|".join(parts) + "|"

    lines = []
    lines.append(render(headers))
    lines.append(sep)
    for row in body:
        lines.append(render(row))

    table = "\n".join(lines)

    print("=" * 100)
    print("消融实验汇总表（可直接粘进 README / 简历）")
    print("=" * 100)
    print(table)
    print()

    print(f"baseline（最差的一行）: {baseline.get('run_name')}  test acc={base_acc:.4f}")
    best = rows[0]
    print(f"最好的一行            : {best.get('run_name')}  "
          f"test acc={best.get('test_acc', 0):.4f}  "
          f"提升 {(best.get('test_acc', 0) or 0) - base_acc:+.4f}")

    print("""
怎么把这个表变成简历语言（照这个句式改）：
  「以从零训练的 SmallCNN 为 baseline（test acc XX%），
    改用 ImageNet 预训练 ResNet18 并采用两阶段训练（先冻结 backbone 训 head，
    再以判别式学习率 backbone 1e-4 / head 1e-3 微调），准确率提升至 XX%（+XXpt）；
    在此基础上加入 RandAugment 与 RandomErasing 再提升 XXpt。」

注意：表里的「提升」一列是相对 baseline 的。
     简历上每个数字都要能指向表里某一行，面试官问起来你能直接翻出来。
""")

    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"表格已写入 {args.out}")


if __name__ == "__main__":
    main()

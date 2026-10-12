# -*- coding: utf-8 -*-
"""数据体检：类别分布、不均衡、重复图、损坏图。

用法::

    python project_cv/scripts/audit_data.py --data-dir data/flowers

把输出整理进 README 的「数据」一节 —— 这就是简历上
「数据体检发现 XX 张重复图 / X 类不均衡」那句话的来源。
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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as data_mod  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="图像数据集体检")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--json-out", default=None,
                   help="把报告也存成 json（建议放到项目根，便于写进 README）")
    args = p.parse_args()

    try:
        report = data_mod.audit_dataset(args.data_dir)
    except FileNotFoundError as e:
        raise SystemExit(f"错误: {e}\n请用 --data-dir 指向正确的数据集目录。")
    data_mod.print_audit(report)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n报告已写入 {args.json_out}")

    print("""
下一步:
  1. 有重复图  -> python project_cv/scripts/dedup_data.py --data-dir <dir> --apply
  2. 要划分数据 -> python project_cv/scripts/split_data.py --data-dir <dir>
  3. 类别不均衡 -> 训练时加 --class-weights
""")


if __name__ == "__main__":
    main()

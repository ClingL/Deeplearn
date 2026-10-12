# -*- coding: utf-8 -*-
"""重复图片检测与清理。

用法::

    # 只看报告
    python project_cv/scripts/dedup_data.py --data-dir data/flowers

    # 实际清理（把重复项移到 <data-dir>/_duplicates/，不直接删除）
    python project_cv/scripts/dedup_data.py --data-dir data/flowers --apply

为什么要去重：
  1. 同一张图同时出现在训练集和验证集 -> 指标虚高（数据泄漏）
  2. 同一类的重复图会让该类被过度采样
  3. **跨类别重复**是标注错误，必须人工确认

安全设计：默认只报告，`--apply` 也只是**移动**到 `_duplicates/`，
不会真删，你确认没问题后自己删那个目录。
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
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as data_mod  # noqa: E402

IMG_EXTS = data_mod.IMG_EXTS


def find_scan_root(root: Path) -> Path:
    scan = root
    subs = [d for d in root.iterdir() if d.is_dir()]
    if len(subs) == 1 and any(c.is_dir() for c in subs[0].iterdir()):
        scan = subs[0]
    return scan


def main():
    p = argparse.ArgumentParser(description="重复图片检测与清理")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--apply", action="store_true", help="实际移动重复文件（默认只报告）")
    p.add_argument("--quarantine", default="_duplicates",
                   help="重复文件移到的子目录名（相对 --data-dir）")
    args = p.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    scan = find_scan_root(root)
    print(f"[dedup] 扫描目录: {scan}")

    by_hash: dict = defaultdict(list)
    n_files = 0
    for f in scan.rglob("*"):
        if f.suffix.lower() in IMG_EXTS:
            n_files += 1
            by_hash[hashlib.md5(f.read_bytes()).hexdigest()].append(f)

    dup_groups = {h: v for h, v in by_hash.items() if len(v) > 1}
    n_extra = sum(len(v) - 1 for v in dup_groups.values())

    print(f"[dedup] 总图片 {n_files} 张，唯一 {len(by_hash)} 张")
    print(f"[dedup] 重复组 {len(dup_groups)} 组，冗余文件 {n_extra} 张 "
          f"（占 {n_extra / max(n_files, 1):.1%}）")

    if not dup_groups:
        print("\n没有重复图片。")
        return

    # 跨类别重复 = 标注冲突，单独标出来
    cross_label = []
    for h, files in dup_groups.items():
        classes = {f.parent.name for f in files}
        if len(classes) > 1:
            cross_label.append((h, files, classes))

    # 类别均衡性影响
    class_dup = defaultdict(int)
    for h, files in dup_groups.items():
        for f in sorted(files)[1:]:
            class_dup[f.parent.name] += 1

    print(f"\n前 10 组重复（每组只显示前 3 个路径）:")
    for i, (h, files) in enumerate(list(dup_groups.items())[:10], 1):
        classes = sorted({f.parent.name for f in files})
        tag = "  [!] 跨类别!" if len(classes) > 1 else ""
        print(f"  #{i} {h[:10]}  {len(files)} 张  类别={classes}{tag}")
        for f in sorted(files)[:3]:
            print(f"        {f.relative_to(scan)}")

    if class_dup:
        print(f"\n各冗余文件所属类别（去重后这些类会'变少'）:")
        for cls, n in sorted(class_dup.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {cls:<24} {n:>5} 张冗余")

    if cross_label:
        print(f"\n[!]  发现 {len(cross_label)} 组**跨类别重复** —— 同一张图被标成了不同类别。")
        print("   这是标注错误，必须人工判断该归到哪一类，脚本不会自动处理。")
        for h, files, classes in cross_label[:5]:
            print(f"   类别 {sorted(classes)}:")
            for f in sorted(files)[:4]:
                print(f"     {f.relative_to(scan)}")

    if not args.apply:
        print(f"\n[报告模式] 未改动任何文件。")
        print(f"要清理请加 --apply，重复项会被移到 {root / args.quarantine}/")
        print("（跨类别重复的组会被跳过，需要你自己处理）")
        print("\n建议把这段结论写进 README：")
        print(f"  「数据体检发现 {n_extra} 张重复图（占 {n_extra / max(n_files,1):.1%}），"
              f"去重后剩 {len(by_hash)} 张」")
        return

    qroot = root / args.quarantine
    moved = skipped = 0
    for h, files in dup_groups.items():
        classes = {f.parent.name for f in files}
        if len(classes) > 1:
            skipped += len(files) - 1
            continue
        keep = sorted(files)[0]
        for f in sorted(files)[1:]:
            rel = f.relative_to(scan)
            dst = qroot / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst = dst.with_name(f"{dst.stem}_{h[:8]}{dst.suffix}")
            shutil.move(str(f), str(dst))
            moved += 1
        print(f"  保留 {keep.relative_to(scan)}，移除 {len(files) - 1} 个副本")

    print(f"\n[done] 移动 {moved} 张到 {qroot}")
    if skipped:
        print(f"       跳过 {skipped} 张跨类别重复（需人工处理）")
    print(f"       去重后图片数: {n_files - moved}")
    print("\n确认无误后可以手动删除:")
    print(f"  Remove-Item -Recurse -Force '{qroot}'")
    print("\n注意：去重后请重新体检一次（audit_data.py），"
          "并检查类别分布有没有变得更不均衡。")


if __name__ == "__main__":
    main()

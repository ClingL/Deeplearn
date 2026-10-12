# -*- coding: utf-8 -*-
"""训练主脚本（两阶段迁移学习 + 消融实验）。

用法示例（完整命令见 project_cv/README.md）::

    # baseline: 从零训练的小 CNN
    python project_cv/src/train.py --data-dir data/flowers_split \\
        --run-name 01_baseline --arch smallcnn --no-pretrained \\
        --augment basic --epochs 40

    # 主实验: 预训练 ResNet18 + 判别式学习率微调
    python project_cv/src/train.py --data-dir data/flowers_split \\
        --run-name 03_resnet_finetune --arch resnet18 --pretrained \\
        --epochs 40 --lr 1e-3 --backbone-lr 1e-4 --freeze-epochs 3

每个 run 的产物落在 outputs/<run_name>/，最后用 scripts/make_report.py 聚合成表格。
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
import time
from pathlib import Path

import torch
import torch.nn as nn

# 让脚本无论从哪个目录调用都能 import src.*
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data as data_mod           # noqa: E402
from src import engine                      # noqa: E402
from src import models as models_mod        # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="图像分类训练脚本（迁移学习 + 消融实验）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据
    p.add_argument("--data-dir", required=True, help="数据集目录（结构见 src/data.py 说明）")
    p.add_argument("--img-size", type=int, default=224, help="输入分辨率")
    p.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    p.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Windows 上必须配合 __main__ 保护，本脚本已加；默认 0 最稳")
    # 模型
    p.add_argument("--arch", default="resnet18",
                   help="smallcnn / resnet18 / resnet34 / resnet50 / efficientnet_b0 / mobilenet_v3_small")
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=True,
                   help="使用 ImageNet 预训练权重（默认开）")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   help="随机初始化，从零训练（做 baseline 用）")
    # 训练
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="head / 全网络的学习率")
    p.add_argument("--backbone-lr", type=float, default=None,
                   help="backbone 的学习率，默认取 lr/10（判别式学习率）")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--freeze-epochs", type=int, default=0,
                   help="前 N 轮冻结 backbone 只训 head，第 N+1 轮解冻微调；0 表示不冻结")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.0, help="mixup 的 alpha，0 表示关闭")
    p.add_argument("--grad-clip", type=float, default=0.0, help="梯度裁剪阈值，0 表示关闭")
    p.add_argument("--class-weights", action="store_true", help="按类别频率给损失加权")
    p.add_argument("--augment", default="basic", choices=["none", "basic", "strong"])
    p.add_argument("--patience", type=int, default=10, help="早停耐心值，0 表示关闭早停")
    p.add_argument("--monitor", default="acc", choices=["acc", "f1_macro", "loss"],
                   help="早停和选最优模型依据的指标")
    p.add_argument("--seed", type=int, default=42)
    # 输出
    p.add_argument("--out-dir", default=None, help="默认 <项目根>/outputs")
    p.add_argument("--run-name", default=None, help="不填会根据配置自动生成")
    p.add_argument("--log-every", type=int, default=0, help="每 N 个 iteration 打印一次，0 关闭")
    return p.parse_args()


def build_optimizer(model, arch, lr, backbone_lr, weight_decay, frozen):
    """构造优化器。frozen=True 时只优化 head；否则用两组不同学习率。"""
    backbone, head = models_mod.split_params(model, arch)
    if frozen:
        params = [p for p in head if p.requires_grad]
        print(f"[opt] 冻结阶段：只优化 head，lr={lr:.1e}（{len(params)} 个张量）")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    groups = [
        {"params": [p for p in backbone if p.requires_grad], "lr": backbone_lr,
         "name": "backbone"},
        {"params": [p for p in head if p.requires_grad], "lr": lr, "name": "head"},
    ]
    print(f"[opt] 判别式学习率：backbone lr={backbone_lr:.1e} / head lr={lr:.1e}")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def main():
    args = parse_args()

    engine.set_seed(args.seed)
    device = engine.get_device()

    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr / 10.0
    if args.backbone_lr is None:
        print(f"[cfg] 未指定 --backbone-lr，默认取 lr/10 = {backbone_lr:.1e}")

    out_root = Path(args.out_dir) if args.out_dir else (ROOT / "outputs")
    run_name = args.run_name or (
        f"{args.arch}_{'pre' if args.pretrained else 'scratch'}_{args.augment}"
        f"{'_ft' if args.freeze_epochs > 0 else ''}"
    )
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"run: {run_name}")
    print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print("=" * 72)
    for k, v in vars(args).items():
        print(f"  {k:<16} = {v}")
    print(f"  {'backbone_lr':<16} = {backbone_lr}")

    # ---------------- 数据 ----------------
    print("\n" + "-" * 72)
    print("构建数据集")
    print("-" * 72)
    train_loader, val_loader, test_loader, class_names, data_info = data_mod.build_loaders(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        augment=args.augment,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        class_weights=args.class_weights,
    )
    print(f"[data] 结构: {data_info['结构']}")
    print(f"[data] 类别数: {len(class_names)} -> {class_names[:10]}"
          + (" ..." if len(class_names) > 10 else ""))
    print(f"[data] train {data_info['train数量']} / val {data_info['val数量']} / "
          f"test {data_info['test数量']}")
    print(f"[data] 每类数量: {data_mod.class_distribution(train_loader.dataset)}")
    print(f"[data] batch: train {len(train_loader)} / val {len(val_loader)} / test {len(test_loader)}")

    # ---------------- 模型 ----------------
    print("\n" + "-" * 72)
    print("构建模型")
    print("-" * 72)
    model = models_mod.build_model(
        args.arch, n_classes=len(class_names),
        pretrained=args.pretrained, img_size=args.img_size,
    ).to(device)

    if args.freeze_epochs > 0:
        models_mod.freeze_backbone(model, args.arch, freeze=True)
        print(f"[model] 前 {args.freeze_epochs} 轮冻结 backbone（只训 head）")
    models_mod.print_model_summary(model, args.arch)

    # ---------------- 损失 ----------------
    weight = None
    if args.class_weights and data_info.get("类别权重"):
        weight = torch.tensor(data_info["类别权重"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    print(f"[loss] CrossEntropyLoss(label_smoothing={args.label_smoothing}, "
          f"weight={'有' if weight is not None else '无'})")

    # ---------------- 优化器 / 调度器 ----------------
    optimizer = build_optimizer(model, args.arch, args.lr, backbone_lr,
                                args.weight_decay, frozen=args.freeze_epochs > 0)
    stage1_epochs = args.freeze_epochs if args.freeze_epochs > 0 else 0

    # 余弦调度的周期长度必须小心：
    #   有解冻     -> 从解冻那一轮开始衰减，周期 = 剩余轮数
    #   全程冻结   -> freeze_epochs >= epochs，周期 = 总轮数
    # 绝不能让它算出 0 或 1：
    #   T_max = 0 -> ZeroDivisionError，脚本直接崩
    #   T_max = 1 -> 学习率在 0 和初始值之间反复横跳，等于每两轮才学一次，训练废掉
    if stage1_epochs >= args.epochs:
        cosine_tmax = args.epochs
        print(f"[sched] freeze_epochs({args.freeze_epochs}) >= epochs({args.epochs})，"
              f"backbone 全程冻结，这是「纯特征提取」实验")
    else:
        cosine_tmax = max(1, args.epochs - stage1_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_tmax)
    print(f"[sched] CosineAnnealingLR (T_max={cosine_tmax})")

    early = engine.EarlyStopping(patience=args.patience if args.patience > 0 else 10**9,
                                 mode="min" if args.monitor == "loss" else "max")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_f1_macro": [], "lr": [], "monitor": args.monitor}
    test_metrics = {}

    # ---------------- 训练循环 ----------------
    print("\n" + "=" * 72)
    print(f"开始训练（共 {args.epochs} 轮，监控指标 = {args.monitor}）")
    print("=" * 72)
    header = (f"{'epoch':>5} | {'lr':>8} | {'tr_loss':>8} {'tr_acc':>7} | "
              f"{'va_loss':>8} {'va_acc':>7} {'va_f1':>7} | {'耗时':>7}")
    print(header)
    print("-" * len(header))

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        # ---- 解冻时刻 ----
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            print("\n" + ">" * 72)
            print(f">>> 第 {epoch} 轮：解冻 backbone，开始微调（判别式学习率）")
            print(">" * 72)
            models_mod.freeze_backbone(model, args.arch, freeze=False)
            models_mod.print_model_summary(model, args.arch)
            optimizer = build_optimizer(model, args.arch, args.lr, backbone_lr,
                                        args.weight_decay, frozen=False)
            remaining = max(1, args.epochs - args.freeze_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        tr = engine.train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler, mixup_alpha=args.mixup,
            grad_clip=args.grad_clip, epoch=epoch, log_every=args.log_every,
        )
        va = engine.evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_f1_macro"].append(va["f1_macro"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        monitor_value = {"acc": va["acc"], "f1_macro": va["f1_macro"],
                         "loss": va["loss"]}[args.monitor]
        improved = early.step(monitor_value, model, epoch)
        flag = "  * best" if improved else ""

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5} | {lr_now:>8.2e} | {tr['loss']:>8.4f} {tr['acc']:>7.4f} | "
              f"{va['loss']:>8.4f} {va['acc']:>7.4f} {va['f1_macro']:>7.4f} | "
              f"{tr['time']:>6.1f}s{flag}")

        if early.should_stop:
            print(f"\n[early-stop] 验证 {args.monitor} 连续 {args.patience} 轮没有改善，"
                  f"在第 {epoch} 轮停止。")
            break

    total_time = time.time() - t_start
    print(f"\n训练结束，总耗时 {total_time / 60:.1f} 分钟")
    print(f"最好的一轮是 epoch {early.best_epoch}，"
          f"val {args.monitor} = {early.best:.4f}")

    # ---------------- 载入最优权重，评测试集 ----------------
    if early.best_state is not None:
        model.load_state_dict(early.best_state)
        print("[ckpt] 已载入验证集最优权重")

    print("\n" + "=" * 72)
    print("测试集评估（只评一次，这个数字才能写进简历）")
    print("=" * 72)
    te = engine.evaluate(model, test_loader, criterion, device)
    test_metrics = {k: v for k, v in te.items() if not k.startswith("y_")}
    for k, v in test_metrics.items():
        print(f"  test {k:<16} = {v:.4f}")
    print(f"\n  val  acc = {history['val_acc'][early.best_epoch - 1]:.4f}"
          f"   (epoch {early.best_epoch})")
    print(f"  test acc = {te['acc']:.4f}")
    gap = history['val_acc'][early.best_epoch - 1] - te['acc']
    print(f"  val - test = {gap:+.4f}")
    if abs(gap) > 0.05:
        print("  [!]  差距偏大，说明验证集太小或划分不均匀，指标不太可信。"
              "建议增大 val 比例或做交叉验证。")

    # ---------------- 保存产物 ----------------
    torch.save({
        "model_state": model.state_dict(),
        "arch": args.arch,
        "class_names": class_names,
        "img_size": args.img_size,
        "best_epoch": early.best_epoch,
        "val_metrics": {"loss": history["val_loss"][early.best_epoch - 1],
                        "acc": history["val_acc"][early.best_epoch - 1],
                        "f1_macro": history["val_f1_macro"][early.best_epoch - 1]},
        "test_metrics": test_metrics,
        "args": vars(args),
    }, run_dir / "best.pt")
    torch.save({"model_state": model.state_dict(), "arch": args.arch,
                "class_names": class_names, "img_size": args.img_size},
               run_dir / "last.pt")
    engine.save_json(vars(args), run_dir / "config.json")
    engine.save_json(history, run_dir / "history.json")

    n_params = models_mod.count_params(model)["total"]
    engine.save_json({
        "run_name": run_name,
        "arch": args.arch,
        "pretrained": args.pretrained,
        "augment": args.augment,
        "freeze_epochs": args.freeze_epochs,
        "backbone_lr": backbone_lr,
        "lr": args.lr,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": early.best_epoch,
        "params_total": n_params,
        "val_acc": history["val_acc"][early.best_epoch - 1],
        "val_f1_macro": history["val_f1_macro"][early.best_epoch - 1],
        "val_loss": history["val_loss"][early.best_epoch - 1],
        "test_acc": test_metrics["acc"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_loss": test_metrics["loss"],
        "train_minutes": round(total_time / 60, 2),
    }, run_dir / "metrics.json")

    engine.plot_curves(history, run_dir / "curves.png",
                       best_epoch=early.best_epoch,
                       stage_boundary=args.freeze_epochs)
    print(f"\n产物目录: {run_dir}")
    print("  best.pt / last.pt / config.json / history.json / metrics.json / curves.png")

    print("\n下一步：")
    print(f"  1. 打开 {run_dir / 'curves.png'} 看两条曲线是否开口（过拟合信号）")
    print(f"  2. 错误分析: python project_cv/src/evaluate.py "
          f"--ckpt {run_dir / 'best.pt'} --data-dir {args.data_dir}")
    print("  3. 全部实验跑完后: python project_cv/scripts/make_report.py")


if __name__ == "__main__":
    main()

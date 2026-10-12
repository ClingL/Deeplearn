# -*- coding: utf-8 -*-
r"""
第 04 课：标准训练模板（可复用，直接搬去你自己的项目）

运行（必须先跑第 03 课生成数据）：
    .\.venv\Scripts\python.exe lessons\04_train_template.py

这一课的目标不是讲新概念，而是给你一个**工程上正确的训练脚本骨架**：
    - 配置集中管理（CONFIG）
    - 固定随机种子（可复现）
    - train / val 分离，训练只用 train 更新参数
    - 每个 epoch 打印 train_loss / val_loss / val_acc / val_f1
    - 保存验证集上最好的模型（不是最后一个！）
    - 早停 EarlyStopping（验证集不涨了就停）
    - 画 loss/acc 曲线，帮你判断过拟合
    - 训练结束载入最好权重，在 test 上只评一次

这份模板你以后换个 Dataset + 换个模型就能直接用。
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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 画图用非交互后端，避免某些环境报错
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# CONFIG：所有超参数集中放这里。面试官问"你调了什么"，你能直接指出来。
# ============================================================================
CONFIG = {
    "seed": 42,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,     # L2 正则，抑制过拟合
    "epochs": 80,
    "hidden_dims": [128, 64],
    "dropout": 0.3,           # 另一个抑制过拟合的手段
    "patience": 12,           # 早停：验证 loss 连续多少轮没进步就停
    "num_workers": 0,         # Windows 上先用 0
}


def sec(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    """固定所有随机源，保证实验可复现。写论文/做实验必备。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
sec("1. 读取第 03 课处理好的数据")
# ============================================================================
NPZ = DATA_DIR / "processed.npz"
if not NPZ.exists():
    raise SystemExit(
        f"找不到 {NPZ}\n"
        "请先运行第 03 课生成数据：\n"
        "    .\\.venv\\Scripts\\python.exe lessons\\03_data_pipeline.py"
    )

d = np.load(NPZ)
X_train, y_train = d["X_train"], d["y_train"]
X_val, y_val = d["X_val"], d["y_val"]
X_test, y_test = d["X_test"], d["y_test"]

print(f"train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")
INPUT_DIM = X_train.shape[1]

# 标签是 0/1 -> 二分类。用 CrossEntropyLoss 需要 long 类型标签。
y_train_l = y_train.astype(np.int64)
y_val_l = y_val.astype(np.int64)
y_test_l = y_test.astype(np.int64)
print("标签取值:", np.unique(y_train_l), " 正例比例:", y_train_l.mean().round(4))


# ============================================================================
sec("2. Dataset / DataLoader")
# ============================================================================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


loaders = {
    "train": DataLoader(TabularDataset(X_train, y_train_l), batch_size=CONFIG["batch_size"],
                        shuffle=True, num_workers=CONFIG["num_workers"], drop_last=True),
    "val": DataLoader(TabularDataset(X_val, y_val_l), batch_size=256,
                      shuffle=False, num_workers=CONFIG["num_workers"]),
    "test": DataLoader(TabularDataset(X_test, y_test_l), batch_size=256,
                       shuffle=False, num_workers=CONFIG["num_workers"]),
}
print({k: len(v) for k, v in loaders.items()}, "  (batch 数)")


# ============================================================================
sec("3. 模型：MLP（多层感知机）")
# ============================================================================
class MLP(nn.Module):
    """通用 MLP 构造器：Linear -> BatchNorm -> ReLU -> Dropout，重复 N 次。

    关键认识：
      - Linear 负责"线性变换"，是唯一有参数的层
      - ReLU 负责非线性（没有它，堆多少层都等价于一层）
      - BatchNorm 让每层输入分布稳定 -> 允许更大学习率、收敛更快
      - Dropout 训练时随机丢神经元 -> 正则化，防过拟合
    """

    def __init__(self, in_dim, hidden_dims, n_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))   # 输出层不加激活！CrossEntropyLoss 内含 softmax
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


set_seed(CONFIG["seed"])
model = MLP(INPUT_DIM, CONFIG["hidden_dims"], n_classes=2, dropout=CONFIG["dropout"]).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"\n输入维度 {INPUT_DIM}, 总参数量 {n_params:,}")


# ============================================================================
sec("4. 损失函数 / 优化器 / 学习率调度")
# ============================================================================
criterion = nn.CrossEntropyLoss()

# AdamW 是现在最常用的默认优化器：
#   - Adam 自适应学习率，好调
#   - W 表示 decoupled weight decay，比 Adam 的 L2 更正确
# SGD + momentum 在 CV 任务里有时泛化更好，但需要调 lr
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                              weight_decay=CONFIG["weight_decay"])

# 学习率调度：训练后期降低 lr，让模型精细收敛到更好的点
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

print("criterion : nn.CrossEntropyLoss()  (内部 = LogSoftmax + NLLLoss，别自己再加 softmax)")
print("optimizer : AdamW(lr=%.0e, weight_decay=%.0e)" % (CONFIG["lr"], CONFIG["weight_decay"]))
print("scheduler : CosineAnnealingLR (学习率余弦下降)")


# ============================================================================
sec("5. 训练 / 验证 一个 epoch")
# ============================================================================
def run_epoch(model, loader, criterion, optimizer=None):
    """optimizer 为 None 就是验证模式。

    对比记住三件事：
      train: model.train() + 反向 + 更新 + 不用 no_grad
      eval : model.eval()  + 不反向      + torch.no_grad()
    因为 Dropout / BatchNorm 在两种模式下行为不同，忘了切换会得到错误的指标。
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n = 0.0, 0
    all_pred, all_true = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            logits = model(xb)                    # 1) 前向
            loss = criterion(logits, yb)          # 2) 损失

            if is_train:
                optimizer.zero_grad()             # 3) 清梯度
                loss.backward()                   # 4) 反向
                optimizer.step()                  # 5) 更新

            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            all_pred.append(logits.argmax(dim=1).cpu())
            all_true.append(yb.cpu())

    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()
    return {
        "loss": total_loss / n,
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "_pred": y_pred,
        "_true": y_true,
    }


# ============================================================================
sec("6. 完整训练循环（带早停 + 保存最好模型）")
# ============================================================================
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
best_val_loss = float("inf")
best_state = None
best_epoch = -1
epochs_no_improve = 0

print(f"{'epoch':>5} | {'lr':>8} | {'tr_loss':>8} {'tr_acc':>7} | {'va_loss':>8} {'va_acc':>7} {'va_f1':>7} | {'耗时':>6}")
print("-" * 82)

t_start = time.time()
for epoch in range(1, CONFIG["epochs"] + 1):
    t0 = time.time()
    tr = run_epoch(model, loaders["train"], criterion, optimizer)
    va = run_epoch(model, loaders["val"], criterion, optimizer=None)
    scheduler.step()
    dt = time.time() - t0

    history["train_loss"].append(tr["loss"]); history["train_acc"].append(tr["acc"])
    history["val_loss"].append(va["loss"]);   history["val_acc"].append(va["acc"])
    history["val_f1"].append(va["f1"])

    flag = ""
    # 关键：保存"验证损失最小"的模型，而不是最后一轮的
    if va["loss"] < best_val_loss - 1e-5:
        best_val_loss = va["loss"]
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_no_improve = 0
        flag = "  * best"
    else:
        epochs_no_improve += 1

    if epoch % 2 == 0 or epoch <= 3 or flag:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5} | {lr_now:>8.5f} | {tr['loss']:>8.4f} {tr['acc']:>7.4f} | "
              f"{va['loss']:>8.4f} {va['acc']:>7.4f} {va['f1']:>7.4f} | {dt:>5.1f}s{flag}")

    if epochs_no_improve >= CONFIG["patience"]:
        print(f"\n早停触发：验证 loss 连续 {CONFIG['patience']} 轮没有改善，在第 {epoch} 轮停止。")
        break

print(f"\n训练总耗时 {time.time() - t_start:.1f}s")
print(f"最好的一轮是 epoch {best_epoch}，val_loss = {best_val_loss:.4f}")

# 载入最好的权重 —— 这一步千万别忘，否则你评的是"最后一轮"的模型
model.load_state_dict(best_state)


# ============================================================================
sec("7. 测试集评估（只评一次！）")
# ============================================================================
te = run_epoch(model, loaders["test"], criterion, optimizer=None)
print(f"test loss      : {te['loss']:.4f}")
print(f"test accuracy  : {te['acc']:.4f}")
print(f"test precision : {te['precision']:.4f}")
print(f"test recall    : {te['recall']:.4f}")
print(f"test f1        : {te['f1']:.4f}")

print(f"""
为什么用 F1 而不只看 accuracy：
  如果正例只占 5%，模型全猜负例也有 95% 准确率，但一个正例都抓不到（recall=0）。
  F1 = 2*P*R/(P+R) 兼顾精确率和召回率，类别不平衡时才是真实水平。
  本数据集正例比例 {y_test_l.mean():.2%}，看一下 accuracy 和 f1 的差距。
""")

# 混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(te["_true"], te["_pred"])
print("混淆矩阵 (行=真实, 列=预测):")
print(f"{'':>12}{'pred=0':>10}{'pred=1':>10}")
print(f"{'true=0':>12}{cm[0,0]:>10}{cm[0,1]:>10}")
print(f"{'true=1':>12}{cm[1,0]:>10}{cm[1,1]:>10}")
print(f"\n漏报(ture=1 预测成 0) = {cm[1,0]}，误报(true=0 预测成 1) = {cm[0,1]}")
print("→ 业务上哪种错更贵？这决定了你要优化 precision 还是 recall。")


# ============================================================================
sec("8. 画曲线：一眼看出过拟合 / 欠拟合")
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].axvline(best_epoch - 1, ls="--", c="gray", label=f"best epoch={best_epoch}")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss")
axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["val_acc"], label="val")
axes[1].plot(history["val_f1"], label="val_f1", ls=":")
axes[1].axvline(best_epoch - 1, ls="--", c="gray")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].set_title("Accuracy / F1")
axes[1].legend(); axes[1].grid(alpha=.3)

plt.tight_layout()
curve_path = OUT_DIR / "04_training_curves.png"
plt.savefig(curve_path, dpi=110)
plt.close()
print("曲线已保存 ->", curve_path)

print("""
怎么读这张图（面试必问）：
  1) train 和 val 的 loss 都在降          -> 正常，继续训
  2) train loss 降但 val loss 开始上升     -> 过拟合。对策：加数据/增强、加 dropout、
                                             加 weight_decay、减小模型、早停
  3) 两条 loss 都很高且降不下去            -> 欠拟合。对策：加大模型、训更久、
                                             调大 lr、检查特征/标签有没有搞错
  4) loss 剧烈震荡或变 nan                 -> lr 太大 / 没有标准化 / 梯度爆炸
  5) val 指标比 train 还好                 -> 可疑！八成是数据泄漏，或者 val 太难/太小
""")


# ============================================================================
sec("9. 保存产物（部署/复现都需要）")
# ============================================================================
ckpt_path = OUT_DIR / "04_best_model.pt"
torch.save({
    "model_state": model.state_dict(),
    "config": CONFIG,
    "input_dim": INPUT_DIM,
    "best_epoch": best_epoch,
    "test_metrics": {k: v for k, v in te.items() if not k.startswith("_")},
}, ckpt_path)
print("模型已保存 ->", ckpt_path)

with open(OUT_DIR / "04_history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)
print("训练历史已保存 ->", OUT_DIR / "04_history.json")


# ============================================================================
sec("10. 推理：加载模型预测新数据")
# ============================================================================
def predict(model, X_new: np.ndarray) -> np.ndarray:
    """部署时的标准写法：eval + no_grad + 别忘 .to(DEVICE)"""
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X_new, dtype=torch.float32).to(DEVICE)
        return model(xb).argmax(dim=1).cpu().numpy()


sample = X_test[:5]
print("5 条样本预测:", predict(model, sample))
print("真实标签    :", y_test_l[:5])

print("""
注意：这里的 X_new 必须是**已经用第 03 课那个 scaler 处理过**的特征。
线上服务里最容易出的 bug 就是：训练时标准化了，推理时忘了 —— 效果直接崩。
""")


# ============================================================================
sec("11. 练习")
# ============================================================================
print("""
练习 1：把 CONFIG 里的 dropout 改成 0，weight_decay 改成 0，跑一遍。
        对比 val_loss 曲线，亲眼看到过拟合。

练习 2：把 hidden_dims 改成 [4]（极小模型），看是否欠拟合。

练习 3：把 optimizer 换成 SGD(lr=0.01, momentum=0.9)，对比收敛速度和最终指标。

练习 4：把 loss 的权重加上 pos_weight（处理类别不平衡）：
        nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(DEVICE))
        然后看 recall 怎么变。

练习 5（面试重点）：把 val 集去掉，直接用 test 调超参数，指标会怎样？
        想清楚为什么这是错的 —— 这就是"用测试集调参"的作弊。

练习 6：加一个"最佳模型保存 + 断点续训"功能：
        训练中断后能从 checkpoint 继续。
""")

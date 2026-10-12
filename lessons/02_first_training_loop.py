# -*- coding: utf-8 -*-
r"""
第 02 课：从零手写训练循环

运行：
    .\.venv\Scripts\python.exe lessons\02_first_training_loop.py

本课用最简单的问题（线性回归）把"训练"这件事彻底讲透：
    造数据 -> 定义模型 -> 定义损失 -> 梯度下降 -> 看 loss 下降

一定要看懂第 3 节（手写版）。那 5 行就是所有深度学习训练的骨架，
后面 CNN、Transformer、大模型微调，换的只是零件，骨架一模一样。
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)


def sec(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
sec("1. 造数据：真实关系是 y = 2x + 3，再加一点噪声")
# ----------------------------------------------------------------------------
# 模型的任务：只看 (x, y) 这些点，把 2 和 3 这两个"真实参数"找回来。
N = 200
TRUE_W, TRUE_B = 2.0, 3.0

x = torch.randn(N, 1)                                   # 200 个样本，每个 1 个特征
noise = 0.3 * torch.randn(N, 1)
y = TRUE_W * x + TRUE_B + noise

print("x.shape =", x.shape, " y.shape =", y.shape)
print("前 5 个样本:")
for i in range(5):
    print(f"  x={x[i].item():+.3f}  y={y[i].item():+.3f}")
print(f"\n目标：从这些点学出 w≈{TRUE_W}, b≈{TRUE_B}")

# ----------------------------------------------------------------------------
sec("2. 损失函数：怎么衡量「预测得好不好」")
# ----------------------------------------------------------------------------
# 均方误差 MSE = mean((pred - y)^2)。越小越好。

def mse(pred, target):
    return ((pred - target) ** 2).mean()

bad_pred = torch.zeros_like(y)          # 瞎猜全 0
print("瞎猜预测全 0 时，loss =", f"{mse(bad_pred, y).item():.4f}")
print("用真参数预测时，  loss =", f"{mse(TRUE_W * x + TRUE_B, y).item():.4f}", " <- 这就是下限(噪声方差)")
print("\n→ 训练的目标就是把这个数字压到接近噪声下限。")

# ----------------------------------------------------------------------------
sec("3. 【核心】手写梯度下降 —— 真正的训练循环")
# ----------------------------------------------------------------------------
# 5 步骨架，请背下来：
#   1) 前向 forward
#   2) 算损失 loss
#   3) 清梯度 zero_grad
#   4) 反向 backward
#   5) 更新 step

w = torch.zeros(1, requires_grad=True)      # 参数：初始为 0，需要梯度
b = torch.zeros(1, requires_grad=True)

LR = 0.1                                    # 学习率：每一步走多远

print(f"{'轮次':>4} | {'loss':>9} | {'w':>8} | {'b':>8}")
print("-" * 40)

for epoch in range(1, 31):
    # 1) 前向
    pred = w * x + b
    # 2) 损失
    loss = mse(pred, y)
    # 3) 清梯度（不清就会累加，见第 01 课第 6 节）
    if w.grad is not None:
        w.grad.zero_()
        b.grad.zero_()
    # 4) 反向：算出 dloss/dw 和 dloss/db
    loss.backward()
    # 5) 更新：往梯度的反方向走一小步
    with torch.no_grad():
        w -= LR * w.grad
        b -= LR * b.grad

    if epoch % 3 == 0 or epoch == 1:
        print(f"{epoch:>4} | {loss.item():>9.4f} | {w.item():>8.4f} | {b.item():>8.4f}")

print(f"\n学到的 w = {w.item():.4f} (真实 {TRUE_W})")
print(f"学到的 b = {b.item():.4f} (真实 {TRUE_B})")

print("""
观察：
  - loss 一开始很大，然后快速下降 -> 说明梯度方向是对的
  - w、b 从 0 慢慢爬向 2 和 3 -> 这就是"学习"
  
把 LR 改成 0.001 再跑：收敛会慢很多（步子太小）
把 LR 改成 1.5   再跑：loss 可能爆炸成 nan（步子太大，直接跨过最低点）
   → 这就是为什么要调学习率。这是深度学习最重要的一个超参数。
""")

# ----------------------------------------------------------------------------
sec("4. 同样的东西，用官方零件写（nn.Module + 优化器）")
# ----------------------------------------------------------------------------
# 实际项目都用这种写法。注意：逻辑和上面手写版完全一样。

model = nn.Linear(1, 1)                       # 输入 1 维，输出 1 维
criterion = nn.MSELoss()                      # 就是上面的 mse
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("初始参数:", {k: v.item() for k, v in model.state_dict().items()})

for epoch in range(1, 31):
    model.train()
    pred = model(x)                 # 1) 前向
    loss = criterion(pred, y)       # 2) 损失
    optimizer.zero_grad()           # 3) 清梯度（注意顺序：backward 之前）
    loss.backward()                 # 4) 反向
    optimizer.step()                # 5) 更新

    if epoch % 10 == 0:
        print(f"epoch {epoch:>3}  loss={loss.item():.4f}  w={model.weight.item():.4f}  b={model.bias.item():.4f}")

print("\n→ 结果和手写版一致。以后就用这个写法。")

# ----------------------------------------------------------------------------
sec("5. 加上 batch：真实数据太大，一次喂不完")
# ----------------------------------------------------------------------------
# 为什么要 batch：
#   - 数据百万级，全量算一次梯度显存装不下
#   - 小批量梯度有噪声，反而更容易跳出局部极值
# 术语：batch_size / iteration(一次参数更新) / epoch(过完一遍全部数据)
#   一个 epoch 的 iteration 数 = ceil(N / batch_size)

BATCH_SIZE = 32
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"N={N}, batch_size={BATCH_SIZE} -> 每个 epoch 有 {len(loader)} 个 iteration")

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

for epoch in range(1, 21):
    model.train()
    epoch_loss, n_batch = 0.0, 0
    for xb, yb in loader:              # xb: [32,1]  yb: [32,1]
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batch += 1
    if epoch % 5 == 0:
        print(f"epoch {epoch:>3}  平均 loss={epoch_loss / n_batch:.4f}  w={model.weight.item():.4f}  b={model.bias.item():.4f}")

print("""
注意：
  - 一个 epoch 里做了 len(loader) 次参数更新，所以收敛更快
  - shuffle=True 很重要！否则模型会学到"数据顺序"这种假规律
  - 训练时 loss 要取多个 batch 的平均，不能只看最后一个 batch
""")

# ----------------------------------------------------------------------------
sec("6. 参数、梯度长什么样（调试必看）")
# ----------------------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(DEVICE)

dummy = torch.randn(8, 10).to(DEVICE)
out = model(dummy)
loss = out.sum()
loss.backward()

print(f"{'层名':<22}{'参数形状':<18}{'梯度形状':<18}")
print("-" * 58)
for name, p in model.named_parameters():
    print(f"{name:<22}{str(tuple(p.shape)):<18}{str(tuple(p.grad.shape)):<18}")

total = sum(p.numel() for p in model.parameters())
print(f"\n总参数量 = {total:,}")

print("""
这段代码是排查问题的利器。以后遇到：
  - 梯度是 None        -> 这个参数没参与计算，或者被 no_grad 包住了
  - 梯度全 0           -> 梯度消失 / 死 ReLU / loss 与参数无关
  - 梯度特别大(>1e3)   -> 梯度爆炸，需要梯度裁剪或调小学习率
  - 参数量爆炸          -> 计算一下显存和 FLOPs
""")

# ----------------------------------------------------------------------------
sec("7. 练习")
# ----------------------------------------------------------------------------
print("""
练习 1：把 LR 依次改成 0.001 / 0.01 / 0.1 / 0.5 / 1.5，记录 loss 什么时候最小，
        什么时候发散。这是你第一次"调参"。

练习 2：把噪声 0.3 改成 0，训练能不能把 w、b 精确学到 2 和 3？

练习 3：把模型改成 y = w2*x^2 + w1*x + b（二次），
        提示：x 要变成 [x, x**2]，或者用 nn.Sequential 加层。

练习 4：把 loss 从 MSE 换成 L1Loss（nn.L1Loss），结果有什么区别？

练习 5（思考）：为什么 optimizer.zero_grad() 要放在 backward() 之前、
        而不是 step() 之后？两种写法有什么区别？
""")

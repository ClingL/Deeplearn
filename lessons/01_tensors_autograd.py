# -*- coding: utf-8 -*-
r"""
第 01 课：张量 & 自动求导（深度学习的地基）

运行：
    .\.venv\Scripts\python.exe lessons\01_tensors_autograd.py

学完你应该能回答：
  - shape / dtype / device 分别是什么，为什么 90% 的报错都出在这三样
  - 广播(broadcasting)规则是什么
  - requires_grad / backward / grad 三者的关系
  - 为什么必须 zero_grad()，为什么验证时要 no_grad()

这一课不训练任何模型，但这一课不懂，后面全是抄代码。
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


def sec(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
sec("0. 环境")
# ----------------------------------------------------------------------------
print(f"torch 版本 : {torch.__version__}")
print(f"使用设备   : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"显存       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ----------------------------------------------------------------------------
sec("1. 张量三件套：shape / dtype / device")
# ----------------------------------------------------------------------------
# 张量 = 多维数组 + 数据类型 + 所在设备。
# 神经网络里，一次前向就是一堆张量运算；报错几乎都是"形状对不上"或"设备不一致"。

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print("x =\n", x)
print("shape :", x.shape)        # torch.Size([2, 3]) -> 2 行 3 列
print("dtype :", x.dtype)        # torch.float32
print("device:", x.device)       # cpu

# 常用构造
print("\ntorch.zeros(2,3)      :", torch.zeros(2, 3).shape)
print("torch.ones(2,3)       :", torch.ones(2, 3).shape)
print("torch.randn(2,3)      : 标准正态 N(0,1)")
print("torch.arange(5)       :", torch.arange(5))
print("torch.eye(3)          : 3x3 单位阵")

# dtype：深度学习默认 float32；索引必须是 int64；标签常用 int64
print("\n整数张量 dtype      :", torch.tensor([1, 2, 3]).dtype)      # int64
print("浮点张量 dtype      :", torch.tensor([1.0, 2.0]).dtype)      # float32
print("类型转换 x.float()/x.long()/x.to(torch.float64)")
print("x.to(torch.float64):", x.to(torch.float64).dtype)

# device：模型和数据必须在同一个设备上，否则报 Expected all tensors to be on the same device
x_gpu = x.to(DEVICE)
print(f"\nx.to('{DEVICE}') device =", x_gpu.device)
print("→ 规律：数据用 .to(DEVICE)，模型用 model.to(DEVICE)，两边都要做")

# ----------------------------------------------------------------------------
sec("2. 形状变换：view / reshape / permute / squeeze / unsqueeze")
# ----------------------------------------------------------------------------
a = torch.arange(12)
print("a            :", a, "shape", a.shape)

print("\na.view(3, 4)  :\n", a.view(3, 4))
print("→ view 要求内存连续；不确定时用 reshape，更安全")

print("\na.reshape(2, 2, 3).shape =", a.reshape(2, 2, 3).shape)

img = torch.randn(3, 32, 32)          # 一张图：3 通道 × 32 × 32
print("\n单张图 img.shape          :", img.shape)
print("加 batch 维 unsqueeze(0)  :", img.unsqueeze(0).shape)   # -> [1,3,32,32]
batch = torch.randn(8, 3, 32, 32)
print("一个 batch 8 张          :", batch.shape)
print("去掉 batch 维 squeeze(0) :", torch.randn(1, 3, 32, 32).squeeze(0).shape)

# permute：换轴顺序。图像 HWC -> CHW 常用
hwc = torch.randn(32, 32, 3)
print("\nHWC", hwc.shape, "-> CHW", hwc.permute(2, 0, 1).shape)

# -1 = 自动推断那一维
print("\na.reshape(4, -1).shape =", a.reshape(4, -1).shape)

# ----------------------------------------------------------------------------
sec("3. 广播 (broadcasting)：小张量自动扩展到大的")
# ----------------------------------------------------------------------------
# 规则：从最后一维往前对齐，每一维要么相等、要么其中一个是 1、要么其中一个不存在。
A = torch.randn(4, 3)
b = torch.randn(3)        # 会被广播成 [4,3]
c = torch.randn(4, 1)     # 会被广播成 [4,3]

print("A.shape      :", A.shape)
print("(A + b).shape:", (A + b).shape)
print("(A * c).shape:", (A * c).shape)

# 经典用途：给一批图片的每个通道做归一化
batch = torch.randn(8, 3, 32, 32)
mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)   # 手动对齐成 4 维才好广播
print("\n归一化 (batch - mean).shape:", (batch - mean).shape)

print("\n会报错的例子：torch.randn(4,3) + torch.randn(4,)")
print("→ 最后一位 3 vs 4，既不等也不是 1，广播失败")

# ----------------------------------------------------------------------------
sec("4. 矩阵乘与聚合（dim 到底消掉哪一维）")
# ----------------------------------------------------------------------------
A = torch.randn(4, 3)
B = torch.randn(3, 5)
print("A@B.shape =", (A @ B).shape, " (4,3)@(3,5) -> (4,5)：中间那维必须相等且被消掉")

# dim 的含义：dim=k 表示"把第 k 维压掉"
v = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print("\nv =\n", v)
print("v.sum()          =", v.sum(),   " 全部求和")
print("v.sum(dim=0)     =", v.sum(dim=0), " 压掉第0维(行) -> 得到 3 个数，按列求和")
print("v.sum(dim=1)     =", v.sum(dim=1), " 压掉第1维(列) -> 得到 2 个数，按行求和")
print("v.mean(dim=0)    =", v.mean(dim=0))
print("v.max(dim=1)     =", v.max(dim=1), " 返回 (值, 索引)")

# 最常用：分类任务里取每个样本的最大概率对应的类别
logits = torch.randn(4, 10)          # 4 个样本，10 分类
pred = logits.argmax(dim=1)
print("\nlogits.shape", logits.shape, "-> argmax(dim=1) =", pred, " shape", pred.shape)

# ----------------------------------------------------------------------------
sec("5. 索引、掩码、拼接")
# ----------------------------------------------------------------------------
x = torch.arange(10)
print("x[2:5]      :", x[2:5])
print("x[x % 2 == 0]:", x[x % 2 == 0], " <- 布尔掩码，常用于按条件筛选/置零")
print("torch.cat([x, x]).shape :", torch.cat([x, x]).shape, " 沿已有维度拼")
print("torch.stack([x, x]).shape:", torch.stack([x, x]).shape, " 新建一维堆叠")

# ----------------------------------------------------------------------------
sec("6. 自动求导 autograd —— 深度学习的核心引擎")
# ----------------------------------------------------------------------------
# 你只要定义前向计算，PyTorch 自动记录计算图，backward() 就能算出所有参数的梯度。

w = torch.tensor([2.0], requires_grad=True)   # 需要求梯度的东西，加 requires_grad=True
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])

y = w * x + b          # 前向
print("y =", y, " grad_fn =", y.grad_fn, " <- 说明它是计算图里的节点")
print("y.requires_grad =", y.requires_grad)

y.backward()           # 反向：求 dy/dw 和 dy/db

print("\nw.grad =", w.grad, "  (dy/dw = x = 3)")
print("b.grad =", b.grad, "  (dy/db = 1)")

# 手动验证：y = w*x + b，dy/dw = x = 3，dy/db = 1  v
print("→ 梯度就是'这个参数变大一点点，y 会变多少'。梯度下降就是往反方向走。")

# 多条路径会累加
w2 = torch.tensor([1.0], requires_grad=True)
loss = (w2 * 2).sum() + (w2 * 3).sum()
loss.backward()
print("\nw2.grad =", w2.grad, "  因为 2 + 3 = 5（两条路径的梯度相加）")

# 梯度会累加！所以每轮必须清零
print("\n反复 backward 会累加（这就是必须 zero_grad 的原因）：")
w3 = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    (w3 * 2).backward()
    print(f"  第{i+1}次 backward 后 w3.grad = {w3.grad}")
print("  → 手动清零：w3.grad.zero_()，或优化器 optimizer.zero_grad()")

# ----------------------------------------------------------------------------
sec("7. detach 与 no_grad：切断梯度")
# ----------------------------------------------------------------------------
a = torch.tensor([2.0], requires_grad=True)
b = a * 3
print("b.requires_grad            =", b.requires_grad)
print("b.detach().requires_grad   =", b.detach().requires_grad, " <- 从计算图里摘出来，当常量用")

with torch.no_grad():
    c = a * 3
print("no_grad 下 c.requires_grad =", c.requires_grad)

print("""
用途总结：
  - 验证/推理时用 torch.no_grad()：省显存、更快（不建计算图）
  - 记录指标时用 .item() 或 .detach()：避免把整个计算图留在内存里，导致显存泄漏
  - 对抗训练/EMA/target network 时需要 detach 当常量
""")

# 正确做法演示
total = 0.0
for i in range(5):
    loss = (torch.randn(1, requires_grad=True) ** 2).sum()
    total += loss.item()          # .item() 取出 python 数字，不留在图里
print("累加 loss 正确姿势 total =", round(total, 4))

# ----------------------------------------------------------------------------
sec("8. 练习（改代码再跑，别只看）")
# ----------------------------------------------------------------------------
print("""
练习 1：造一个 shape (16, 3, 32, 32) 的张量，求每个通道(第1维)的均值，让结果 shape 是 (3,)。
        提示：mean(dim=(0,2,3))

练习 2：A=(5,4) B=(4,2)，用 @ 相乘，输出形状是多少？（先猜再跑）

练习 3：x = torch.tensor([1.,2.,3.], requires_grad=True)
        y = (x**2).sum()，然后 y.backward()，x.grad 是多少？为什么？

练习 4：把本文件第 6 节的例子改成 y = w*x + b，x 是 shape (4,3) 的矩阵，
        w 是 (3,)，b 是 (1,) —— 体会广播 + 求导一起工作的感觉。

练习 5（重要）：下面会报什么错？为什么？怎么改？
        model 在 cuda 上，数据在 cpu 上。
""")

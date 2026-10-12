# -*- coding: utf-8 -*-
r"""
第 05 课：CNN 图像分类（CIFAR-10）

运行：
    .\.venv\Scripts\python.exe lessons\05_cnn_images.py

第一次运行会自动下载 CIFAR-10 到 data/（约 170MB）。
你的 5070 跑这个大概 1-2 分钟一个 epoch（默认 12 轮）。

学完你应该能回答：
  - 为什么图像不用 MLP 而用卷积？
  - 卷积核在干嘛？输出尺寸怎么算？
  - 池化有什么用？
  - 数据增强为什么能防过拟合？
  - 图像数据的 shape 约定是什么（NCHW）
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

import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def sec(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


sec("0. 环境")
print("device:", DEVICE)
print("torchvision:", torchvision.__version__)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ----------------------------------------------------------------------------
sec("1. 为什么图像不用 MLP？先算一笔账")
# ----------------------------------------------------------------------------
print("""
CIFAR-10 一张图是 3 x 32 x 32 = 3072 个数。

如果用 MLP，第一层就 512 个神经元：
    参数量 = 3072 * 512 = 1,572,864

而且这只是"一件事"：MLP 把图片拉平，**丢掉了空间结构**。
把猫的图片整体平移 3 个像素，对 MLP 来说是完全不同的输入 —— 荒谬。

卷积的两个核心思想（面试必答）：
  1. 局部连接 + 权值共享：一个 3x3 的卷积核在整张图上滑动，参数量只有 9*C。
     -> 参数量从百万级降到几千，且天然具备平移不变性
  2. 层次化特征：浅层学边缘/颜色，中层学纹理/部件，深层学物体
     这也是"深度学习"能work的关键
""")

W, H, C = 32, 32, 3
print(f"MLP 第一层参数量        : {W*H*C*512:,}")
print(f"一个 3x3 卷积(C=3->64)   : {3*3*3*64 + 64:,}")

# ----------------------------------------------------------------------------
sec("2. 卷积输出尺寸怎么算（手算一遍就懂了）")
# ----------------------------------------------------------------------------
print("""
公式：  H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1

记住三个直觉：
  - kernel_size=3, padding=1, stride=1  -> 尺寸不变
  - stride=2                            -> 尺寸大约减半
  - MaxPool2d(2)                        -> 尺寸精确减半
""")

x = torch.randn(1, 3, 32, 32)
print("输入:", tuple(x.shape))

conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
print("Conv2d(3,64,k=3,p=1)  ->", tuple(conv(x).shape), "  空间尺寸不变，通道 3->64")

conv_s2 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)
print("Conv2d(...,stride=2)  ->", tuple(conv_s2(x).shape), "  尺寸减半")

pool = nn.MaxPool2d(2)
print("MaxPool2d(2)          ->", tuple(pool(x).shape), "  尺寸减半")

print("\n完整算一遍（本课用的网络结构）:")
print(f"  输入                 : 3 x 32 x 32")
print(f"  conv1 k3 p1          : 32 x 32 x 32    (32)  -> maxpool -> 16")
print(f"  conv2 k3 p1          : 64 x 16 x 16    (64)  -> maxpool -> 8")
print(f"  conv3 k3 p1          : 128 x 8 x 8     (128) -> maxpool -> 4")
print(f"  全局平均池化 (GAP)   : 128 x 1 x 1     <- 把 4x4 压成 1x1")
print(f"  全连接               : 128 -> 10")

# ----------------------------------------------------------------------------
sec("3. 数据：transforms + DataLoader")
# ----------------------------------------------------------------------------
# 图像数据的三步标准处理：
#   ToTensor()      : PIL 图片(HWC, 0-255) -> 张量(CHW, 0-1)
#   Normalize(mean, std) : (x - mean) / std，用数据集统计出来的值
#   随机增强         : 只在训练时用！验证/测试绝对不能用随机增强

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 随机裁剪（先补边再裁）
    transforms.RandomHorizontalFlip(),          # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

test_tf = transforms.Compose([                  # 验证/测试：只做确定性变换
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

print("训练增强: RandomCrop(32,pad=4) + RandomHorizontalFlip + ToTensor + Normalize")
print("测试变换: ToTensor + Normalize   <- 不能有随机性！")

train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
test_set = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
print(f"\ntrain {len(train_set)} 张   test {len(test_set)} 张")
print("一张图的张量形状:", tuple(train_set[0][0].shape), " <- CHW (通道在前)")
print("图片是 PIL 对象时形状是 HWC，ToTensor 会自动换成 CHW，别搞混")

BATCH_SIZE = 128

# num_workers 说明（Windows 上很容易踩的坑）：
#   在 Windows / macOS 上，num_workers > 0 会真正 fork 子进程，
#   而子进程会重新 import 当前脚本。如果脚本没有 if __name__ == "__main__": 保护，
#   就会无限递归启动子进程，报 "An attempt has been made to start a new process..."。
#   本课为了方便阅读是平铺写的，所以先用 num_workers=0。
#   想加速的话，把下面的逻辑包进 main() 并用 if __name__ == "__main__": main() 调用，
#   然后把 num_workers 设成 4 左右（经验值：CPU 核数的一半）。
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                         num_workers=0, pin_memory=True)

xb, yb = next(iter(train_loader))
print(f"\n一个 batch: xb {tuple(xb.shape)}  yb {tuple(yb.shape)}")
print("注意 batch 里的图每次都不一样（因为增强了）")

# ----------------------------------------------------------------------------
sec("4. 看一眼数据（存成图片，打开 outputs/05_samples.png 看）")
# ----------------------------------------------------------------------------
def denormalize(t):
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)

fig, axes = plt.subplots(5, 10, figsize=(14, 7))
# 同一张图增强 10 次，看增强效果
raw_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=False,
                                       transform=transforms.ToTensor())
for r in range(5):
    img, label = raw_set[r]
    for c in range(10):
        aug, _ = train_set[r]      # 每次调用 __getitem__ 都会重新随机增强
        axes[r, c].imshow(denormalize(aug).permute(1, 2, 0).numpy())
        axes[r, c].axis("off")
        if c == 0:
            axes[r, c].set_ylabel(CLASSES[label])
axes[0, 0].set_title("同一张图的 10 次随机增强", loc="left")
plt.tight_layout()
p = OUT_DIR / "05_samples.png"
plt.savefig(p, dpi=110)
plt.close()
print("已保存 ->", p, "（打开看看增强长什么样）")

# 类别分布是否均衡
cnt = Counter(train_set.targets)
print("\n训练集类别分布:", {CLASSES[k]: cnt[k] for k in sorted(cnt)})
print("每类一样多 -> 不需要做类别重采样/加权")

# ----------------------------------------------------------------------------
sec("5. 定义 CNN")
# ----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU，重复两次，再 MaxPool。这是 CNN 的标准积木。"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.max_pool2d(x, 2)


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.block1(x)          # -> 32 x 16 x 16
        x = self.block2(x)          # -> 64 x 8 x 8
        x = self.block3(x)          # -> 128 x 4 x 4
        x = F.adaptive_avg_pool2d(x, 1)   # 全局平均池化 -> 128 x 1 x 1
        x = torch.flatten(x, 1)           # -> 128
        return self.head(x)               # -> 10


model = SmallCNN().to(DEVICE)
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量 {n_params:,}  —— 对比一下刚才 MLP 单层的 157 万")

print("""
两个设计要点：
  1. 卷积层都用 bias=False：因为后面紧跟 BatchNorm，BN 有自己的偏置，bias 冗余
  2. 用全局平均池化(GAP) 代替 Flatten+大 FC：
     - 参数量大幅减少（Flatten 到 128*4*4=2048 再连 512 要 100 万参数）
     - 对不同输入尺寸更鲁棒，抗过拟合
     这个技巧在 ResNet / 现代网络里都是标配
""")

# ----------------------------------------------------------------------------
sec("6. 训练")
# ----------------------------------------------------------------------------
EPOCHS = 12
LR = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, total_steps=EPOCHS * len(train_loader), pct_start=0.3
)

print(f"epochs={EPOCHS}  lr={LR}  optimizer=AdamW  scheduler=OneCycleLR(max_lr=3e-3)")
print("OneCycleLR：前 30% 把 lr 从很小升到 max_lr，后面余弦降下来。对大 lr 很鲁棒。\n")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    per_class_correct = torch.zeros(10)
    per_class_total = torch.zeros(10)
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * xb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
        for c in range(10):
            m = yb == c
            per_class_total[c] += m.sum().item()
            per_class_correct[c] += (pred[m] == c).sum().item()
    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "per_class_acc": (per_class_correct / per_class_total.clamp(min=1)).tolist(),
    }


history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}
best_acc, best_state = 0.0, None
t_all = time.time()

print(f"{'epoch':>5} | {'lr':>8} | {'tr_loss':>8} {'tr_acc':>7} | {'te_loss':>8} {'te_acc':>7} | {'耗时':>6}")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    run_loss, run_correct, run_n = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad(set_to_none=True)   # set_to_none=True 比填 0 更省内存
        loss.backward()
        optimizer.step()
        scheduler.step()

        run_loss += loss.item() * xb.size(0)
        run_correct += (logits.argmax(1) == yb).sum().item()
        run_n += xb.size(0)

    tr_loss, tr_acc = run_loss / run_n, run_correct / run_n
    te = evaluate(model, test_loader)
    dt = time.time() - t0

    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["test_loss"].append(te["loss"]); history["test_acc"].append(te["acc"])
    history["lr"].append(optimizer.param_groups[0]["lr"])

    flag = ""
    if te["acc"] > best_acc:
        best_acc = te["acc"]
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        flag = "  * best"

    print(f"{epoch:>5} | {optimizer.param_groups[0]['lr']:>8.5f} | {tr_loss:>8.4f} {tr_acc:>7.4f} | "
          f"{te['loss']:>8.4f} {te['acc']:>7.4f} | {dt:>5.1f}s{flag}")

print(f"\n总耗时 {time.time() - t_all:.1f}s")
print(f"最好测试准确率 {best_acc:.4f}")

model.load_state_dict(best_state)      # 载入最好的
final = evaluate(model, test_loader)

# ----------------------------------------------------------------------------
sec("7. 结果分析：看每个类别的准确率")
# ----------------------------------------------------------------------------
print(f"整体准确率: {final['acc']:.4f}\n")
print(f"{'类别':<10}{'准确率':>9}   {'条形图'}")
print("-" * 55)
order = sorted(range(10), key=lambda i: final["per_class_acc"][i])
for i in order:
    a = final["per_class_acc"][i]
    bar = "#" * int(a * 40)
    print(f"{CLASSES[i]:<10}{a:>9.4f}   {bar}")

print("""
观察规律（这也是面试会问的）：
  - 猫/狗 准确率通常最低 -> 它们长得像，而且 CIFAR 分辨率只有 32x32
  - 车/船/飞机 通常最高   -> 形状规整、背景干净
  -> 说明模型的错误是"有结构"的，不是随机的。分析 per-class 指标
     比只看一个总 accuracy 有价值得多。
""")

# 混淆矩阵
@torch.no_grad()
def get_confusion(model, loader):
    model.eval()
    cm = np.zeros((10, 10), dtype=int)
    for xb, yb in loader:
        pred = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
        for t, p in zip(yb.numpy(), pred):
            cm[t, p] += 1
    return cm


cm = get_confusion(model, test_loader)
print("最容易混淆的 5 对 (真实 -> 预测, 次数):")
pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            pairs.append((cm[i, j], CLASSES[i], CLASSES[j]))
pairs.sort(reverse=True)
for n, a, b in pairs[:5]:
    print(f"  {a:>6} -> {b:<6} {n:>4} 次")

# ----------------------------------------------------------------------------
sec("8. 保存曲线和模型")
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["test_loss"], label="test")
axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["test_acc"], label="test")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=.3)
axes[2].plot(history["lr"], c="orange"); axes[2].set_title("Learning Rate (OneCycle)")
axes[2].set_xlabel("epoch"); axes[2].grid(alpha=.3)
plt.tight_layout()
p = OUT_DIR / "05_cnn_curves.png"
plt.savefig(p, dpi=110)
plt.close()
print("曲线 ->", p)

p = OUT_DIR / "05_cnn_best.pt"
torch.save({"model_state": model.state_dict(), "classes": CLASSES,
            "best_acc": best_acc, "per_class_acc": final["per_class_acc"]}, p)
print("模型 ->", p)

with open(OUT_DIR / "05_history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

# ----------------------------------------------------------------------------
sec("9. 练习（这才是真正学东西的地方）")
# ----------------------------------------------------------------------------
print("""
练习 1（必做，最能提升）：把 RandomCrop 和 RandomHorizontalFlip 删掉，
        跑到 12 epoch，对比 test_acc。你会看到掉好几个点 —— 这就是数据增强的价值。

练习 2：把 SmallCNN 换成 ResNet18（torchvision.models.resnet18(num_classes=10)），
        看准确率能到多少。你会发现：架构远比你自己瞎调参重要。

练习 3：把 EPOCHS 调到 50，看能不能到 90%+。

练习 4：把 GAP 换成 Flatten + nn.Linear(128*4*4, 10)，参数量涨多少？准确率变好吗？

练习 5：可视化第一层卷积核（model.block1.conv1.weight），
        提示 weight shape 是 [32, 3, 3, 3]，把 3x3x3 当成小图存出来。

练习 6（进阶）：把 BN 全删掉，或者把 BN 换成 GroupNorm，对比结果。
        体会 BN 为什么几乎必加。

练习 7（思考题）：为什么验证/测试时不能用 RandomHorizontalFlip？
        如果用了，指标会偏高还是偏低？
""")

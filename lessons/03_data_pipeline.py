# -*- coding: utf-8 -*-
"""
第 03 课：数据处理全流程（实习干活 70% 的时间都在这）

运行：
    .\.venv\Scripts\python.exe lessons\03_data_pipeline.py

面试高频、工作高频。本课走一遍真实流程：
    原始 CSV（有缺失/有类别/有离群值）
      -> pandas 读入 & 体检
      -> 清洗（缺失、异常）
      -> 划分 train/val/test（分层）
      -> 编码（类别 -> 数字，标准化数值）
      -> Dataset / DataLoader
      -> 取出一个 batch 检查形状

【最重要的一句话】任何统计量（均值/方差/编码表/最大最小值）都只能
用训练集 fit，然后 transform 验证集和测试集。否则就是数据泄漏，指标全假。
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

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sec(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
RAW = DATA_DIR / "raw_churn.csv"

# ----------------------------------------------------------------------------
sec("1. 造一份'脏'数据（真实数据都长这样）")
# ----------------------------------------------------------------------------
# 模拟一个"客户流失预测"数据集：数值特征 + 类别特征 + 缺失值 + 离群值。
rng = np.random.default_rng(0)
N = 2000

df = pd.DataFrame({
    "age":          rng.integers(18, 70, N).astype(float),
    "monthly_fee":  rng.normal(80, 25, N).round(2),
    "tenure_month": rng.integers(1, 72, N).astype(float),
    "city":         rng.choice(["北京", "上海", "广州", "深圳"], N),
    "plan":         rng.choice(["basic", "plus", "premium"], N),
})

# ---------------------------------------------------------------------------
# 【关键】标签必须由特征决定，否则模型什么都学不到。
# 如果这里写 churn = rng.integers(0,1,N)（纯随机），那么后面第 04 课无论怎么训，
# 准确率都会卡在 50% 左右 —— 因为数据里根本没有可学的规律。
# 这叫"标签和特征独立"，是新手最容易踩的坑之一。
#
# 我们编一个符合业务直觉的规律：
#   月费越高        -> 越可能流失（贵）
#   在网时间越长    -> 越不容易流失（有黏性）
#   basic 套餐      -> 更容易流失；premium -> 更忠诚
#   一线城市        -> 竞争激烈，略容易流失
# ---------------------------------------------------------------------------
logit = (
    1.1 * (df["monthly_fee"] - 80) / 25
    - 1.2 * (df["tenure_month"] - 36) / 21
    + np.where(df["plan"] == "basic", 0.9,
               np.where(df["plan"] == "premium", -0.9, 0.0))
    + np.where(df["city"].isin(["北京", "上海"]), 0.4, -0.4)
    - 0.2
)
prob = 1.0 / (1.0 + np.exp(-logit))
# 用概率采样而不是硬阈值：保留真实数据里「同样的客户，有的流失有的没流失」的噪声。
# 这也是为什么模型不可能做到 100% 准确 —— 真实数据一定有噪声。
df["churn"] = (rng.random(N) < prob).astype(int)

print("标签生成规律（= 模型要学的信号）:")
print(f"  流失(正例)比例          : {df['churn'].mean():.3f}")
print(f"  月费 与 流失 相关性     : {df['monthly_fee'].corr(df['churn']):+.3f}   (应为正)")
print(f"  在网时长 与 流失 相关性 : {df['tenure_month'].corr(df['churn']):+.3f}   (应为负)")
print(f"  各套餐流失率            : {df.groupby('plan')['churn'].mean().round(3).to_dict()}")
print("  -> 第 04 课跑完，如果模型准确率明显高于正例比例，说明它真的学到了规律。")
print()

# 人为注入脏数据
df.loc[rng.choice(N, 120, replace=False), "age"] = np.nan              # 缺失
df.loc[rng.choice(N, 60, replace=False), "monthly_fee"] = np.nan       # 缺失
df.loc[rng.choice(N, 15, replace=False), "monthly_fee"] = 9999.0       # 离群值
df = pd.concat([df, df.iloc[:20]], ignore_index=True)                  # 重复行

df.to_csv(RAW, index=False, encoding="utf-8-sig")
print(f"已生成 {RAW}  shape={df.shape}")

# ----------------------------------------------------------------------------
sec("2. pandas 读入 + 体检（拿到数据先做这 5 件事）")
# ----------------------------------------------------------------------------
df = pd.read_csv(RAW, encoding="utf-8-sig")

print("--- df.head() ---")
print(df.head())

print("\n--- df.shape ---", df.shape)

print("\n--- df.info() ---")
df.info()

print("\n--- df.describe() 数值列统计 ---")
print(df.describe().T)

print("\n--- 缺失值 ---")
print(df.isna().sum())

print("\n--- 重复行数 ---", df.duplicated().sum())

print("\n--- 类别列的取值分布 ---")
for c in ["city", "plan"]:
    print(f"{c}: {df[c].value_counts().to_dict()}")

print("\n--- 标签分布（分类任务必看，判断是否类别不平衡）---")
print(df["churn"].value_counts(normalize=True).round(4).to_dict())

print("""
拿到任何数据集的固定动作：
  shape -> info -> describe -> 缺失 -> 重复 -> 类别分布 -> 标签分布
这 7 步能让你在 5 分钟内判断数据能不能用、坑在哪。
""")

# ----------------------------------------------------------------------------
sec("3. 清洗：重复 / 缺失 / 离群")
# ----------------------------------------------------------------------------
print("清洗前:", df.shape)

# 3.1 去重
df = df.drop_duplicates().reset_index(drop=True)
print("去重后:", df.shape)

# 3.2 离群值：monthly_fee 出现 9999，明显是异常录入
#     方法一：分位数裁剪（推荐，稳）
q1, q99 = df["monthly_fee"].quantile([0.01, 0.99])
print(f"\nmonthly_fee 的 1%/99% 分位: {q1:.2f} / {q99:.2f}")
before = df["monthly_fee"].max()
df["monthly_fee"] = df["monthly_fee"].clip(q1, q99)
print(f"裁剪前最大值 {before:.1f} -> 裁剪后 {df['monthly_fee'].max():.2f}")

# 3.3 缺失值
#     数值列 -> 中位数填充（比均值抗离群）
#     类别列 -> 众数填充，或单独当一类 "Unknown"
#     缺失特别多的列(>50%)可以考虑直接删掉
num_cols = ["age", "monthly_fee", "tenure_month"]
cat_cols = ["city", "plan"]

for c in num_cols:
    med = df[c].median()
    n_miss = df[c].isna().sum()
    df[c] = df[c].fillna(med)
    print(f"{c:>13}: 填充 {n_miss:>4} 个缺失值 -> 中位数 {med:.2f}")

for c in cat_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode()[0])

print("\n清洗后缺失值总数:", df.isna().sum().sum())
print("清洗后 shape:", df.shape)

# ----------------------------------------------------------------------------
sec("4. 划分数据集：先划分，再做任何统计！")
# ----------------------------------------------------------------------------
# 顺序极其重要：
#   正确: 先 split，再在 train 上 fit scaler -> transform val/test
#   错误: 全量 fit scaler 再 split  <- 数据泄漏！验证集信息漏进训练了
#
# stratify=y 保证训练/验证/测试集里正负样本比例一致（分类任务必加）

X = df[num_cols + cat_cols].copy()
y = df["churn"].values

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)

print(f"train: {len(X_train):>5} 条  正例比例 {y_train.mean():.4f}")
print(f"val  : {len(X_val):>5} 条  正例比例 {y_val.mean():.4f}")
print(f"test : {len(X_test):>5} 条  正例比例 {y_test.mean():.4f}")
print("\n三份比例接近 -> stratify 生效")

# ----------------------------------------------------------------------------
sec("5. 特征工程：数值标准化 + 类别编码")
# ----------------------------------------------------------------------------
# 5.1 数值特征：StandardScaler -> 均值0方差1
#     为什么必须做：不同特征量纲差太多（age~50, fee~80, tenure~36），
#     不标准化的话梯度方向会被大尺度特征主导，收敛慢甚至不收敛。
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])     # 只用 train fit！
X_val_num = scaler.transform(X_val[num_cols])             # 用 train 的统计量
X_test_num = scaler.transform(X_test[num_cols])

print("scaler 学到的均值:", dict(zip(num_cols, scaler.mean_.round(3))))
print("scaler 学到的标准差:", dict(zip(num_cols, scaler.scale_.round(3))))
print("train 标准化后均值:", X_train_num.mean(axis=0).round(6))
print("train 标准化后方差:", X_train_num.var(axis=0).round(6))
print("val 标准化后均值(不一定是0):", X_val_num.mean(axis=0).round(4), " <- 正常，因为用的是 train 的统计量")

# 5.2 类别特征 -> 数字
#     方案A: 序数编码（用 pandas category，写死顺序）适合有序类别(低/中/高)
#     方案B: 独热编码 One-Hot 适合无序类别(城市) —— 类别少时用
#     方案C: Embedding 适合高基数类别(用户ID) —— 后面课讲
#
# 这里用独热，注意必须对齐列（训练集出现的类别，验证集不一定全有）
def one_hot(df_part, categories):
    out = pd.DataFrame(index=df_part.index)
    for col, cats in categories.items():
        for cat in cats:
            out[f"{col}_{cat}"] = (df_part[col] == cat).astype(np.float32)
    return out

# 类别表从 train 里学，保证 train/val/test 列完全一致
categories = {c: sorted(X_train[c].unique()) for c in cat_cols}
print("\n从 train 学到的类别表:", categories)

cat_train = one_hot(X_train, categories).values
cat_val = one_hot(X_val, categories).values
cat_test = one_hot(X_test, categories).values

X_train_np = np.hstack([X_train_num, cat_train]).astype(np.float32)
X_val_np = np.hstack([X_val_num, cat_val]).astype(np.float32)
X_test_np = np.hstack([X_test_num, cat_test]).astype(np.float32)

print("最终特征形状 train/val/test:", X_train_np.shape, X_val_np.shape, X_test_np.shape)

# ----------------------------------------------------------------------------
sec("6. 封装成 Dataset + DataLoader")
# ----------------------------------------------------------------------------
# 两种写法：
#   A. TensorDataset(x_tensor, y_tensor) —— 数据已经是张量时最省事
#   B. 自定义 Dataset —— 数据是图片路径/超大文件/需要在线增强时必须用

# --- 写法 A：TensorDataset ---
ds_train = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train_np),
    torch.from_numpy(y_train).float().unsqueeze(1),   # [N] -> [N,1] 对齐输出形状
)
print("写法A TensorDataset: len =", len(ds_train), " 第一条 x.shape =", ds_train[0][0].shape)


# --- 写法 B：自定义 Dataset（重点掌握，工作中都用这个）---
class TabularDataset(Dataset):
    """把 numpy 特征/标签包成一个 Dataset。
    必须实现 __len__ 和 __getitem__。
    __getitem__ 返回 (特征, 标签) 一条样本，DataLoader 会帮你拼成 batch。
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


ds_train = TabularDataset(X_train_np, y_train)
ds_val = TabularDataset(X_val_np, y_val)
ds_test = TabularDataset(X_test_np, y_test)

BATCH_SIZE = 64
loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nbatch_size={BATCH_SIZE}")
print(f"train: {len(ds_train)} 条 -> {len(loader_train)} 个 batch")
print(f"val  : {len(ds_val)} 条 -> {len(loader_val)} 个 batch")
print(f"test : {len(ds_test)} 条 -> {len(loader_test)} 个 batch")

xb, yb = next(iter(loader_train))
print(f"\n一个 batch: xb.shape={tuple(xb.shape)}  yb.shape={tuple(yb.shape)}  dtype={xb.dtype}")
print("yb 前 10 个标签:", yb[:10].flatten().tolist())
print("最后一个 batch 会比较小（无法整除）:", next(iter(DataLoader(ds_train, batch_size=64)))[0].shape)

print("""
Dataset / DataLoader 记住三件事：
  1. Dataset 负责"怎么取第 i 条"，DataLoader 负责"怎么拼成 batch、打乱、并行加载"
  2. 训练集 shuffle=True，验证/测试集必须 shuffle=False（不然指标会飘）
  3. Windows 上 num_workers>0 可能报错，先用 0，稳了再调大
""")

# ----------------------------------------------------------------------------
sec("7. 收尾：把处理好的数据存下来，避免每次重跑")
# ----------------------------------------------------------------------------
np.savez_compressed(
    DATA_DIR / "processed.npz",
    X_train=X_train_np, y_train=y_train,
    X_val=X_val_np, y_val=y_val,
    X_test=X_test_np, y_test=y_test,
)
print("已保存 ->", DATA_DIR / "processed.npz")

import joblib
joblib.dump({"scaler": scaler, "categories": categories, "num_cols": num_cols,
             "cat_cols": cat_cols, "feature_dim": X_train_np.shape[1]},
            DATA_DIR / "preprocess.pkl")
print("已保存 ->", DATA_DIR / "preprocess.pkl")
print("""
为什么要存：推理时新来一条数据，必须用和训练时完全一样的 scaler 和编码表处理，
          否则线上线下不一致，效果直接崩。这是工业界的常识。
""")

# ----------------------------------------------------------------------------
sec("8. 练习")
# ----------------------------------------------------------------------------
print("""
练习 1：把缺失去掉而不是填充（dropna），训练集还剩多少条？和填充比哪个好？

练习 2：把 'city' 从独热改成 LabelEncoder / 序数编码，维度会从多少变成多少？

练习 3：故意制造数据泄漏 —— 在全量 df 上 fit StandardScaler 再 split，
        对比 val 指标有没有虚高。（体会"泄漏"为什么危险）

练习 4：给 TabularDataset 加一个 __getitem__ 里的随机噪声增强（训练时用），
        体会 Dataset 的扩展性。

练习 5（思考题，面试常问）：
        如果某个类别特征有 1 万个取值（比如用户ID），独热会怎样？该怎么处理？
""")

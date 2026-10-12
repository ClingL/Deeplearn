# 深度学习速成 → 找实习

目标：**4 周**从零到能写出一个能讲清楚的深度学习项目 + 过掉实习面试的基础八股。

你的机器：RTX 5070 (12GB)。够用了，跑 CIFAR-10、微调 ResNet 都没问题。

---

## 0. 环境（已经帮你装好）

虚拟环境在 `.venv/`，Python 3.12 + PyTorch (CUDA 12.8 版，支持 5070 的 Blackwell 架构)。

**运行任何一课的固定命令（在 `deeplearn` 目录下）：**

```powershell
.\run.cmd lessons\01_tensors_autograd.py
```

`run.cmd` 是个包装脚本，帮你做了三件事：
1. 自动用 `.venv` 里的 Python —— 不用写长路径，也不用 `activate`
2. 把控制台和 Python 的编码都固定成 UTF-8 —— **中文不会乱码**
3. 自动切到项目根目录 —— `--data-dir data\flowers` 这类相对路径不会出错

它不受 PowerShell 执行策略限制，直接能跑。

**想把输出存成文件发我**（日志是 UTF-8，我能正确读到中文）：

```powershell
.\run.cmd project_cv\src\train.py --data-dir data\flowers_split --run-name 01_baseline --arch smallcnn --no-pretrained --epochs 40 > outputs\01.log 2>&1
```

原始写法也可以用（效果一样，只是要自己注意编码）：

```powershell
.\.venv\Scripts\python.exe lessons\01_tensors_autograd.py
```

先跑这一句验证 GPU 能用：

```powershell
.\run.cmd -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望输出类似：`2.11.0+cu128 True NVIDIA GeForce RTX 5070`

> 如果 `cuda.is_available()` 是 False，告诉我，我帮你查（通常是装成了 CPU 版）。

想用 Jupyter 交互式学（推荐，边跑边看）：

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

---

## 1. 学习路线（4 周）

| 周 | 主题 | 文件 | 产出 |
|---|---|---|---|
| 第 1 周 | 张量 / 自动求导 / 训练循环 | `lessons/01` `02` | 能手写梯度下降，理解 backward |
| 第 2 周 | 数据处理 / 标准训练模板 | `lessons/03` `04` | 能把任意 CSV 变成能训的数据 |
| 第 3 周 | CNN / 迁移学习 | `lessons/05` `06` | 能微调预训练模型，精度 90%+ |
| 第 4 周 | 做项目 + 背面试题 | `PROJECT.md` `INTERVIEW.md` | 简历上一条项目 + 面试能答 |

**每天 2 小时**：1 小时跑代码 + 改代码，1 小时看概念/记笔记。
**关键原则：每课的代码你都要动手改一个地方再跑一遍。** 只跑不改 = 没学。

---

## 2. 每课怎么用

每个 `lessons/*.py` 文件都是：
- 从上到下分节（`sec("...")` 打印标题），**输出就是讲义**
- 注释里写了「观察什么」，跑完对着输出看
- 结尾有 `练习`，答案在下一课开头或者你自己想

建议流程：
1. 先跑一遍，看输出
2. 回头看代码，每行都能说出干嘛的
3. 改一个超参数（学习率、batch size、层数）再跑，看结果怎么变
4. 卡住了把报错/输出贴给我

---

## 3. 目录

```
deeplearn/
├─ run.cmd                ← 【入口】运行任何脚本都用它，中文不乱码
├─ README.md              ← 你在这
├─ RESUME.md              ← 简历怎么写（通用）
├─ RESUME_DRAFT.md        ← 你的简历草稿（含 5 天双项目计划）
├─ PROJECT.md             ← 简历项目怎么做
├─ INTERVIEW.md           ← 面试题库
├─ lessons/               ← 教学代码（6 课，输出即讲义）
│  ├─ 01_tensors_autograd.py    张量 + 自动求导
│  ├─ 02_first_training_loop.py 从零手写训练循环
│  ├─ 03_data_pipeline.py       数据处理全流程
│  ├─ 04_train_template.py      标准训练模板
│  ├─ 05_cnn_images.py          CNN 图像分类
│  └─ 06_transfer_learning.py   迁移学习
├─ project_cv/            ← 【项目 A】图像分类工程骨架（对口 CV 算法岗）
│  ├─ README.md                  跑实验的完整命令 + 消融表格模板
│  ├─ src/    data.py models.py engine.py train.py evaluate.py
│  └─ scripts/ audit_data.py split_data.py dedup_data.py make_report.py
└─ project_rag/           ← 【项目 B】多模态文档问答（对口大模型应用岗）
   └─ README.md + main.py + src/
```

数据会自动下载/生成到 `data/`，模型和图表存到 `outputs/`。

---

## 4. 一句话心法（先记住，后面会反复验证）

深度学习 = **用一个可微的函数拟合数据**。

训练循环永远是这 5 步，背下来：

```
1. 前向：pred = model(x)
2. 算损失：loss = criterion(pred, y)
3. 清梯度：optimizer.zero_grad()
4. 反向：loss.backward()
5. 更新：optimizer.step()
```

所有模型、所有任务、所有论文代码，都是这个骨架换了零件。

---

## 5. 找实习的现实提醒

- 实习面试**不会**问你 Transformer 的推导，但**一定**会问：
  - 训练不收敛怎么排查？
  - 过拟合怎么办？
  - BN 是干嘛的？训练和推理有什么不同？
  - 数据泄漏是什么？
  - 为什么验证集 loss 上升要早停？
- 你需要的不是"学过多少模型"，而是**一个你能从头讲到尾的项目**（见 `PROJECT.md`）。
- 简历上的项目要能说出：数据多少条、怎么处理、模型多大、训了多久、指标多少、遇到什么问题、怎么解决的。

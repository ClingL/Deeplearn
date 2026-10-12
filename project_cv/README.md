# 图像分类项目骨架（迁移学习 + 消融实验）

> 这是给你做**项目 A** 用的骨架。`lessons/05`、`06` 是教学代码，
> 这个 `project_cv/` 是**能写进简历的项目模板**：有配置、有消融、有错误分析、有报告。
>
> 你要做的事：**换一个数据集**，跑出你自己的一组数字。

---

## 一、先准备数据

### 1. 下载数据集

推荐（Kaggle 直接搜名字下载，选「按类别分好的文件夹」那种）：

| 数据集 | 类别 | 图片数 | 说明 |
|---|---|---|---|
| Oxford Flower 102 | 102 | 8,189 | 推荐，类别多、难度适中 |
| TrashNet | 6 | 2,527 | 数据少，能凸显迁移学习价值 |
| Stanford Dogs | 120 | 20,580 | 难，类间差异小 |
| 自选 | - | - | 任何按类别分文件夹的图像数据集都行 |

解压后应该长这样（每个类别一个文件夹）：

```
data/flowers/
├─ 0/   img_0001.jpg ...
├─ 1/   ...
└─ ...
```

### 2. 检查目录结构

```powershell
.\.venv\Scripts\python.exe project_cv\scripts\audit_data.py --data-dir data\flowers
```

这会做**数据体检**，输出：
- 类别数、每类图片数、类别不均衡比例
- 损坏/无法读取的图片
- 重复图片（按 MD5 去重）
- 图片尺寸分布

**把这份输出贴进你的 README**——这就是简历上「数据体检发现 87 张重复图、3 类不均衡」那一行的来源。

### 3. 划分数据集（如果只有一个目录）

```powershell
.\.venv\Scripts\python.exe project_cv\scripts\split_data.py --data-dir data\flowers --val-ratio 0.2
```

会生成 `data/flowers_split/{train,val}/`，分层切分。
（如果数据集本身已经给了 train/val 目录，跳过这步，直接把它当 `--data-dir`。）

---

## 二、跑实验（简历上那张表的来源）

按顺序跑，每条命令产出一个 `outputs/<run_name>/`，最后聚合成表格。

### 实验 1：Baseline —— 从零训练的小 CNN

```powershell
.\.venv\Scripts\python.exe project_cv\src\train.py `
  --data-dir data\flowers_split --run-name 01_baseline `
  --arch smallcnn --no-pretrained --augment basic `
  --epochs 40 --batch-size 64 --lr 1e-3
```

### 实验 2：换成预训练 ResNet18（只训 head，冻结 backbone）

```powershell
.\.venv\Scripts\python.exe project_cv\src\train.py `
  --data-dir data\flowers_split --run-name 02_resnet_frozen `
  --arch resnet18 --pretrained --augment basic `
  --epochs 40 --batch-size 64 --lr 1e-3 --freeze-epochs 40
```

### 实验 3：解冻微调 + 判别式学习率（主实验）

```powershell
.\.venv\Scripts\python.exe project_cv\src\train.py `
  --data-dir data\flowers_split --run-name 03_resnet_finetune `
  --arch resnet18 --pretrained --augment basic `
  --epochs 40 --batch-size 64 --lr 1e-3 --backbone-lr 1e-4 --freeze-epochs 3
```

### 实验 4：加强数据增强（消融：增强贡献多少）

```powershell
.\.venv\Scripts\python.exe project_cv\src\train.py `
  --data-dir data\flowers_split --run-name 04_strong_aug `
  --arch resnet18 --pretrained --augment strong `
  --epochs 40 --batch-size 64 --lr 1e-3 --backbone-lr 1e-4 --freeze-epochs 3
```

### 实验 5：更强骨干（可选加分）

```powershell
.\.venv\Scripts\python.exe project_cv\src\train.py `
  --data-dir data\flowers_split --run-name 05_effnetb0 `
  --arch efficientnet_b0 --pretrained --augment strong `
  --epochs 40 --batch-size 32 --lr 1e-3 --backbone-lr 1e-4 --freeze-epochs 3
```

### 实验 6：无增强（消融：增强到底有没有用）

```powershell
.\.venv\Scripts\python.exe project_cv\src\train.py `
  --data-dir data\flowers_split --run-name 06_no_aug `
  --arch resnet18 --pretrained --augment none `
  --epochs 40 --batch-size 64 --lr 1e-3 --backbone-lr 1e-4 --freeze-epochs 3
```

---

## 三、生成消融对比表

```powershell
.\.venv\Scripts\python.exe project_cv\scripts\make_report.py --outputs-dir outputs
```

会打印一张 markdown 表格，**直接粘进 README 和简历**：

```
| 实验 | 骨干 | 预训练 | 增强 | 微调 | 最好 val acc | test acc | 参数量 |
```

---

## 四、错误分析（面试加分项）

```powershell
.\.venv\Scripts\python.exe project_cv\src\evaluate.py `
  --ckpt outputs\04_strong_aug\best.pt --data-dir data\flowers_split
```

产出 `outputs/04_strong_aug/eval/`：
- `confusion_matrix.png` —— 混淆矩阵热力图
- `per_class_acc.txt` —— 每个类别的准确率（找出最差的类）
- `top_confusions.txt` —— 最容易混淆的类别对
- `misclassified.png` —— 预测错的样本拼图

**把「最易混的 3 组类别 + 你分析的原因」写进 README。**
这一段是本科生简历里没有的东西，是你作为硕士的差异化。

---

## 五、每轮实验的产物

```
outputs/<run_name>/
├─ best.pt          验证集最优权重
├─ last.pt          最后一轮
├─ config.json      这次跑的全部超参数（可复现）
├─ history.json     每轮 loss / acc / lr
├─ metrics.json     最终 val/test 指标（供 make_report.py 聚合）
└─ curves.png       train/val 曲线图
```

---

## 六、README 里必须有的东西（照抄填空）

```markdown
## 结果

| 实验 | 骨干 | 预训练 | 增强 | 微调策略 | val acc | test acc | F1 |
|---|---|---|---|---|---|---|---|
| 01 baseline | SmallCNN | 否 | basic | 从零 | 待填 | 待填 | 待填 |
| 02 frozen   | ResNet18 | 是 | basic | 冻结 | 待填 | 待填 | 待填 |
| 03 finetune | ResNet18 | 是 | basic | 判别式LR | 待填 | 待填 | 待填 |
| 04 strong   | ResNet18 | 是 | strong| 判别式LR | 待填 | 待填 | 待填 |
| 05 effnet   | EffNetB0 | 是 | strong| 判别式LR | 待填 | 待填 | 待填 |
| 06 no-aug   | ResNet18 | 是 | none  | 判别式LR | 待填 | 待填 | 待填 |

## 数据
- 来源：【Kaggle 数据集名 + 链接】
- 规模：【XX】类、【XXXX】张
- 数据体检发现：【XX】张重复图（已去重）、【X】类不均衡【1:X】、无损坏图
- 处理：去重 → 分层切分 8:2 → 少数类加权采样

## 关键发现
- 从零训练 vs 迁移学习：待填（差 XX 个百分点）
- 数据增强贡献：待填（04 vs 06）
- 判别式学习率 vs 全冻结：待填（03 vs 02）
- 最易混的 3 组类别：【A/B】【C/D】【E/F】，原因是【待填】
- 试过但没用的方案：【待填，例如 mixup 在这个数据集上没提升，因为……】

## 运行
见上方命令。环境：Python 3.12 + PyTorch 2.11 + CUDA 12.8，单卡 RTX 5070。
```

---

## 七、常见问题

**Q: 显存不够 / batch size 报 OOM？**
调小 `--batch-size`（32 或 16），或调小 `--img-size`。

**Q: 报错 `num_workers` 相关？**
Windows 上默认已设成 0。想加速就设 `--num-workers 4`，
但必须保证脚本有 `if __name__ == "__main__":` 保护（这个骨架已经写好了）。

**Q: 第一次跑要下预训练权重？**
是，ResNet18 约 45MB，EfficientNet-B0 约 20MB，自动下到 `~/.cache/torch`。
（如果想要下到项目里，设环境变量 `TORCH_HOME`。）

**Q: 跑太慢？**
40 epoch 在 5070 上，8000 张图大概 5-10 分钟。
嫌慢可以先 `--epochs 15` 看趋势，定稿再跑满。

**Q: 准确率很低（比如 20%）？**
按顺序查：
1. `--pretrained` 加了吗？
2. `--backbone-lr` 是不是太大了（应比 head 小 10 倍）？
3. 数据划分对不对（类别文件夹有没有搞错）？
4. 学习率是不是太大（看 curves.png 有没有震荡）
5. 图片张数是不是太少（每类 <50 张就很难）

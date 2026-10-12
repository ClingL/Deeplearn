# 错误分析: 05_effnetb0

- 划分: test，样本数 980
- 准确率: **0.9755**，F1(macro): 0.9711
- 错分样本数: 24

## 最差的 3 个类别

| 类别 | 准确率 | 样本数 |
|---|---|---|
| carnation | 0.7143 | 7 |
| sweet pea | 0.7143 | 7 |
| japanese anemone | 0.7143 | 7 |

## 最容易混淆的 5 组

| 次数 | 真实 | 预测 |
|---|---|---|
| 2 | sweet pea | bougainvillea |
| 1 | sword lily | monkshood |
| 1 | sword lily | cyclamen |
| 1 | sword lily | canterbury bells |
| 1 | sweet william | tree poppy |

## 我的分析（必须自己写，面试会问）

- 为什么这几类容易混？（长得像 / 训练样本少 / 标注有歧义 / 分辨率不够）
- 你打算怎么改？（补数据 / 针对性增强 / 换更大输入 / 类别合并）

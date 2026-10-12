# 多模态文档问答系统 (RAG)

一个从 PDF 出发的检索增强生成 (RAG) 系统:

```
PDF -> 解析 -> 切块 -> 向量化 -> 索引 -> 混合检索 -> 组装 Prompt -> 大模型带引用回答
```

- **阶段一 (主线)**: 纯文本 RAG。PDF 抽文本、切块、bge 向量、FAISS 检索、BM25 关键词检索、
  RRF 混合召回, 最后由大模型给出**带页码引用**的回答。
- **阶段二 (可选)**: 用 CLIP 把 PDF 内嵌的图片也编码进索引, 支持"用文字找图"的跨模态检索。
  通过 `--with-images` 开关启用。

技术栈: PyMuPDF / sentence-transformers (bge-small-zh-v1.5) / CLIP / FAISS / 自研 BM25 /
OpenAI 兼容接口 (DeepSeek、智谱、阿里百炼、OpenAI 均可)。

> 说明: 本项目刻意不依赖 `jieba` 和 `rank-bm25`。
> BM25 与中文分词都在 `src/index.py` 里自己实现 (约 40 行), 中文用
> "单字 + 字符二元组 (bigram)" 切分。这样做的好处是没有第三方编译依赖, 随处可跑;
> 代价见文末「已知限制」。

---

## 一、系统架构

```
                        离线阶段: 建立索引 (python main.py ingest)
                        ============================================

   PDF 文件
      |
      | (1) 解析  PyMuPDF
      v
  +--------------+                    +----------------+
  |  每页文本     |                    |  每页内嵌图片   |
  +------+-------+                    +--------+-------+
         |                                     |
         | (2) 切块: 段落 -> 句子 -> 按尺寸装箱   | (2') 图片记录 + 过滤小图标
         |     chunk_size / chunk_overlap       |
         v                                     v
  +--------------+                    +----------------+
  |  文本 chunk   |                    |  图片 chunk     |
  |  chunk_id    |                    |  chunk_id      |
  |  doc / page  |   <-- 溯源元数据    |  doc / page    |
  +------+-------+                    +--------+-------+
         |                                     |
         | (3) 向量化                           | (3') 向量化
         |  bge-small-zh-v1.5 (512 维)          |  CLIP (512 维)
         v                                     v
  +--------------+                    +----------------+
  |  text.faiss  |                    |  image.faiss   |
  |  IndexFlatIP |                    |  IndexFlatIP   |
  |  向量已归一化 |                    |  向量已归一化   |
  +--------------+                    +----------------+
         |                                     |
         +------------------+------------------+
                            |  统一存放在同一个 index_dir/
                            v
                +---------------------------+
                |  bm25_tokens.json         |
                |  (文本侧关键词索引)         |
                +---------------------------+
                |  chunks.jsonl             |
                |  (每个 chunk 的 doc/page)  |
                +---------------------------+


                        在线阶段: 问答 (python main.py ask / eval)
                        ============================================

   用户问题
      |
      +------------------> bge 编码查询 -------------> text.faiss  ---> 向量 top-N
      |                   (查询侧加 instruction 前缀)
      |
      +------------------> tokenize + 自研 BM25 ------> 关键词 top-N
      |                   (单字 + 字符 bigram)
      |
      +------------------> CLIP 文本塔编码 -----------> image.faiss ---> 图像 top-N   [可选]
                          (跨模态: 用文字搜图)
                            |
                            v
                 RRF 融合 (Reciprocal Rank Fusion)
                 只用"排名", 不做分数归一化
                            |
                            v
                  top-k chunks (带 doc / page / chunk_id)
                            |
                            v
        组装 Prompt: 片段编号 + 出处页码 + 强制要求标注 [编号]
                            |
                            v
        LLM (OpenAI 兼容接口: DeepSeek / 智谱 / 阿里百炼 / OpenAI / 本地 Ollama)
                            |
                            v
        带引用编号的回答  +  终端打印命中的原文片段 (便于人工核对)
```

数据流一句话总结:
**PDF -> 解析 -> 切块 -> 向量化 -> 索引 -> 检索 -> 生成**, 其中"检索"是向量与关键词双路并行,
"生成"必须带出处。

---

## 二、和「普通文本 RAG」的区别, 为什么要做多模态

普通文本 RAG 只处理文字, 而真实的技术文档 (论文、产品手册、财报) 里, 大量关键信息
是**以图表形式存在**的:

| 信息类型 | 纯文本 RAG 的结果 | 本项目的处理 |
| --- | --- | --- |
| 正文段落 | 能检索到 | 走文本链路 (bge + BM25) |
| 折线图 / 柱状图 | 完全检索不到 (文字层里往往只有 "图 3" 这种标题, 坐标轴和趋势都在图上) | 抽出图片, 用 CLIP 编码进图像索引, 可用"哪一年增长最快"这类问题召回 |
| 架构图 / 流程图 | 检索不到 | 同上 |
| 表格 | 只能抽到被压扁的文本流, 行列关系丢失 | 能抽到部分文本, 但结构会乱, 属于已知限制 |
| 公式截图 | 检索不到 | 作为图片进入图像索引 |

具体带来三个能力差异:

1. **跨模态召回**: 用户问"图 4 里 attention 的结构是什么样的", 纯文本 RAG 只能靠
   "图 4" 这个字符串碰运气命中图注; 本系统会额外用 CLIP 去图像索引里找语义相近的图。
2. **双路互补的召回质量**: 向量召回擅长"语义相近但用词不同" (问"怎么加快训练",
   能召回"优化器的收敛速度"), BM25 擅长"必须精确命中"的专有名词、编号、人名、超参数。
   只用向量会漏掉精确匹配, 只用关键词会漏掉同义表达, 所以两者都要。
3. **可以解释、可以核对**: 每个 chunk 都带 `{doc, page, chunk_id}`, Prompt 里明确写出
   出处页码, 回答必须标注 `[编号]`, 终端同时打印命中的原文。没有引用的 RAG 在企业场景
   是不可用的, 因为你无法判断模型是在总结还是在编。

诚实说明: 本项目的多模态是**简易实现**——文本和图像分别建索引, 查询时两边都查再合并
(详见文末「已知限制」)。这是一个工程上常见且性价比很高的做法, 但它不等于训练一个
统一的跨模态检索模型。

---

## 三、目录结构

```
project_rag/
├─ README.md                       本文件
├─ .env.example                    LLM 配置模板 (复制成 .env 后填写)
├─ requirements-rag.txt            本项目额外依赖
├─ main.py                         CLI 入口: ingest / ask / eval
├─ data/
│   └─ eval_set.example.jsonl      评测集格式示例 (复制成 eval_set.jsonl 后改成自己的数据)
├─ src/
│   ├─ __init__.py
│   ├─ config.py                   读 .env、固定模型缓存到 ./hf_cache、统一终端输出
│   ├─ loaders.py                  PDF -> (每页文本, 每页图片), 用 PyMuPDF
│   ├─ chunking.py                 文本切块, 保留 (doc, page, chunk_id) 元数据
│   ├─ embedding.py                文本向量 bge-small-zh-v1.5 / 图像向量 CLIP
│   ├─ index.py                    FAISS 索引 + 自研 BM25 + RRF 融合
│   ├─ rag.py                      检索 -> Prompt -> 调 LLM -> 带引用回答
│   └─ evaluate.py                 recall@k / MRR@k / 答案准确率 + markdown 报告
├─ hf_cache/                       模型缓存 (首次运行自动创建, 不会污染用户主目录)
└─ index_dir/                      索引输出目录 (运行 ingest 后生成)
```

---

## 四、安装

前置: 已经有一个可用的 Python 3.12 虚拟环境在仓库根目录的 `.venv`
(torch / numpy 等已装好)。

```powershell
# 在仓库根目录 C:\Users\Cling\Documents\deeplearn 下执行
# 安装本项目额外依赖 (sentence-transformers / faiss-cpu / pymupdf / openai / ...)
.venv\Scripts\python.exe -m pip install -r project_rag\requirements-rag.txt
```

**首次运行会下载模型, 请保持网络畅通:**

| 模型 | 用途 | 大小 |
| --- | --- | --- |
| `BAAI/bge-small-zh-v1.5` | 文本向量 (必需) | 约 100 MB |
| `openai/clip-vit-base-patch32` | 图像向量 (仅 `--with-images` 需要) | 约 600 MB |

缓存目录已经由 `src/config.py` 固定在项目内的 `project_rag/hf_cache/`,
不会写进 `C:\Users\<你>\.cache`。如果下载很慢或连不上 HuggingFace, 可以先设置国内镜像:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

---

## 五、运行

所有命令都在 `project_rag` 目录下执行。把下面的 `python` 理解为该虚拟环境的解释器
(最稳妥的写法是直接用 `..\.venv\Scripts\python.exe`)。

### 第 0 步: 配置大模型

```powershell
cd project_rag
copy .env.example .env
# 然后用编辑器打开 .env, 至少填好这三项:
#   LLM_API_KEY   = 你的 key
#   LLM_BASE_URL  = https://api.deepseek.com/v1     (DeepSeek 示例)
#   LLM_MODEL     = deepseek-chat
```

`.env.example` 里已经写好 DeepSeek / 智谱 / 阿里百炼 / Moonshot / OpenAI / 本地 Ollama
的 base_url 与模型名写法。本项目统一走 OpenAI 兼容接口, 换服务商只改这三个变量。

### 第 1 步: 建立索引 (Stage 1, 纯文本)

```powershell
python main.py ingest --pdf data/your_doc.pdf --out index_dir
```

### 第 2 步: 提问

```powershell
python main.py ask --index index_dir "这篇文档的核心结论是什么?"
python main.py ask --index index_dir "论文里用的学习率是多少?" -k 5
python main.py ask --index index_dir "图 3 说明了什么?" --show-context
```

`--show-context` 会把送给大模型的完整上下文和 Prompt 打印出来, 用来排查
"检索到的内容对不对"。

### 第 3 步: 评测

```powershell
copy data\eval_set.example.jsonl data\eval_set.jsonl
# 编辑 eval_set.jsonl, 换成你自己 PDF 的问题、答案、正确页码

python main.py eval --index index_dir --eval-set data/eval_set.jsonl -k 5
python main.py eval --index index_dir --eval-set data/eval_set.jsonl --judge llm
python main.py eval --index index_dir --eval-set data/eval_set.jsonl --out-md results.md
```

### Stage 2: 加上图像检索

```powershell
# 建索引时开启 (会额外抽取 PDF 内嵌图片并用 CLIP 编码)
python main.py ingest --pdf data/your_doc.pdf --out index_dir_img --with-images

# 提问时也开启
python main.py ask --index index_dir_img --with-images "图 4 里画了什么?"
```

---

## 六、参数说明

### ingest

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--pdf` | 必填 | 要解析的 PDF 路径 |
| `--out` | 必填 | 索引输出目录 |
| `--with-images` | 关闭 | 抽取图片并用 CLIP 建图像索引 (Stage 2) |
| `--chunk-size` | 512 | 切块最大字符数 (消融实验的主要变量) |
| `--chunk-overlap` | 64 | 相邻块重叠字符数 |
| `--min-image-size` | 64 | 图片最小边长, 小于该值当图标过滤 |
| `--text-model` | `BAAI/bge-small-zh-v1.5` | 文本向量模型 |
| `--clip-model` | `openai/clip-vit-base-patch32` | 图像向量模型 |
| `--device` | 自动 | 强制 `cpu` 或 `cuda` |

### ask / eval (共用)

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--index` | 必填 | ingest 生成的索引目录 |
| `-k` | 5 | 检索返回的 chunk 数 |
| `--no-bm25` | 关闭 | 关掉 BM25, 只用向量检索 (消融实验) |
| `--with-images` / `--no-images` | 关闭 | 启用/禁用 CLIP 图像检索 |
| `--rrf-k` | 60 | RRF 平滑常数 |
| `--candidate-n` | 20 | 每路召回各取多少候选再融合 |

### eval 专有

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--eval-set` | 必填 | JSONL 评测集 |
| `--judge` | `keyword` | `keyword` 关键词命中率 / `llm` 大模型裁判 / `both` 两者都算 |
| `--no-llm` | 关闭 | 完全不调用大模型, 用检索到的原文算关键词命中率 (相当于检索上界) |
| `--details` | 关闭 | 报告里附上逐题明细表 |
| `--out-md` | 无 | 把 markdown 结果表写入文件, 方便贴进本 README |

### 评测集格式 (`data/eval_set.jsonl`)

每行一个 JSON 对象, **不能换行**:

```json
{"question": "这篇论文提出的方法叫什么?", "answer": "参考答案", "source_page": 3, "source_doc": "paper.pdf"}
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `question` | 是 | 问题 |
| `answer` | 是 | 参考答案, 用于算答案准确率 |
| `source_page` | 建议 | 正确内容所在页码 (从 1 开始), 用于算 recall@k |
| `source_doc` | 多文档时建议 | 正确内容所在文件名 |

`source_page` 和 `source_doc` 至少要有一个, 否则该题无法判断检索是否命中。
以下划线开头的字段 (如 `_comment`) 会被自动忽略。

---

## 七、指标定义 (报告里的数字是怎么算出来的)

假设评测集有 `N` 条样本, 每题检索 top-k 个 chunk。

- **recall@k** = 在所有"标注了出处"的样本里, 检索结果前 k 个 chunk 中**至少有一个**
  的 `(doc, page)` 与标准出处一致的比例。它衡量检索上限: 检索不到, 大模型一定答不对。
- **MRR@k** = 第一个命中 chunk 的排名倒数的平均值 (`1/rank`, 没命中记 0)。
  recall@k 只告诉你"有没有找到", MRR@k 还能区分"排第 1"和"排第 5"。
- **关键词命中率 (答案准确率, `--judge keyword`)** = 从**参考答案**里抽关键词
  (长度 >= 2 的 bigram 或英文词, 去停用词), 看有多少比例出现在**模型回答**里, 再对
  所有样本取平均。便宜、可复现、不花钱, 适合大规模跑消融实验; 缺点是只看词是否出现,
  不判断语义。
- **LLM 裁判平均分 (`--judge llm`)** = 让大模型对比"模型回答"和"参考答案", 打
  `1 / 0.5 / 0` 分 (完全正确 / 部分正确 / 错误或编造), 再取平均。更接近人的判断,
  但花 token, 且裁判本身有随机性。

---

## 八、试验记录 (消融实验)

> 这张表是简历上「消融实验」那一行的来源。跑完请把**真实数字**填进来,
> 一行都不要编。每行对应下面给出的命令。

### 8.1 待填表格

| 编号 | chunk_size / overlap | 检索方式 | 图像检索 | recall@5 | MRR@5 | 答案准确率 (关键词) | 答案准确率 (LLM 裁判) | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 基线 | 512 / 64 | 仅向量 | 关 | 待填 | 待填 | 待填 | 待填 | 最小可用版本 |
| A2 +BM25 | 512 / 64 | 向量 + BM25 | 关 | 待填 | 待填 | 待填 | 待填 | 验证混合检索的收益 |
| A3 | 256 / 32 | 向量 + BM25 | 关 | 待填 | 待填 | 待填 | 待填 | 小块, 召回更细但上下文更碎 |
| A4 | 1024 / 128 | 向量 + BM25 | 关 | 待填 | 待填 | 待填 | 待填 | 大块, 上下文更全但噪声更多 |
| A5 +图像 | 512 / 64 | 向量 + BM25 + CLIP | 开 | 待填 | 待填 | 待填 | 待填 | Stage 2, 跨模态召回 |

### 8.2 每行对应的命令

```powershell
# A1 基线: 只建向量索引, 检索时关掉 BM25
python main.py ingest --pdf data/your_doc.pdf --out idx_cs512
python main.py eval --index idx_cs512 --eval-set data/eval_set.jsonl -k 5 --no-bm25

# A2 加上 BM25 混合检索 (同一个索引, 不用重建)
python main.py eval --index idx_cs512 --eval-set data/eval_set.jsonl -k 5

# A3 小块
python main.py ingest --pdf data/your_doc.pdf --out idx_cs256 --chunk-size 256 --chunk-overlap 32
python main.py eval --index idx_cs256 --eval-set data/eval_set.jsonl -k 5

# A4 大块
python main.py ingest --pdf data/your_doc.pdf --out idx_cs1024 --chunk-size 1024 --chunk-overlap 128
python main.py eval --index idx_cs1024 --eval-set data/eval_set.jsonl -k 5

# A5 加图像 (Stage 2)
python main.py ingest --pdf data/your_doc.pdf --out idx_img --with-images
python main.py eval --index idx_img --eval-set data/eval_set.jsonl -k 5 --with-images
```

做实验的两条纪律:

1. **一次只改一个变量**, 否则说不清收益来自哪里。
2. 每次 `eval` 都把 markdown 表存下来, 用 `--out-md results_A2.md`,
   最后直接对比, 也方便贴进本 README。

### 8.3 定性观察 (可选, 但面试很加分)

跑 `ask` 的时候顺便记录几条典型 case, 例如:

| 问题 | 命中页 | 回答对不对 | 分析 |
| --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 |

比如"某题 BM25 命中了但向量没命中, 因为问题里出现了精确的型号编号"这类记录,
在面试里比一个孤立的准确率数字更有说服力。

---

## 九、已知限制

诚实列出, 面试时主动说反而加分:

1. **表格解析弱**。PyMuPDF 抽表格只能得到被压扁的文本流, 行列对应关系会丢失,
   跨行跨列的复杂表格基本抽不对。生产环境应换成专门的表格解析 (如
   `pdfplumber` 的 table 抽取, 或版面分析模型 + 结构化后处理)。
2. **扫描版 PDF 必须先用 OCR**。本项目依赖 PDF 的文字层, 对于整页是图片的扫描件,
   抽出来是空文本。程序会检测这种情况并给出警告, 但不会自动 OCR。需要先跑
   OCRmyPDF / PaddleOCR 生成带文字层的 PDF。
3. **跨模态检索是简易实现**。文本索引和图像索引是分开建、分开查, 最后用 RRF 合并结果,
   **没有**把两种模态对齐到同一个向量空间, 也没有做跨模态的 rerank。
   另外 `clip-vit-base-patch32` 的文本塔主要在英文语料上训练, **中文 query 搜图的效果
   明显弱于英文 query**, 这是当前方案最明显的短板。要改进需要换成中文 CLIP
   (如 Chinese-CLIP) 或自建图文对比学习。
4. **BM25 用字符 bigram 近似分词, 未使用专业分词器**。因为 `jieba` 在当前环境无法安装
   (纯源码包, 没有 Windows 预编译 wheel), 本项目自己实现了"单字 + 字符二元组"的切分。
   它不需要词典、对新词和专有名词很鲁棒, 但相比成熟分词器:
   - 单个 chunk 产生的 token 数量大约翻倍, 建索引和检索时内存/耗时更高;
   - 无法利用词性、词典边界等语言学信息, 语义上的"词"级别匹配会略弱。
   如果要更好效果, 只需替换 `src/index.py` 里的 `tokenize()` 函数 (接口不变),
   例如换成 jieba 或 pkuseg, 其余代码无需改动。
5. **引用粒度只到页**, 不能定位到具体句子/段落。页码可信但不够精确。
6. **没有 Rerank 阶段**。目前是 RRF 融合后直接取 top-k, 没有用 cross-encoder
   做精排, 也没有 MMR 去重, top-k 里可能有内容重复的 chunk。
7. **图像 chunk 会占用 top-k 名额**。图像检索和文本检索共享同一个 k,
   所以开启图像检索后, 文本上下文里能塞进去的段落会变少。
8. **`IndexFlatIP` 是暴力检索**, 数据量在几万 chunk 以内完全没问题,
   再大就应该换 IVF / HNSW (FAISS 都支持, 只需改 `src/index.py` 建索引的那几行)。
9. **没有多轮对话和 query 改写能力**。每轮都是独立检索, 代词指代
   ("它的第二个参数是什么") 会检索不准。
10. **评测集需要人工标注, 样本少时指标波动大**。10 条样本的 recall@5 差 0.1
    可能只是噪声, 结论要谨慎。另外 LLM 裁判本身有随机性, 换模型结论可能变。

---

## 十、面试可能被追问的问题

**1. 为什么要做混合检索? 向量检索和 BM25 各自擅长什么?**
向量检索基于语义, 擅长"同义不同词" (问"如何加快训练", 能召回"优化器的收敛速度");
但它对低频专有名词、编号、精确超参数不敏感, 而且对"必须原样出现"的词容易漏。
BM25 基于词频统计, 擅长精确匹配 (人名、型号、`0.001` 这种数字), 但完全不理解同义。
两者互补, 所以并行召回后融合。我的消融实验里 A1 (仅向量) 对比 A2 (加 BM25)
就是为了量化这一项的收益。

**2. RRF 是怎么做的? 为什么要用排名而不是分数?**
RRF 的公式是 `score(chunk) = sum over 各路 1/(k + rank)`, k 通常取 60, rank 从 1 开始。
用排名的根本原因是**向量余弦相似度和 BM25 分数不在同一个量纲上**:
余弦在 [-1,1] 之间, BM25 是无上界的正数, 且随查询长度、语料规模漂移。
直接加权求和需要精细的分数归一化, 做不好就退化成单路检索。
只取名次则天然免疫量纲问题, 而且对异常分数鲁棒 (某一路给出 100 分也不会碾压其他路)。

**3. 切块大小 (chunk_size) 怎么定? overlap 有什么用? 你怎么验证的?**
没有理论最优值, 是**召回精度和上下文完整性的权衡**: 块太小, 单块信息不完整,
一句话被切两半, 检索到的片段答不了问题; 块太大, 一个块里混了多个主题,
向量的语义被平均掉, 检索精度下降, 而且浪费上下文窗口和 token。
overlap 是为了防止答案正好落在两块的交界处被切断, 一般取块大小的 10%~20%。
我的做法是固定其他变量, 只改 chunk_size 跑 256 / 512 / 1024 三档 (README 第 8 节 A3/A1/A4),
比较 recall@5 和答案准确率。切块策略也不是硬切, 而是优先在段落和句号边界切开。

**4. 为什么选 bge-small-zh-v1.5? 为什么不微调? 查询侧为什么加前缀?**
选它是因为: 中文效果好、模型小 (约 100MB, CPU 上也能跑)、512 维向量检索快、
`small` 版本对个人项目足够。不微调是因为标注成本高 (需要大量"问题-正确段落"对),
而当语料规模只有几百到几千 chunk 时, 通用模型 + BM25 混合检索的收益比微调更大,
微调容易在小数据上过拟合。bge 官方建议给**查询**加 instruction 前缀
("为这个句子生成表示以用于检索相关文章：") 而文档侧不加, 这是它训练时的非对称设定,
不加会损失几个点的召回。代码里 `encode_query` 和 `encode_documents` 就是这么区分的。

**5. 为什么用 `IndexFlatIP` 加向量归一化? 和 L2、IVF、HNSW 有什么区别?**
归一化后向量的内积等于余弦相似度 (`a·b = |a||b|cos = cos`, 当 `|a|=|b|=1`),
所以用内积索引就能做余弦检索, 避免了 L2 距离对向量长度敏感的问题。
`IndexFlat` 是精确暴力检索, 结果 100% 准确, 几万条以内毫秒级, 适合当前规模。
`IndexFlatL2` 是最小欧氏距离, 对未归一化向量更合适; 归一化后它与 IP 的排序等价,
但 IP 省去了开方、更直接。
数据量到百万级就该换 `IndexIVFFlat` (倒排 + 聚类, 需要训练, 用 nprobe 换速度)
或 `IndexHNSWFlat` (图索引, 查询快、召回高, 但内存开销大、构建慢)。
改动只在 `src/index.py` 建索引的几行, 检索接口不变。

**6. 多模态部分是怎么实现的? 为什么两个索引要分开?**
文本用 bge 编码, 图片用 CLIP 的图像塔编码, 分别建两个 FAISS 索引;
查询时用 bge 编码问题去查文本索引, 同时用 CLIP 的文本塔编码同一个问题去查图像索引,
两路结果再和 BM25 一起做 RRF 融合。分开建的原因是 **bge 和 CLIP 的向量空间不对齐**:
虽然维度可能都是 512, 但它们是两个独立训练的语义空间, 把文本向量和图像向量放进同一个
FAISS 索引里算内积是没有意义的。真正的统一方案需要训练一个把图文映射到同一空间的模型
(CLIP 本身就是用对比学习做到这一点的, 但它是英文的), 那是另一个量级的工程。

**7. 怎么减少幻觉? 引用溯源是怎么实现的?**
四层措施:
(1) 每个 chunk 从切块阶段就带上 `{chunk_id, doc, page}` 元数据, 一路传到 Prompt;
(2) 拼 Prompt 时给片段编号并写出"来源: xxx.pdf 第 3 页", 同时用 system prompt 强制要求
用 `[编号]` 标注每条关键信息;
(3) 明确要求"资料不足时回答根据现有资料无法确定, 不要编造", 并且不允许使用资料外的知识;
(4) 生成温度默认 0.2。另外终端会同时打印命中的原文片段, 用户可以自己核对回答有没有依据。
要更进一步还应加"引用有效性校验" (检查回答里的 [n] 是否真的存在于上下文中) 和
答案与原文的蕴含关系检查, 这是我下一步想做的。

**8. recall@k 和答案准确率有什么区别? 你为什么要用 LLM 当裁判?**
recall@k 衡量**检索**环节: 正确出处有没有进入前 k, 它是整个系统的上限,
检索不到就一定答不对。答案准确率衡量**端到端**效果, 检索对了但模型总结错了、
或者检索错了但模型碰巧答对了, 都会体现在这里。两者一起看才能定位瓶颈在哪一环。
用 LLM 当裁判是因为没有标准答案的自动评测器, 纯字符串匹配 (BLEU/ROUGE) 对语义等价
但措辞不同的回答惩罚过重。但 LLM 裁判的问题也很明确: 有随机性、有位置偏好、
会被自己的知识带偏, 而且花 token。所以本项目**两种都实现了** (`--judge keyword|llm|both`),
用便宜的关键词命中率跑大规模消融实验, 用 LLM 裁判在最终结论上做交叉验证。

**9. 如果 PDF 是扫描件 / 有复杂表格 / 有公式, 你的系统怎么办?**
扫描件: 抽不到文字层, 程序会检测总字符数过少并给出警告, 需要先做 OCR
(OCRmyPDF / PaddleOCR / Adobe 导出可搜索 PDF) 再入库。
复杂表格: 当前实现会抽成一团没有行列结构的文本, 属于已知弱点; 更好的做法是用
`pdfplumber` 抽取表格结构, 转成 markdown 或 HTML 保留行列, 再切块。
公式: 图片形式的公式可以走 CLIP 图像链路被召回, 但里面的符号信息检索不到,
需要专门的公式识别 (如 pix2tex) 或直接依赖 LaTeX 源码。

**10. 这套系统如果要上线, 你会改什么?**
按优先级: (1) 换成 IVF/HNSW 索引并加**增量索引**, 避免每次新增文档都全量重建;
(2) 加 cross-encoder 精排 (Rerank) 和 MMR 去重, 提升 top-k 的精度;
(3) 加"引用有效性校验"和服务端缓存, 降低幻觉与延迟;
(4) 做 query 改写以支持多轮对话和代词消解;
(5) 工程侧: 检索与生成解耦成服务、加并发与限流、大模型调用加重试与降级、
记录每次问答的检索结果便于回溯和收集 badcase;
(6) 成本侧: 小问题走小模型, 长文档做分层摘要。

---

## 十一、常见报错排查

| 现象 | 原因与解决办法 |
| --- | --- |
| `未检测到可用的 LLM_API_KEY` | 还没有 `.env`。执行 `copy .env.example .env` 并填写 `LLM_API_KEY` |
| 调用大模型报 401 | key 错了或过期, 检查 `.env` 里的 `LLM_API_KEY` 是否复制完整 |
| 调用大模型报 404 | `LLM_BASE_URL` 或 `LLM_MODEL` 不对。DeepSeek 必须是 `https://api.deepseek.com/v1` |
| 请求超时 | 调大 `.env` 里的 `LLM_TIMEOUT`, 或检查网络/代理 |
| 提示"几乎没有可提取文本" | 扫描版 PDF, 需要先做 OCR |
| 模型下载卡住 | 设置镜像: `$env:HF_ENDPOINT = "https://hf-mirror.com"` 后重跑 |
| `无法加载索引` | 还没跑过 ingest, 或 `--index` 指向的目录不对 |
| 终端中文乱码 | 控制台编码问题, 先执行 `chcp 65001` 切到 UTF-8 |
| `缺少 xxx 依赖` | 用虚拟环境安装: `.venv\Scripts\python.exe -m pip install -r project_rag\requirements-rag.txt` |

---

## 十二、后续可做的改进

- 加 Rerank (cross-encoder) 精排, 并对比加/不加的 recall@5
- 表格结构化抽取 (pdfplumber), 把表格转成 markdown 再切块
- 扫描件 OCR 自动化
- 换成中文 CLIP (Chinese-CLIP) 改善中文 query 搜图
- 增量索引: 新增 PDF 时只追加, 不重建
- 简单的 Web 界面 (Gradio / Streamlit), 支持上传 PDF 并高亮引用原文

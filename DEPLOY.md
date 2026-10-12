# 部署到 Ubuntu 服务器

> 目标：把这个项目在服务器上跑起来，重建环境 + 传数据 + 后台训练。
> 全程不需要 sudo，不会动服务器的系统 Python。

---

## 一、先理解：为什么 `.venv` 不能直接拷过去

看一下本机的 `.venv/pyvenv.cfg`：

```
home = C:\Users\Cling\Documents\deeplearn\.uv-python\cpython-3.12-windows-x86_64-none
implementation = CPython
version_info = 3.12.13
```

三个致命问题：

| 问题 | 说明 |
|---|---|
| **绝对路径写死** | `home` 指向本机 `C:\Users\Cling\...`，服务器上没这个路径 |
| **目录结构不同** | Windows 是 `.venv\Scripts\python.exe`，Linux 是 `.venv/bin/python` |
| **二进制不兼容** | torch 在 Windows 是 `.pyd`/`.dll`，Linux 是 `.so`，不能混用 |

**结论：虚拟环境必须在服务器上重建。** 好消息是重建只要一条命令（见第三节）。

---

## 二、本地：把代码交给 git

服务器上用 `git clone` 拿代码，所以先在本地建仓库。

`.gitignore` 已经排除了 `.venv/`、`.cache/`、`data/`、`outputs/*.pt` —— 
这些都不该进 git（虚拟环境 5GB、数据集几百 MB、模型权重几十 MB）。

在本地 `deeplearn` 目录执行：

```powershell
git init
git add .
git commit -m "深度学习项目：课程 + CV 项目 + RAG 项目"

# 检查一下哪些文件会被提交（确认没有 .venv / data / .cache）
git status --short

# 传到 GitHub（先在网页上建一个空仓库，不要初始化 README）
git remote add origin https://github.com/你的用户名/deeplearn.git
git branch -M main
git push -u origin main
```

**推之前确认这几件事：**

- `git status --short` 里**不应该出现** `.venv/`、`.cache/`、`data/` 里的东西
- 如果仓库是私有的也没关系，服务器上用 SSH key 或 token 拉取
- 提交体积应该只有几百 KB（纯代码和文档）

> 不想用 GitHub，也可以用学校/公司的 GitLab，或者直接在服务器上
> `git init` 然后把本地当 remote 推过去。命令一样。

---

## 三、服务器：clone + 一条命令建环境

```bash
# 1. 找一个有空间的目录（torch 加上依赖约 6-8 GB）
cd ~/work            # 或 /data/你的用户名/
git clone https://github.com/你的用户名/deeplearn.git
cd deeplearn

# 2. 建环境（自动识别 GPU 并选对 CUDA 版本）
bash setup.sh
```

`setup.sh` 会依次做：

1. 检查 Python 版本（需要 ≥ 3.10，低了会直接告诉你三种解决方案）
2. 用 `nvidia-smi` 识别 **GPU 型号 / 驱动版本 / 算力等级**
3. 按算力自动选 CUDA 频道 —— **RTX 5090 是 sm_120，会自动选 cu128**
4. 创建 `.venv` 并装 PyTorch + 依赖
5. 安装 `sitecustomize.py`（把模型缓存留在项目内，不占 home 配额）
6. 配置 matplotlib 中文字体
7. 跑一次 GPU 自检

**手动指定版本**（自动识别不对时）：

```bash
TORCH_CUDA=cu126 bash setup.sh     # 指定 CUDA 版本
bash setup.sh --cpu                # 强制 CPU 版
PYTHON=python3.12 bash setup.sh    # 指定 Python 解释器
```

**不该做的事**：不要 `sudo pip install`，不要往系统 Python 里装东西，
不要把 `.venv` 从本机拷上去。

---

## 四、数据怎么传（git 里没有数据）

`.gitignore` 排除了 `data/`，因为数据集动辄几百 MB。三种办法：

### 办法 A：服务器直接下载（推荐，不占你本地上传带宽）

以 Kaggle 的 Oxford Flower 102 为例：

```bash
# 一次性配置 kaggle 凭据：本地 ~/.kaggle/kaggle.json 传到服务器同名目录
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
# （把 kaggle.json 放进去）

pip install kaggle            # 或 .venv/bin/pip install kaggle
kaggle datasets download -d nunenuh/pytorch-challange-flower-dataset
unzip -q pytorch-challange-flower-dataset.zip -d data/flowers_raw
```

整理成 `data/flowers/<类别名>/*.jpg` 的结构，然后跑体检：

```bash
./run.sh project_cv/scripts/audit_data.py --data-dir data/flowers
```

### 办法 B：从本地 scp 上去

```powershell
# 在本地（Windows）执行
scp -r data\flowers 用户名@服务器IP:~/work/deeplearn/data/
```

数据小的话这样最快；大的话很慢，优先用办法 A。

### 办法 C：共享盘 / 网盘

学校集群常见，直接从共享盘 `ln -s` 软链到项目里：

```bash
ln -s /shared/datasets/flowers data/flowers
```

---

## 五、后台跑训练（很容易踩的坑）

**SSH 一断，前台跑的训练就死了。** 必须用 tmux 或 nohup。

### 用 tmux（推荐，可以随时回去看）

```bash
tmux new -s train                # 新建会话
cd ~/work/deeplearn
./run.sh project_cv/src/train.py \
    --data-dir data/flowers_split --run-name 03_resnet_finetune \
    --arch resnet18 --pretrained --augment basic \
    --epochs 40 --lr 1e-3 --backbone-lr 1e-4 --freeze-epochs 3

# 按 Ctrl+B 然后按 D  → 脱离会话，训练继续跑
# 下次登进来： tmux attach -t train
# 看所有会话： tmux ls
```

### 用 nohup（更简单，不习惯 tmux 就用这个）

```bash
mkdir -p outputs
nohup ./run.sh project_cv/src/train.py \
    --data-dir data/flowers_split --run-name 03_resnet_finetune \
    --arch resnet18 --pretrained --epochs 40 \
    > outputs/03.log 2>&1 &

# 看进度（Ctrl+C 只退出查看，不影响训练）
tail -f outputs/03.log

# 看还剩几个在跑
ps aux | grep train.py
```

### 一次跑完所有实验

```bash
mkdir -p outputs
for exp in "01_baseline --arch smallcnn --no-pretrained --augment basic" \
           "02_resnet_frozen --arch resnet18 --pretrained --augment basic --freeze-epochs 40" \
           "03_resnet_finetune --arch resnet18 --pretrained --augment basic --freeze-epochs 3 --backbone-lr 1e-4" \
           "04_strong_aug --arch resnet18 --pretrained --augment strong --freeze-epochs 3 --backbone-lr 1e-4"; do
    set -- $exp
    name=$1; shift
    echo "===== $name ====="
    ./run.sh project_cv/src/train.py --data-dir data/flowers_split \
        --run-name "$name" --epochs 40 "$@" 2>&1 | tee "outputs/$name.log"
done
```

跑完聚合表格：

```bash
./run.sh project_cv/scripts/make_report.py --outputs-dir outputs
```

---

## 六、常见报错

| 报错 | 原因 / 解决 |
|---|---|
| `Permission denied` 跑 `./setup.sh` | 用 `bash setup.sh`，或 `chmod +x setup.sh run.sh` |
| `python3 -m venv` 失败 / `ensurepip is not available` | `sudo apt-get install -y python3-venv python3-pip` |
| `Could not find a version that satisfies torch` | Python 太旧，`bash setup.sh` 会给三种解决方案 |
| `torch.cuda.is_available()` 是 False | 装成 CPU 版了。重装：`TORCH_CUDA=cu128 bash setup.sh`（先 `rm -rf .venv`） |
| `CUDA out of memory` | 调小 `--batch-size`（64 → 32 → 16），或调小 `--img-size` |
| `UnicodeEncodeError: 'ascii' codec` | 服务器 locale 是 C。`run.sh` 已经设了 `PYTHONUTF8=1`，用 `./run.sh` 跑就不会有 |
| 图表里中文是方框 | `sudo apt-get install -y fonts-noto-cjk`，然后重跑 `bash setup.sh` |
| `nvidia-smi: command not found` | 驱动没装或没在 PATH。没 GPU 就用 `bash setup.sh --cpu` |
| 训练很慢 | 确认 `torch.cuda.is_available()` 是 True；确认 `--batch-size` 没有小到浪费 GPU |

---

## 七、把结果拿回本地

```bash
# 打包关键产物（曲线图 + 指标 + 日志，不含模型权重）
cd ~/work/deeplearn
zip -r results.zip outputs -x "outputs/*.pt" "outputs/*.pth"
```

```powershell
# 在本地执行
scp 用户名@服务器IP:~/work/deeplearn/results.zip .
```

拿到 `outputs/*/metrics.json` 和 `curves.png` 之后，就可以：

1. 用 `make_report.py` 输出消融表 → 填进 [RESUME_DRAFT.md](RESUME_DRAFT.md) 的 `⚠️待跑实验` 位置
2. 用 `evaluate.py` 的 `summary.md` 写「错误分析」那一节

---

## 八、检查清单

- [ ] 本地 `git status --short` 里没有 `.venv/`、`.cache/`、`data/`
- [ ] 服务器上 `bash setup.sh` 跑完，最后自检显示 `cuda 可用 True`
- [ ] `./run.sh project_cv/scripts/make_dummy_data.py` 能跑通
- [ ] 烟雾测试能跑完（`--epochs 2`）
- [ ] 真实数据集就位，`audit_data.py` 输出的类别数符合预期
- [ ] 用 tmux 或 nohup 跑长任务，不是前台
- [ ] 跑完把 `outputs/` 结果传回来

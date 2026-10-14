#!/usr/bin/env bash
# =============================================================================
#  共享 GPU 服务器（无 sudo / 无调度器）上的一键环境安装
#
#  和项目根的 setup.sh 的区别：
#      setup.sh        通用版，假设 home 有空间、能联网、有调度器
#      本脚本          针对「home 配额满 + /data 大容量盘 + 无管理员」的环境
#
#  它会做什么（都在用户空间，不需要 sudo）：
#      1. 检查 python >= 3.10（Ubuntu 22.04 自带 3.10，本集群是 3.12，够用）
#      2. 把 pip 缓存 / venv / 模型缓存全部重定向到 --store（默认 /data/$USER/deeplearn）
#         —— 这是关键：torch 装完 3-5GB，写进 home 会把配额撑爆
#      3. 在 $STORE/venv 建虚拟环境，装 torch(CUDA 频道可指定) + requirements.txt
#      4. 把 deploy/sitecustomize.py 装进 venv，让 TORCH_HOME 等留在项目内
#      5. 生成 matplotlib 中文字体配置（没有中文字体时曲线图中文会变方框）
#      6. 在项目根做一个 .venv -> $STORE/venv 的软链，这样 ./run.sh 可以照常用
#      7. 跑一次自检（python / torch / cuda / GPU / 各依赖版本）
#
#  用法：
#      bash deploy/cluster_setup.sh                    # 默认 /data/$USER/deeplearn
#      STORE=/data/xxx/deeplearn bash deploy/cluster_setup.sh
#      TORCH_CUDA=cu124 bash deploy/cluster_setup.sh   # 手动指定 CUDA 频道
#      SKIP_RAG=1 bash deploy/cluster_setup.sh         # 跳过 RAG 依赖（默认就跳）
#
#  跑完之后：
#      ./run.sh project_cv/scripts/make_dummy_data.py
#      ./run.sh project_cv/src/train.py --data-dir data/_dummy --run-name smoke_test \
#          --arch smallcnn --no-pretrained --img-size 32 --epochs 2 --batch-size 16 --out-dir project_cv/outputs
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STORE="${STORE:-/data/${USER:-$(id -un)}/deeplearn}"
VENV="$STORE/venv"
PY="${PYTHON:-python3}"
TORCH_CUDA="${TORCH_CUDA:-}"
SKIP_RAG="${SKIP_RAG:-1}"

info() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[  ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

info "项目目录 : $ROOT"
info "存储目录 : $STORE"

# ---- 0. 磁盘检查 ----------------------------------------------------------
# home 只剩 12G 的话，往 home 里装 torch 是必炸的，先拦住。
HOME_AVAIL="$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2{print $4}')"
if [ -n "$HOME_AVAIL" ] && [ "$HOME_AVAIL" -lt 15728640 ]; then
    warn "home 可用空间不足 15GB（$(df -h "$HOME" | awk 'NR==2{print $4}')）——"
    warn "torch 装完约 3-5GB，加上 pip 缓存会更大，所以本脚本把环境装到 $STORE。"
fi

mkdir -p "$STORE" || fail "无法创建 $STORE（权限？换一个 STORE=... 再试）"
if [ "$STORE" = "$HOME" ] || case "$STORE" in "$HOME"/*) true;; *) false;; esac; then
    warn "STORE 在 home 下面，可能撞配额限制。建议用 /data/<你的目录>。"
fi

DATA_AVAIL="$(df -Pk "$STORE" | awk 'NR==2{print $4}')"
info "存储盘可用: $(df -h "$STORE" | awk 'NR==2{print $4}')"
if [ "${DATA_AVAIL:-0}" -lt 10485760 ]; then
    fail "$STORE 可用空间不足 10GB，装完 torch + 数据会写满。"
fi

# ---- 1. Python 版本 -------------------------------------------------------
command -v "$PY" >/dev/null 2>&1 || fail "找不到 $PY"
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PYMAJ="${PYVER%%.*}"; PYMIN="${PYVER#*.}"
info "Python 版本: $PYVER  ($(command -v "$PY"))"

if [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 10 ]; }; then
    fail "需要 Python >= 3.10，当前 $PYVER。没有 sudo 的话用 uv：
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source \$HOME/.local/bin/env
    uv python install 3.12
    PYTHON=\$(uv python find 3.12) bash deploy/cluster_setup.sh"
fi

if ! "$PY" -c 'import venv' >/dev/null 2>&1; then
    fail "$PY 缺 venv 模块（Ubuntu 上要 python3-venv，但你没 sudo）。用 uv 自带的 Python：
    curl -LsSf https://astral.sh/uv/install.sh | sh && source \$HOME/.local/bin/env
    uv python install 3.12 && PYTHON=\$(uv python find 3.12) bash deploy/cluster_setup.sh"
fi

# ---- 2. CUDA 频道选择 -----------------------------------------------------
# 这台机器：RTX 3090 = sm_86，驱动 580.178.04（CUDA 13.0），cu124/cu128 都能用。
if [ -n "$TORCH_CUDA" ]; then
    CHANNEL="$TORCH_CUDA"
    info "CUDA 频道: $CHANNEL（你手动指定）"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    COMPUTE="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
    info "GPU 算力: ${COMPUTE:-未知}   驱动: ${DRIVER:-未知}"
    case "${COMPUTE:-}" in
        12.*|13.*) CHANNEL="cu128" ;;                 # Blackwell 必须 cu128+
        8.*|9.*)   CHANNEL="cu124" ;;                 # Ampere(3090)/Ada(4090)
        7.*)       CHANNEL="cu121" ;;                 # V100/T4
        *)         CHANNEL="cu124" ;;
    esac
    info "CUDA 频道: $CHANNEL（按算力 $COMPUTE 自动选择）"
else
    warn "当前节点看不到 GPU（共享服务器上正常）。先按 cu124 装——"
    warn "如果之后发现本机是 50 系(Blackwell)，用 TORCH_CUDA=cu128 重装。"
    CHANNEL="cu124"
fi

# ---- 3. 缓存全部重定向到 $STORE（本脚本的核心价值）------------------------
export PIP_CACHE_DIR="$STORE/pip-cache"
export TMPDIR="$STORE/tmp"
export TORCH_HOME="$STORE/torch"
export HF_HOME="$STORE/huggingface"
export MPLCONFIGDIR="$STORE/matplotlib"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR"
ok "pip 缓存   -> $PIP_CACHE_DIR"
ok "模型缓存   -> $TORCH_HOME"

# ---- 4. 建 venv -----------------------------------------------------------
if [ -x "$VENV/bin/python" ]; then
    info "复用已有虚拟环境: $VENV（想重建先 rm -rf $VENV）"
else
    info "创建虚拟环境 -> $VENV"
    "$PY" -m venv "$VENV" || fail "创建 venv 失败"
fi
VPY="$VENV/bin/python"
[ -x "$VPY" ] || fail "虚拟环境不完整：找不到 $VPY"

info "升级 pip / setuptools / wheel"
"$VPY" -m pip install --upgrade pip setuptools wheel --quiet

# ---- 5. 安装 PyTorch ------------------------------------------------------
info "安装 PyTorch（$CHANNEL，约 1-3GB，第一次比较久）"
"$VPY" -m pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/$CHANNEL"

if [ -f "$ROOT/requirements.txt" ]; then
    info "安装 requirements.txt"
    "$VPY" -m pip install -r "$ROOT/requirements.txt"
fi

# RAG 依赖在共享服务器上默认跳过：要联网下 sentence-transformers / faiss(pypi) 等，
# 而且和 CV 实验无关。需要时 SKIP_RAG=0。
if [ "$SKIP_RAG" != "1" ] && [ -f "$ROOT/project_rag/requirements-rag.txt" ]; then
    info "安装 RAG 项目依赖"
    "$VPY" -m pip install -r "$ROOT/project_rag/requirements-rag.txt" || \
        warn "RAG 依赖安装失败，不影响 project_cv"
else
    info "跳过 RAG 依赖（需要时用 SKIP_RAG=0 重跑）"
fi

# ---- 6. sitecustomize（模型缓存留在项目内）--------------------------------
if [ -f "$ROOT/deploy/sitecustomize.py" ]; then
    SITE_DIR="$("$VPY" -c 'import site; print(site.getsitepackages()[0])')"
    cp "$ROOT/deploy/sitecustomize.py" "$SITE_DIR/sitecustomize.py"
    ok "已安装 sitecustomize.py -> $SITE_DIR"
fi

# ---- 7. matplotlib 中文字体 ----------------------------------------------
CJK_FONT=""
if command -v fc-list >/dev/null 2>&1; then
    CJK_FONT="$(fc-list :lang=zh family 2>/dev/null | head -n1 || true)"
fi
if [ -z "$CJK_FONT" ]; then
    warn "系统没有中文字体，曲线图里的中文会显示成方框。"
    warn "没有 sudo 的话可以把任意 .ttf 放到 ~/.fonts 再 fc-cache -f，"
    warn "或者干脆接受方框（不影响指标数字）。"
else
    ok "中文字体已就绪: $CJK_FONT"
fi
cat > "$MPLCONFIGDIR/matplotlibrc" <<'EOF'
## 由 deploy/cluster_setup.sh 生成
font.family: sans-serif
font.sans-serif: Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, Microsoft YaHei, SimHei, DejaVu Sans
axes.unicode_minus: False
figure.dpi: 110
savefig.dpi: 110
EOF
ok "matplotlib 配置 -> $MPLCONFIGDIR/matplotlibrc"

# ---- 8. 项目根做 .venv 软链，让 run.sh 继续可用 ---------------------------
if [ -e "$ROOT/.venv" ] && [ ! -L "$ROOT/.venv" ]; then
    warn "项目根已存在真实目录 .venv（可能是从本机拷来的），没有动它。"
    warn "如果它不能用，先 rm -rf .venv 再重跑本脚本。"
elif [ -L "$ROOT/.venv" ] && [ "$(readlink -f "$ROOT/.venv")" = "$(readlink -f "$VENV")" ]; then
    ok "软链已就绪: .venv -> $VENV"
else
    [ -L "$ROOT/.venv" ] && rm -f "$ROOT/.venv"
    ln -s "$VENV" "$ROOT/.venv"
    ok "已建立软链: .venv -> $VENV（这样 ./run.sh 和 README 里的命令都能直接用）"
fi

# ---- 9. 自检 --------------------------------------------------------------
echo
info "=============== 环境自检 ==============="
"$VPY" - <<'PYEOF'
import os, sys, warnings
warnings.filterwarnings("ignore")
print(f"  python        {sys.version.split()[0]}")
try:
    import torch
    print(f"  torch         {torch.__version__}")
    print(f"  cuda 可用     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU           {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  显存空闲      {free/2**20:.0f} MiB / {total/2**20:.0f} MiB")
        x = torch.randn(256, 256, device="cuda"); _ = x @ x
        print("  GPU 矩阵乘    通过")
    else:
        print("  (没用到 GPU；共享机器上如果卡被别人占满，也会是 False)")
except Exception as e:
    print(f"  torch 导入失败: {e}")
for mod in ("numpy", "pandas", "sklearn", "matplotlib", "PIL", "tqdm"):
    try:
        m = __import__(mod); print(f"  {mod:<13} {getattr(m, '__version__', '-')}")
    except Exception as e:
        print(f"  {mod:<13} 缺失 ({type(e).__name__})")
from matplotlib import rcParams
print(f"  中文字体      {rcParams['font.sans-serif'][0]}")
for k in ("PIP_CACHE_DIR", "TORCH_HOME", "HF_HOME", "MPLCONFIGDIR"):
    print(f"  {k:<13} {os.environ.get(k)}")
PYEOF

echo
ok "环境就绪。下一步跑烟雾测试："
cat <<EOF

  ./run.sh project_cv/scripts/make_dummy_data.py
  ./run.sh project_cv/src/train.py --data-dir data/_dummy --run-name smoke_test \\
      --arch smallcnn --no-pretrained --img-size 32 --epochs 2 --batch-size 16 \\
      --out-dir project_cv/outputs

  跑正式实验前先看显卡空不空：
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

  然后：bash deploy/cluster_train.sh
EOF

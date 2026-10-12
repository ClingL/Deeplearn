#!/usr/bin/env bash
# =============================================================================
#  在 Ubuntu / Debian 服务器上重建本项目的 Python 环境
#
#  用法：
#      cd deeplearn
#      bash setup.sh              # 全自动（推荐）
#      bash setup.sh --cpu        # 强制装 CPU 版
#      TORCH_CUDA=cu126 bash setup.sh   # 手动指定 CUDA 版本
#
#  脚本会做这些事：
#      1. 检查 Python 版本（需要 >= 3.10）
#      2. 用 nvidia-smi 识别 GPU 型号 / 驱动版本 / 算力等级
#      3. 根据算力等级自动选择匹配的 CUDA wheel 频道
#         （RTX 5090 等 Blackwell 是 sm_120，必须 cu128 以上）
#      4. 创建 .venv 并安装依赖
#      5. 安装 sitecustomize.py（把模型缓存留在项目内）
#      6. 配置 matplotlib 中文字体
#      7. 跑一次 GPU 自检
#
#  特点：不需要 sudo，不改系统 Python，全部装在项目目录里。
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="${PYTHON:-python3}"
FORCE_CPU=0

for arg in "$@"; do
    case "$arg" in
        --cpu) FORCE_CPU=1 ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $arg（可用: --cpu）" >&2; exit 1 ;;
    esac
done

info() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[  ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
info "项目目录: $ROOT"
# -----------------------------------------------------------------------------

# ---- 1. Python 版本 ----
command -v "$PY" >/dev/null 2>&1 || fail "找不到 $PY。Ubuntu 上装一下：
    sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"

PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PYMAJ="${PYVER%%.*}"
PYMIN="${PYVER#*.}"
info "Python 版本: $PYVER  ($(command -v "$PY"))"

if [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 10 ]; }; then
    cat >&2 <<EOF

[error] PyTorch 2.11 需要 Python >= 3.10，当前是 $PYVER。

Ubuntu 20.04 自带 3.8，太旧。三个办法：

  1) 用 conda（最省事）：
       conda create -n dl python=3.12 -y && conda activate dl
       PYTHON=\$(which python) bash setup.sh

  2) 装 deadsnakes 源（Ubuntu 20.04/22.04 都行）：
       sudo apt-get install -y software-properties-common
       sudo add-apt-repository -y ppa:deadsnakes/ppa
       sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv
       PYTHON=python3.12 bash setup.sh

  3) 用 uv（不用 sudo，会自己下 Python）：
       curl -LsSf https://astral.sh/uv/install.sh | sh
       uv python install 3.12
       PYTHON=\$(uv python find 3.12) bash setup.sh

EOF
    exit 1
fi

# ---- 2. 识别 GPU ----
GPU_NAME=""; DRIVER=""; COMPUTE=""; HAS_GPU=0

if [ "$FORCE_CPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        HAS_GPU=1
        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
        DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
        # compute_cap 需要较新的 nvidia-smi；拿不到就留空，后面按驱动版本判断
        COMPUTE="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
    else
        warn "找到 nvidia-smi 但它执行失败（驱动没装好？）。先按 CPU 版处理。"
    fi
fi

if [ "$HAS_GPU" -eq 1 ]; then
    info "GPU: ${GPU_NAME:-未知}"
    info "驱动版本: ${DRIVER:-未知}"
    info "算力等级: ${COMPUTE:-未知（nvidia-smi 太旧，按驱动版本判断）}"
else
    warn "没有检测到可用 GPU，将安装 CPU 版 PyTorch（能跑通，但训练极慢）。"
fi

# ---- 3. 选择 CUDA wheel 频道 ----
choose_channel() {
    if [ -n "${TORCH_CUDA:-}" ]; then
        printf '%s' "$TORCH_CUDA"; return
    fi
    if [ "$HAS_GPU" -eq 0 ]; then
        printf 'cpu'; return
    fi

    # Blackwell（RTX 50 系 / B200）= sm_120，必须 cu128 及以上
    case "${COMPUTE:-}" in
        12.*|13.*) printf 'cu128'; return ;;
    esac

    # 其余按驱动版本选：CUDA 12.8 需要 >= 570，12.6 需要 >= 560，
    # 12.4 需要 >= 550，12.1 需要 >= 530
    local major=""
    if [ -n "$DRIVER" ]; then
        major="${DRIVER%%.*}"
    fi
    if [ -z "$major" ]; then
        printf 'cu128'; return
    fi

    case "$major" in
        ''|*[!0-9]*) printf 'cu128'; return ;;
    esac

    if   [ "$major" -ge 570 ]; then printf 'cu128'
    elif [ "$major" -ge 560 ]; then printf 'cu126'
    elif [ "$major" -ge 550 ]; then printf 'cu124'
    elif [ "$major" -ge 530 ]; then printf 'cu121'
    else
        warn "驱动 $DRIVER 太旧（需要 >= 530）。要么升级驱动，要么用 --cpu。" >&2
        printf 'cu121'
    fi
}

CHANNEL="$(choose_channel)"

if [ "$CHANNEL" = "cpu" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    info "PyTorch 频道: CPU 版"
else
    TORCH_INDEX="https://download.pytorch.org/whl/${CHANNEL}"
    info "PyTorch 频道: ${CHANNEL}  (index: $TORCH_INDEX)"
    info "如需换版本: TORCH_CUDA=cu126 bash setup.sh"
fi

# ---- 4. 创建虚拟环境 ----
if [ -d "$VENV" ]; then
    info "已存在 $VENV，将复用（如需全新重建，先 rm -rf .venv）"
else
    info "创建虚拟环境 -> $VENV"
    if ! "$PY" -m venv "$VENV" 2>/dev/null; then
        cat >&2 <<EOF

[error] python3 -m venv 失败，通常是缺 python3-venv 包。

Ubuntu / Debian:
    sudo apt-get update && sudo apt-get install -y python3-venv python3-pip

装完再跑一次 bash setup.sh

EOF
        exit 1
    fi
fi

VPY="$VENV/bin/python"
[ -x "$VPY" ] || fail "虚拟环境不完整：找不到 $VPY"

info "升级 pip"
"$VPY" -m pip install --upgrade pip setuptools wheel --quiet

# ---- 5. 安装 PyTorch ----
info "安装 PyTorch（约 2-3 GB，第一次会比较久）"
"$VPY" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

# ---- 6. 安装其余依赖 ----
if [ -f "$ROOT/requirements.txt" ]; then
    info "安装 requirements.txt"
    "$VPY" -m pip install -r "$ROOT/requirements.txt"
fi

# RAG 项目（可选，缺包不影响 CV 项目）
if [ -f "$ROOT/project_rag/requirements-rag.txt" ]; then
    info "安装 RAG 项目依赖"
    if ! "$VPY" -m pip install -r "$ROOT/project_rag/requirements-rag.txt"; then
        warn "RAG 依赖安装失败。不影响 project_cv 和 lessons，可以稍后再试。"
    fi
fi

# ---- 7. 安装 sitecustomize（模型缓存留在项目内）----
if [ -f "$ROOT/deploy/sitecustomize.py" ]; then
    SITE_DIR="$("$VPY" -c 'import site; print(site.getsitepackages()[0])')"
    cp "$ROOT/deploy/sitecustomize.py" "$SITE_DIR/sitecustomize.py"
    ok "已安装 sitecustomize.py -> $SITE_DIR"
    ok "模型缓存在项目内：$ROOT/.cache"
else
    warn "没找到 deploy/sitecustomize.py，跳过（模型会缓存到 ~/.cache）"
fi

# ---- 8. matplotlib 中文字体 ----
CJK_FONT=""
if command -v fc-list >/dev/null 2>&1; then
    CJK_FONT="$(fc-list :lang=zh family 2>/dev/null | head -n1 || true)"
fi

if [ -z "$CJK_FONT" ]; then
    warn "系统没有中文字体。图表里的中文会显示成方框（豆腐块）。"
    warn "装一下（约 100MB）："
    warn "    sudo apt-get install -y fonts-noto-cjk"
    warn "装完重新跑一次 bash setup.sh 即可生效。"
else
    ok "中文字体已就绪: $CJK_FONT"
fi

mkdir -p "$ROOT/.cache/matplotlib"
cat > "$ROOT/.cache/matplotlib/matplotlibrc" <<'EOF'
## 项目级 matplotlib 配置（由 setup.sh 生成）
## matplotlib 导入时自动读取，不需要在脚本里写 rcParams。
## 默认字体 DejaVu Sans 没有中文字形，图表中文会变成方框，所以必须配。

font.family: sans-serif
font.sans-serif: Noto Sans CJK SC, Source Han Sans SC, WenQuanYi Zen Hei, Microsoft YaHei, SimHei, DejaVu Sans

axes.unicode_minus: False
figure.dpi: 110
savefig.dpi: 110
EOF
ok "matplotlib 配置 -> $ROOT/.cache/matplotlib/matplotlibrc"

# ---- 9. 自检 ----
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
        print(f"  算力等级      {torch.cuda.get_device_capability(0)}")
        x = torch.randn(1000, 1000, device="cuda")
        _ = x @ x
        print("  GPU 矩阵乘    通过")
    else:
        print("  (没有用到 GPU；如果是 CPU 版可以忽略)")
except Exception as e:
    print(f"  torch 导入失败: {e}")

for mod in ("numpy", "pandas", "sklearn", "matplotlib", "PIL"):
    try:
        m = __import__(mod)
        print(f"  {mod:<13} {getattr(m, '__version__', '-')}")
    except Exception as e:
        print(f"  {mod:<13} 缺失 ({type(e).__name__})")

from matplotlib import rcParams
print(f"  中文字体      {rcParams['font.sans-serif'][0]}")
for k in ("TORCH_HOME", "HF_HOME", "MPLCONFIGDIR"):
    print(f"  {k:<13} {os.environ.get(k)}")
PYEOF

echo
ok "环境就绪。"
cat <<'EOF'

接下来这样跑（在项目根目录）：

  # 跑课程
  ./run.sh lessons/01_tensors_autograd.py

  # CV 项目：先烟雾测试，再上真实数据集
  ./run.sh project_cv/scripts/make_dummy_data.py
  ./run.sh project_cv/src/train.py --data-dir data/_dummy --run-name smoke_test \
      --arch smallcnn --no-pretrained --img-size 32 --epochs 2 --batch-size 16

  # 把日志存成文件（UTF-8，方便回传排查）
  ./run.sh project_cv/src/train.py --data-dir data/flowers_split \
      --run-name 01_baseline --arch smallcnn --no-pretrained \
      --epochs 40 > outputs/01.log 2>&1

如果 ./setup.sh 报 "Permission denied"，用 bash setup.sh 代替。

注意：数据集不会跟着 git 走（.gitignore 里排除了 data/），
      需要单独传到服务器，见 DEPLOY.md。
EOF

#!/usr/bin/env bash
# =============================================================================
#  用项目自带的虚拟环境运行 Python 脚本（Linux 版，对应 Windows 的 run.cmd）
#
#  用法（在项目根目录）：
#      ./run.sh lessons/01_tensors_autograd.py
#      ./run.sh project_cv/src/train.py --data-dir data/flowers_split \
#          --run-name 01_baseline --arch smallcnn --no-pretrained --epochs 40
#
#  存日志（UTF-8，方便回传排查）：
#      ./run.sh project_cv/src/train.py ... > outputs/01.log 2>&1
#
#  如果提示 "Permission denied"，用 bash run.sh 代替。
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "[错误] 找不到虚拟环境解释器：" >&2
    echo "       $PY" >&2
    echo "       先运行 bash setup.sh 创建环境。" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    cat <<'EOF'
用法: ./run.sh <脚本路径> [参数...]

例子:
  ./run.sh lessons/01_tensors_autograd.py
  ./run.sh project_cv/scripts/audit_data.py --data-dir data/flowers
  ./run.sh project_cv/src/train.py --data-dir data/flowers_split \
      --run-name 01_baseline --arch smallcnn --no-pretrained --epochs 40
EOF
    exit 1
fi

# ---- 1) 强制 UTF-8 ----
# 服务器上常见 LC_ALL=C / POSIX locale，这时 Python 的 stdout 会是 ascii 编码，
# print 中文直接抛 UnicodeEncodeError 把脚本打断。必须显式指定。
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONDONTWRITEBYTECODE=1

# ---- 2) 模型缓存留在项目内 ----
# 不依赖 sitecustomize 也保险（万一它没装上）
export TORCH_HOME="$ROOT/.cache/torch"
export HF_HOME="$ROOT/.cache/huggingface"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
# HuggingFace 下载慢的话取消下面这行注释走镜像站
# export HF_ENDPOINT=https://hf-mirror.com

# ---- 3) 切到项目根，让 data/flowers 这类相对路径稳定生效 ----
cd "$ROOT"

echo "Running: python $*"
exec "$PY" "$@"

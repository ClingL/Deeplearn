#!/usr/bin/env bash
# =============================================================================
#  共享 GPU 服务器（无调度器）上顺序跑完 6 个 CV 实验
#
#  为什么是顺序跑而不是并行：
#      这台机器只有 1 张 RTX 3090，而且经常被别人占着（探测时 24G 里用了 20G）。
#      并行提交只会互相抢显存，全部 OOM。顺序跑、每个实验单独落日志最稳。
#
#  用法：
#      bash deploy/cluster_train.sh                  # 跑全部 6 个
#      bash deploy/cluster_train.sh smoke            # 只跑烟雾测试（2 epoch）
#      EXPERIMENTS="03_resnet_finetune 04_strong_aug" bash deploy/cluster_train.sh
#      EPOCHS=15 bash deploy/cluster_train.sh        # 先短跑看趋势
#      BS=16 bash deploy/cluster_train.sh            # 显存紧张时改 batch size
#
#  显存说明（重要）：
#      探测时卡上只剩约 4GB 空闲。224x224 + ResNet18 的经验值：
#          batch 16  ≈ 2.0-2.5GB      batch 32 ≈ 3.5-4.5GB
#          batch 64  ≈ 6.5-8GB  ← README 里的默认值，现在跑必 OOM
#      所以本脚本默认 BS=16（effnet_b0 用 16）。等卡空出来再放大重跑。
#
#      ⚠️ 改 batch size 会让结果和本机那 6 组（BS=64）**不能严格对比**。
#      README 里要写明「集群复现用 batch 16，学习率按 sqrt 缩放」。
#
#  跑完：
#      ./run.sh project_cv/scripts/make_report.py --outputs-dir project_cv/outputs --baseline 01_baseline
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="project_cv/outputs"          # 必须和本机那 6 组一致，否则 make_report 聚合不到
EPOCHS="${EPOCHS:-40}"
BS="${BS:-16}"
BS_EFF="${BS_EFF:-16}"
WORKERS="${WORKERS:-4}"
DATA_DIR="${DATA_DIR:-data/flowers_split}"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$LOG_DIR"

info() { printf '\033[1;34m[train]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- 前置检查
[ -x "$ROOT/.venv/bin/python" ] || {
    echo "[error] 找不到 .venv/bin/python，先跑：bash deploy/cluster_setup.sh" >&2
    exit 1
}

if [ ! -d "$DATA_DIR" ]; then
    echo "[error] 数据目录不存在：$DATA_DIR" >&2
    echo "        先按 DEPLOY.md / 集群文档把 Flower102 放好，或设 DATA_DIR=... 指定。" >&2
    exit 1
fi

info "项目目录 : $ROOT"
info "数据目录 : $DATA_DIR"
info "产物目录 : $OUT_DIR"
info "epochs=$EPOCHS  batch_size=$BS  (effnet=$BS_EFF)  workers=$WORKERS"

# ---------------------------------------------------------------- 显存体检
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    read -r USED TOTAL <<<"$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tr -d ',')"
    FREE=$(( TOTAL - USED ))
    info "GPU 显存: 已用 ${USED}MiB / 共 ${TOTAL}MiB，空闲约 ${FREE}MiB"
    if [ "$FREE" -lt 6000 ]; then
        warn "空闲显存不足 6GB，batch $BS 可能 OOM。"
        warn "应对：等别人跑完、或 BS=8、或加 --img-size 160 重跑（分辨率变了准确率会掉，要写进 README）。"
    fi
    # 卡被别人占着时，连 head 都可能失败，所以下面每个实验都容忍失败继续跑
else
    warn "当前节点看不到 GPU，训练会退化成 CPU —— 会非常慢，确认一下是不是选错了机器。"
fi

# ---------------------------------------------------------------- 单个实验
run_one() {
    local name="$1"; shift
    local log="$LOG_DIR/$name.log"

    if [ -f "$OUT_DIR/$name/metrics.json" ]; then
        info "跳过 $name（已有 metrics.json，想重跑先删掉 $OUT_DIR/$name）"
        return 0
    fi

    info "===== $name 开始 $(date '+%F %T') ====="
    info "日志: $log"

    # 用 run.sh 保证 UTF-8 + 模型缓存留在项目内；失败只警告不中断整批
    if "$ROOT/run.sh" project_cv/src/train.py \
            --data-dir "$DATA_DIR" \
            --run-name "$name" \
            --out-dir "$OUT_DIR" \
            --epochs "$EPOCHS" \
            --num-workers "$WORKERS" \
            "$@" 2>&1 | tee "$log"; then
        info "===== $name 完成 $(date '+%F %T') ====="
    else
        warn "$name 失败（退出码非 0），详见 $log；继续跑下一个"
    fi
}

# ---------------------------------------------------------------- 实验清单
# 参数和本机跑的 6 组一致（见 project_cv/outputs/*/config.json），
# 只有 batch size 因为显存被降下来了。
run_01() { run_one 01_baseline        --arch smallcnn       --no-pretrained --augment basic  --freeze-epochs 0  --lr 1e-3 --batch-size "$BS"; }
run_02() { run_one 02_resnet_frozen   --arch resnet18      --pretrained    --augment basic  --freeze-epochs 40 --lr 1e-3 --batch-size "$BS"; }
run_03() { run_one 03_resnet_finetune --arch resnet18      --pretrained    --augment basic  --freeze-epochs 3  --backbone-lr 1e-4 --batch-size "$BS"; }
run_04() { run_one 04_strong_aug      --arch resnet18      --pretrained    --augment strong --freeze-epochs 3  --backbone-lr 1e-4 --batch-size "$BS"; }
run_05() { run_one 05_effnetb0        --arch efficientnet_b0 --pretrained  --augment strong --freeze-epochs 3  --backbone-lr 1e-4 --batch-size "$BS_EFF"; }
run_06() { run_one 06_no_aug          --arch resnet18      --pretrained    --augment none   --freeze-epochs 3  --backbone-lr 1e-4 --batch-size "$BS"; }

# ---------------------------------------------------------------- 入口
MODE="${1:-all}"
case "$MODE" in
    smoke)
        info "烟雾测试：2 epoch / 32x32 / dummy 数据"
        run_one smoke_test --arch smallcnn --no-pretrained --img-size 32 --epochs 2 \
                           --batch-size 16 --log-every 10
        ;;
    all)
        if [ -n "${EXPERIMENTS:-}" ]; then
            for e in $EXPERIMENTS; do
                case "$e" in
                    01*|01) run_01 ;;
                    02*|02) run_02 ;;
                    03*|03) run_03 ;;
                    04*|04) run_04 ;;
                    05*|05) run_05 ;;
                    06*|06) run_06 ;;
                    *) warn "未知实验名: $e（可用 01..06）" ;;
                esac
            done
        else
            run_01; run_02; run_03; run_04; run_05; run_06
        fi
        ;;
    *)
        echo "用法: bash deploy/cluster_train.sh [all|smoke]" >&2
        echo "      EXPERIMENTS=\"03_resnet_finetune 04_strong_aug\" bash deploy/cluster_train.sh" >&2
        exit 1
        ;;
esac

echo
info "全部结束。聚合消融表："
echo "  ./run.sh project_cv/scripts/make_report.py --outputs-dir $OUT_DIR --baseline 01_baseline"
echo "  ./run.sh project_cv/src/evaluate.py --ckpt $OUT_DIR/04_strong_aug/best.pt --data-dir $DATA_DIR"

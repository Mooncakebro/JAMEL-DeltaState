#!/bin/bash
# run_eval.sh — 唯一的 eval 入口（小规模 / 全量 / 并行 / 串行 共用）。
#
# 总并行度 = NUM_GPUS * WORKERS_PER_GPU。串行只是 NUM_GPUS=1 WORKERS_PER_GPU=1
# 的特殊情况；不再单独维护脚本。
#
# 每个 worker 是一个独立 python 进程：自己的模型副本 + 自己的浏览器 + 自己的端口。
# 多个 worker 可以共享同一张 GPU（通过 CUDA_VISIBLE_DEVICES 隔离 + 各自独立显存）。
# App 列表按 round-robin 平均分到所有 worker。
#
# recipe 项（采样、图像、memory、prompt 格式、max_input_tokens）严格与训练对齐。
#
# App 集合：
#   - test10 apps：论文表格/曲线口径。本脚本默认就是这一组。
#       vipshop alibaba expedia taobao pinduoduo dongchedi youku keep meituan temu
#   - train86 apps：ScaleWoB 96 apps 去除 test10 后的训练应用集合。
#
# 用法：
#   # 默认：test10 apps，8 GPU × 2 workers/GPU
#   bash shell/run_eval.sh
#
#   # 单 train app debug：单卡单进程
#   APPS=youdao NUM_GPUS=1 WORKERS_PER_GPU=1 \
#     EVAL_OUTPUT=outputs/eval_youdao bash shell/run_eval.sh
#
#   # 全量 train86 apps（训练应用 sanity check）：
#   APPS_MODE=train86 bash shell/run_eval.sh
#
#   # 自定义子集
#   APPS="alibaba jd taobao" NUM_GPUS=4 WORKERS_PER_GPU=1 bash shell/run_eval.sh
#
#   # 指定 ckpt（默认取 OUTPUT_DIR 下 global_step 最大者）
#   CHECKPOINT=outputs/.../global_step_1468 bash shell/run_eval.sh
#
#   # 发布模型目录（actor + compressor）
#   MODEL_PATH=/path/to/jamel_model bash shell/run_eval.sh
#
#   # 显存够多时拉到 3 workers/GPU
#   WORKERS_PER_GPU=3 bash shell/run_eval.sh
#
# 关键路径（可环境变量覆盖）：
#   OUTPUT_DIR        outputs/jamel_sft_ckpt_react_full     (SFT 输出目录)
#   MODEL_PATH        可选：包含 actor/ 和 compressor/ 的发布模型目录
#   CHECKPOINT        最新 global_step_*，或发布模型目录里的 actor 子目录
#   COMPRESSOR_MODEL  本地 Qwen3-VL compressor 目录；MODEL_PATH 会默认使用其 compressor 子目录
#   JAMEL_BASE_MODEL  可选：checkpoint metadata 中 base_model_name_or_path
#                     指向旧机器路径时，用本地 base model 路径覆盖。
#   APPS              显式 app 列表（覆盖 APPS_MODE）
#   APPS_MODE         test10(默认) | train86 | all
#                       - test10: 论文 test10 app
#                       - train86: 训练应用集合
#                       - all:    ScaleWoB 96 apps（不推荐作为论文指标）
#   MAX_STEPS         50         每个 session 的总步数（含所有 episode）
#   NUM_SESSIONS      3          每个 app 独立跑多少个 session
#   EVAL_OUTPUT       outputs/eval_react_full
#   NUM_GPUS          8          GPU 数（>=1）
#   WORKERS_PER_GPU   2          每张卡的独立 python 进程数（>=1）
#   PORT_BASE         8800       worker $i 用 $((PORT_BASE + i))
#   TEMPERATURE       0.8        sampling temperature
#   TOP_P             0.9
#   SCALEWOB_ROOT     env/browser_env/scalewob-env  ScaleWoB 静态文件目录
#
# 显存预算（参考）：7B bf16 ≈ 14GB + Qwen3-VL-2B compressor ≈ 4GB +
#   推理激活/缓存 ≈ 4-8GB ≈ 单 worker 22-26GB。A800 80GB 可放 3 个 worker。
#   显存不够就调小 WORKERS_PER_GPU。

set -euo pipefail

EXTRA_EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat <<'EOF'
Usage:
  [ENV=VALUE ...] bash shell/run_eval.sh

Configuration is provided through environment variables:
  MODEL_PATH        Released JAMEL model directory containing actor/ and compressor/.
  CHECKPOINT        Actor checkpoint. Required unless MODEL_PATH is set or
                    OUTPUT_DIR contains global_step_* checkpoints.
  COMPRESSOR_MODEL  Local Qwen3-VL compressor model directory. Required when
                    using CHECKPOINT instead of MODEL_PATH.
  JAMEL_BASE_MODEL  Legacy-only local base model override for relocated checkpoints.
                    Released JAMEL model directories do not require this.
  APPS              Optional explicit app list, overrides APPS_MODE.
  APPS_MODE         test10 | train86 | all. Default: test10.
  MAX_STEPS         Default: 50.
  NUM_SESSIONS      Default: 3.
  NUM_GPUS          Default: 8.
  WORKERS_PER_GPU   Default: 2.
  EVAL_OUTPUT       Output directory.
  SCALEWOB_ROOT     ScaleWoB static app directory.
  PYTHON_BIN        Optional Python executable. Defaults to .venv/bin/python.

Setup:
  uv sync --locked --python 3.10 --extra dev --extra train
  uv run playwright install chromium
  sudo apt-get install -y fontconfig fonts-noto-cjk fonts-noto-color-emoji
EOF
            exit 0
            ;;
        +model.memory_augment.max_hybrid_recent_items=*)
            MAX_HYBRID_RECENT_ITEMS="${1#*=}"
            shift
            ;;
        --max-hybrid-recent-items)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --max-hybrid-recent-items requires a value." >&2
                exit 2
            fi
            MAX_HYBRID_RECENT_ITEMS="$2"
            shift 2
            ;;
        --max-hybrid-recent-items=*)
            MAX_HYBRID_RECENT_ITEMS="${1#*=}"
            shift
            ;;
        *)
            echo "ERROR: run_eval.sh does not accept positional arguments: $*" >&2
            echo "Use environment variables, --max-hybrid-recent-items, or +model.memory_augment.max_hybrid_recent_items=N." >&2
            exit 2
            ;;
    esac
done

JAMEL_ROOT=${JAMEL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
VERL_AGENT_ROOT=${VERL_AGENT_ROOT:-$(cd "$JAMEL_ROOT/third_party/verl-agent" && pwd)}
export PYTHONPATH="$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$JAMEL_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$JAMEL_ROOT/.venv/bin/python"
    else
        echo "ERROR: JAMEL Python environment not found: $JAMEL_ROOT/.venv/bin/python" >&2
        echo "Run setup first:" >&2
        echo "  cd $JAMEL_ROOT" >&2
        echo "  uv sync --locked --python 3.10 --extra dev --extra train" >&2
        echo "  uv run playwright install chromium" >&2
        exit 2
    fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    RESOLVED_PYTHON_BIN=$(command -v "$PYTHON_BIN" || true)
    if [[ -n "$RESOLVED_PYTHON_BIN" && -x "$RESOLVED_PYTHON_BIN" ]]; then
        PYTHON_BIN="$RESOLVED_PYTHON_BIN"
    else
        echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
        exit 2
    fi
fi

OUTPUT_DIR=${OUTPUT_DIR:-"$JAMEL_ROOT/outputs/jamel_sft_ckpt_react_full"}
MODEL_PATH_FOR_EVAL=${MODEL_PATH:-${JAMEL_MODEL_PATH:-${JAMEL_MODEL:-}}}
COMPRESSOR_MODEL=${COMPRESSOR_MODEL:-}
MAX_STEPS=${MAX_STEPS:-50}
NUM_SESSIONS=${NUM_SESSIONS:-3}
EVAL_OUTPUT=${EVAL_OUTPUT:-"$JAMEL_ROOT/outputs/eval_react_full"}
NUM_GPUS=${NUM_GPUS:-8}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
PORT_BASE=${PORT_BASE:-8800}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.9}
MEMORY_MAX_ITEMS=${MEMORY_MAX_ITEMS:-512}
MEMORY_BUILDER=${MEMORY_BUILDER:-online_tokens}
DELTA_RANK=${DELTA_RANK:-8}
DELTA_MEMORY_SLOTS=${DELTA_MEMORY_SLOTS:-8}
DELTA_SEED=${DELTA_SEED:-13}
HYBRID_RECENT_ITEMS=${HYBRID_RECENT_ITEMS:-32}
MAX_HYBRID_RECENT_ITEMS=${MAX_HYBRID_RECENT_ITEMS:-$HYBRID_RECENT_ITEMS}
APPS_MODE=${APPS_MODE:-test10}
SCALEWOB_ROOT=${SCALEWOB_ROOT:-"$JAMEL_ROOT/env/browser_env/scalewob-env"}
APP_CONFIG=${APP_CONFIG:-"$JAMEL_ROOT/configs/benchmark_apps.json"}

for _positive_name in MAX_STEPS NUM_SESSIONS NUM_GPUS WORKERS_PER_GPU; do
    _positive_value=${!_positive_name}
    if [[ ! "$_positive_value" =~ ^[0-9]+$ || "$_positive_value" -lt 1 ]]; then
        echo "ERROR: $_positive_name must be a positive integer, got '$_positive_value'." >&2
        exit 2
    fi
done
unset _positive_name _positive_value

if [[ "$NUM_GPUS" -lt 1 || "$WORKERS_PER_GPU" -lt 1 ]]; then
    echo "ERROR: NUM_GPUS and WORKERS_PER_GPU must both be >= 1." >&2
    exit 2
fi

if [[ ! -d "$SCALEWOB_ROOT" ]]; then
    echo "ERROR: ScaleWoB root not found: $SCALEWOB_ROOT" >&2
    echo "Run: bash shell/download_scalewob_env.sh --mode all" >&2
    exit 2
fi

if ! command -v fc-list >/dev/null 2>&1; then
    echo "ERROR: fontconfig is not installed; Chinese UI text may render as boxes." >&2
    echo "Install system fonts before evaluation:" >&2
    echo "  sudo apt-get install -y fontconfig fonts-noto-cjk fonts-noto-color-emoji" >&2
    exit 2
fi
if ! fc-list :lang=zh >/dev/null 2>&1 || ! fc-list :lang=zh | grep -q .; then
    echo "ERROR: no Chinese-capable system font was found; Chinese UI text may render as boxes." >&2
    echo "Install system fonts before evaluation:" >&2
    echo "  sudo apt-get install -y fontconfig fonts-noto-cjk fonts-noto-color-emoji" >&2
    echo "  fc-cache -fv" >&2
    exit 2
fi

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = [
    "numpy",
    "torch",
    "transformers",
    "playwright",
    "browsergym",
    "pandas",
    "pyarrow",
    "PIL",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if missing:
    print("ERROR: the selected Python environment is missing runtime dependencies.", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print("Run:", file=sys.stderr)
    print("  uv sync --locked --python 3.10 --extra dev --extra train", file=sys.stderr)
    print("or set PYTHON_BIN to a fully synchronized JAMEL environment.", file=sys.stderr)
    sys.exit(2)

import torch
print(f"[env-check] python={sys.executable}", flush=True)
print(f"[env-check] torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}", flush=True)
PY

EVAL_SCRIPT="$JAMEL_ROOT/jamel/utils/eval/eval_memory_aug_episode.py"

# Resolve released model layout:
#   MODEL_PATH/
#     actor/
#     compressor/
if [[ -n "${CHECKPOINT:-}" && -d "$CHECKPOINT/actor" && -d "$CHECKPOINT/compressor" ]]; then
    MODEL_PATH_FOR_EVAL="$CHECKPOINT"
    CHECKPOINT=""
fi
if [[ -n "$MODEL_PATH_FOR_EVAL" ]]; then
    if [[ ! -d "$MODEL_PATH_FOR_EVAL" ]]; then
        echo "ERROR: MODEL_PATH not found: $MODEL_PATH_FOR_EVAL" >&2
        exit 2
    fi
    if [[ -z "${CHECKPOINT:-}" ]]; then
        CHECKPOINT="$MODEL_PATH_FOR_EVAL/actor"
    fi
    if [[ -z "$COMPRESSOR_MODEL" ]]; then
        COMPRESSOR_MODEL="$MODEL_PATH_FOR_EVAL/compressor"
    fi
    if [[ ! -d "$COMPRESSOR_MODEL" ]]; then
        echo "ERROR: model compressor not found: $COMPRESSOR_MODEL" >&2
        echo "Expected layout: $MODEL_PATH_FOR_EVAL/{actor,compressor}" >&2
        exit 2
    fi
fi

# Resolve checkpoint.
if [[ -z "${CHECKPOINT:-}" ]]; then
    CHECKPOINT=$(ls -d "$OUTPUT_DIR"/global_step_* 2>/dev/null \
                 | awk -F'global_step_' '{print $2"\t"$0}' \
                 | sort -n -k1 | tail -1 | cut -f2)
fi
if [[ -z "$CHECKPOINT" || ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: no checkpoint found. Set CHECKPOINT=... or train first." >&2
    echo "  searched OUTPUT_DIR=$OUTPUT_DIR" >&2
    exit 2
fi
if [[ -z "$COMPRESSOR_MODEL" ]]; then
    echo "ERROR: COMPRESSOR_MODEL is required when MODEL_PATH is not set." >&2
    echo "Use MODEL_PATH=/path/to/jamel_model, or set COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct for a bare actor checkpoint." >&2
    exit 2
fi
if [[ ! -d "$COMPRESSOR_MODEL" ]]; then
    echo "ERROR: COMPRESSOR_MODEL must be a local directory: $COMPRESSOR_MODEL" >&2
    exit 2
fi

# Resolve app list.
# Priority: explicit APPS=... overrides APPS_MODE.
if [[ -n "${APPS:-}" ]]; then
    read -r -a APP_LIST <<< "$APPS"
    APP_SOURCE="explicit (APPS=...)"
else
    case "$APPS_MODE" in
        test10)
            read -r -a APP_LIST <<< "$("$PYTHON_BIN" "$JAMEL_ROOT/scripts/print_app_split.py" test10 --config "$APP_CONFIG")"
            APP_SOURCE="test10"
            ;;
        train86)
            read -r -a APP_LIST <<< "$("$PYTHON_BIN" "$JAMEL_ROOT/scripts/print_app_split.py" train86 --config "$APP_CONFIG")"
            APP_SOURCE="train86"
            ;;
        all)
            read -r -a APP_LIST <<< "$("$PYTHON_BIN" "$JAMEL_ROOT/scripts/print_app_split.py" all --config "$APP_CONFIG")"
            APP_SOURCE="all ScaleWoB 96 apps [WARNING: do not report as test10 metric]"
            ;;
        *)
            echo "ERROR: APPS_MODE must be one of: test10|train86|all (got '$APPS_MODE')." >&2
            exit 2
            ;;
    esac
fi
NUM_APPS=${#APP_LIST[@]}
if [[ $NUM_APPS -eq 0 ]]; then
    echo "ERROR: no apps to evaluate." >&2
    exit 2
fi

TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
# Cap workers at app count to avoid empty workers.
if [[ $TOTAL_WORKERS -gt $NUM_APPS ]]; then
    TOTAL_WORKERS=$NUM_APPS
fi

mkdir -p "$EVAL_OUTPUT"

echo "=== eval ==="
echo "  ckpt:            $CHECKPOINT"
if [[ -n "$MODEL_PATH_FOR_EVAL" ]]; then
    echo "  model path:      $MODEL_PATH_FOR_EVAL"
fi
echo "  app source:      $APP_SOURCE"
echo "  apps:            $NUM_APPS"
echo "  GPUs:            $NUM_GPUS"
echo "  workers/GPU:     $WORKERS_PER_GPU"
echo "  total workers:   $TOTAL_WORKERS"
echo "  sessions:        $NUM_SESSIONS per app"
echo "  max_steps:       $MAX_STEPS per session (agent may reset() to start new episodes)"
echo "  sampling:        temperature=$TEMPERATURE  top_p=$TOP_P"
echo "  memory builder:  $MEMORY_BUILDER"
echo "  delta state:     rank=$DELTA_RANK slots=$DELTA_MEMORY_SLOTS seed=$DELTA_SEED hybrid_recent=$HYBRID_RECENT_ITEMS max_hybrid_recent=$MAX_HYBRID_RECENT_ITEMS"
echo "  scalewob_root:   $SCALEWOB_ROOT"
if [[ -n "${JAMEL_BASE_MODEL:-}" ]]; then
    echo "  base model:      $JAMEL_BASE_MODEL"
fi
echo "  python:          $PYTHON_BIN"
echo "  output:          $EVAL_OUTPUT"
echo

# Round-robin distribute apps across ALL workers (not just GPUs), so multiple
# workers on the same GPU each get their own slice and run in parallel.
declare -a CHUNKS
for ((w=0; w<TOTAL_WORKERS; w++)); do CHUNKS[w]=""; done
for ((idx=0; idx<NUM_APPS; idx++)); do
    w=$((idx % TOTAL_WORKERS))
    CHUNKS[w]+=" ${APP_LIST[idx]}"
done

PIDS=()
WORKER_LABELS=()
for ((w=0; w<TOTAL_WORKERS; w++)); do
    chunk_apps="${CHUNKS[w]# }"
    if [[ -z "$chunk_apps" ]]; then continue; fi
    GPU_ID=$((w % NUM_GPUS))
    SLOT=$((w / NUM_GPUS))
    PORT=$((PORT_BASE + w))
    LABEL="gpu${GPU_ID}_slot${SLOT}"
    LOG="$EVAL_OUTPUT/worker_${LABEL}.log"
    echo "[launch] worker $w  GPU $GPU_ID slot $SLOT  port $PORT  apps:$chunk_apps"

    # Pin worker to one physical GPU via CUDA_VISIBLE_DEVICES; inside the process
    # that GPU is index 0. Multiple workers on the same GPU each allocate their
    # own model copy and KV cache.
    # --env-ids triggers run_all_apps: model loads ONCE per worker, apps run
    # sequentially inside the worker, memory resets between apps.
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$PYTHON_BIN" "$EVAL_SCRIPT" \
        --checkpoint        "$CHECKPOINT" \
        --compressor-model  "$COMPRESSOR_MODEL" \
        --env-ids           $chunk_apps \
        --gpu-id            0 \
        --max-steps         "$MAX_STEPS" \
        --num-sessions      "$NUM_SESSIONS" \
        --memory-max-items  "$MEMORY_MAX_ITEMS" \
        --memory-builder    "$MEMORY_BUILDER" \
        --delta-rank        "$DELTA_RANK" \
        --delta-memory-slots "$DELTA_MEMORY_SLOTS" \
        --delta-seed        "$DELTA_SEED" \
        --hybrid-recent-items "$HYBRID_RECENT_ITEMS" \
        --max-hybrid-recent-items "$MAX_HYBRID_RECENT_ITEMS" \
        --output            "$EVAL_OUTPUT" \
        --port              "$PORT" \
        --temperature       "$TEMPERATURE" \
        --top-p             "$TOP_P" \
        --viewport-width    1280 \
        --viewport-height   720 \
        --model-image-width  640 \
        --model-image-height 360 \
        --scalewob-root     "$SCALEWOB_ROOT" \
        --seed              "$w" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    WORKER_LABELS+=("$LABEL")
done

echo
echo "Launched ${#PIDS[@]} workers."
echo "Logs: $EVAL_OUTPUT/worker_*.log"
echo "Waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[worker ${WORKER_LABELS[$i]}] done OK"
    else
        ec=$?
        echo "[worker ${WORKER_LABELS[$i]}] FAILED (exit $ec)" >&2
        FAILED=1
    fi
done

echo
echo "=== Aggregate ==="
"$PYTHON_BIN" - <<PY
import json, glob, os
base = "$EVAL_OUTPUT"
results = []
for app_dir in sorted(os.listdir(base)):
    full = os.path.join(base, app_dir)
    if not os.path.isdir(full): continue
    files = glob.glob(f"{full}/summary_*.json")
    if not files: continue
    s = json.load(open(sorted(files)[-1]))
    results.append(s)
results.sort(key=lambda x: -x['cumulative_reward'])
print(f"{'App':<22} {'Reward':>8} {'TotalCov':>10} {'Episodes':>10} {'Steps':>7}")
print('-'*60)
for s in results:
    cov = sum(s['coverage_delta_scores'])
    print(f"{s['env_id']:<22} {s['cumulative_reward']:>8.1f} {cov:>10} {s['episodes']:>10} {s['steps']:>7}")
total_r = sum(s['cumulative_reward'] for s in results)
total_c = sum(sum(s['coverage_delta_scores']) for s in results)
print('-'*60)
print(f"{'TOTAL':<22} {total_r:>8.1f} {total_c:>10} {'':>10} {sum(s['steps'] for s in results):>7}")
print(f"\nApps evaluated: {len(results)}")
agg = {
    'checkpoint': "$CHECKPOINT",
    'app_source': "$APP_SOURCE",
    'prompt_format': 'web_prompt',
    'memory_builder': "$MEMORY_BUILDER",
    'delta_rank': int("$DELTA_RANK"),
    'delta_memory_slots': int("$DELTA_MEMORY_SLOTS"),
    'hybrid_recent_items': int("$HYBRID_RECENT_ITEMS"),
    'max_hybrid_recent_items': int("$MAX_HYBRID_RECENT_ITEMS"),
    'max_steps_per_session': $MAX_STEPS,
    'sessions_per_app': $NUM_SESSIONS,
    'total_apps': len(results),
    'total_reward': total_r,
    'total_coverage': total_c,
    'per_app': results,
}
out = os.path.join(base, "aggregate_results.json")
with open(out, "w") as f:
    json.dump(agg, f, indent=2, ensure_ascii=False)
print(f"Saved: {out}")
PY

exit $FAILED

#!/bin/bash
# run_qwen25vl_7b_sft.sh
# Full SFT with Qwen2.5-VL-7B-Instruct + MemoryAugmentedCausalLM.
# The paper-scale recipe uses 8 GPUs; override NPROC_PER_NODE for other setups.
#
# Single source of truth: training and inference share `web_prompt.py`.
# Image input is resized to 640x360 in the dataset (viewport stays 1280x720).
# Response format is `<action>...</action>` (no `<think>` block).
#
# Context length (MAX_LENGTH):
#   MAX_LENGTH=8192 (default) — max token length for prompt + response
#
# Gradient checkpointing (GRADIENT_CHECKPOINTING):
#   GRADIENT_CHECKPOINTING=1 (default) — REQUIRED for max_length=8192 + multimodal.
#   GRADIENT_CHECKPOINTING=0           — only safe for shorter contexts.
#
# Batch size:
#   NPROC_PER_NODE (default 8)
#   MICRO_BATCH_SIZE_PER_GPU (default 1)
#   TRAIN_BATCH_SIZE (default 16)
#
# Validation frequency:
#   VAL_STEPS (default 100)
#
# Checkpoints:
#   SAVE_TOTAL_LIMIT (default 6) — number of checkpoints to retain.
#
# Final packaged model:
#   OUTPUT_MODEL_PATH      optional output JAMEL model path for evaluation.
#   COMPRESSOR_MODEL       required local Qwen3-VL compressor dir when OUTPUT_MODEL_PATH is set.
#
# Examples:
#   TRAIN_FILE=data/jamel_sft_data/jamel_memory_sft_train.parquet \
#   VAL_FILE=data/jamel_sft_data/jamel_memory_sft_val.parquet \
#   OUTPUT_DIR=outputs/jamel_sft_ckpt TOTAL_EPOCHS=3 \
#       bash run_qwen25vl_7b_sft.sh
#   COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct \
#   OUTPUT_MODEL_PATH=outputs/jamel_model \
#       bash run_qwen25vl_7b_sft.sh
set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JAMEL_ROOT=${JAMEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
VERL_AGENT_ROOT=${VERL_AGENT_ROOT:-$(cd "$JAMEL_ROOT/third_party/verl-agent" && pwd)}

export PYTHONPATH="$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH}"

BASE_MODEL_PATH=${BASE_MODEL_PATH:-${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}}
MEMORY_HIDDEN_SIZE=${MEMORY_HIDDEN_SIZE:-2048}
MEMORY_MAX_ITEMS=${MEMORY_MAX_ITEMS:-512}
MEMORY_BUILDER=${MEMORY_BUILDER:-online_tokens}
DELTA_RANK=${DELTA_RANK:-8}
DELTA_MEMORY_SLOTS=${DELTA_MEMORY_SLOTS:-8}
DELTA_SEED=${DELTA_SEED:-13}
READ_WITH_CURRENT_QUERY=${READ_WITH_CURRENT_QUERY:-1}
MEMORY_BUILDER_NORMALIZED=${MEMORY_BUILDER//-/_}
if [[ -z "${ONLINE_DELTA_STATE:-}" ]]; then
    if [[ "$MEMORY_BUILDER_NORMALIZED" == "online_delta_state" ]]; then
        ONLINE_DELTA_STATE=1
    else
        ONLINE_DELTA_STATE=0
    fi
fi
MAX_LENGTH=${MAX_LENGTH:-8192}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}

VAL_STEPS=${VAL_STEPS:-100}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-8}

TRAIN_FILE=${TRAIN_FILE:-}
VAL_FILE=${VAL_FILE:-}
OUTPUT_DIR=${OUTPUT_DIR:-"$JAMEL_ROOT/outputs/jamel_sft_ckpt"}
COMPRESSOR_MODEL=${COMPRESSOR_MODEL:-}
OUTPUT_MODEL_PATH=${OUTPUT_MODEL_PATH:-}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25vl_7b_memory_aug_sft_web_prompt}

if [[ -z "$TRAIN_FILE" ]]; then
    echo "ERROR: TRAIN_FILE is required." >&2
    exit 2
fi
if [[ -z "$VAL_FILE" ]]; then
    echo "ERROR: VAL_FILE is required." >&2
    exit 2
fi
if [[ "$TRAIN_FILE" != *"://"* && ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2
    exit 2
fi
if [[ "$VAL_FILE" != *"://"* && ! -f "$VAL_FILE" ]]; then
    echo "ERROR: VAL_FILE not found: $VAL_FILE" >&2
    exit 2
fi
if [[ -n "$OUTPUT_MODEL_PATH" ]]; then
    if [[ -z "$COMPRESSOR_MODEL" ]]; then
        echo "ERROR: OUTPUT_MODEL_PATH requires COMPRESSOR_MODEL to be set." >&2
        exit 2
    fi
    if [[ ! -d "$COMPRESSOR_MODEL" ]]; then
        echo "ERROR: COMPRESSOR_MODEL must be a local directory when packaging the final model: $COMPRESSOR_MODEL" >&2
        exit 2
    fi
fi

export COMPRESSOR_MODEL
export OUTPUT_MODEL_PATH

# Pick a free port (avoid 18001 held by orterun)
MASTER_PORT=${MASTER_PORT:-$(python3 -c "
import socket, random
for port in random.sample(range(29500, 30500), 200):
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        s.bind(('127.0.0.1', port))
        s.close()
        print(port)
        break
    except OSError:
        pass
")}
export MASTER_PORT
export MASTER_ADDR=127.0.0.1

# wandb offline: prevents NCCL watchdog hang on no-internet machines.
export WANDB_MODE=offline

torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master-addr=127.0.0.1 \
    --master-port=${MASTER_PORT} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.max_length=${MAX_LENGTH} \
    data.truncation=right \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.prompt_key=prompt \
    data.response_key=response \
    "data.custom_cls.path=file://$JAMEL_ROOT/jamel/train/memory/jamel_sft_dataset.py" \
    data.custom_cls.name=JAMELMemoryVLTokenSFTDataset \
    "+data.memory_max_items=$MEMORY_MAX_ITEMS" \
    "+data.memory_hidden_size=$MEMORY_HIDDEN_SIZE" \
    "+data.use_online_delta_state=$ONLINE_DELTA_STATE" \
    model.partial_pretrain="$BASE_MODEL_PATH" \
    "model.custom_cls.path=file://$JAMEL_ROOT/jamel/train/memory/modeling.py" \
    model.custom_cls.name=MemoryAugmentedCausalLM \
    model.memory_augment.memory_hidden_size="$MEMORY_HIDDEN_SIZE" \
    "+model.memory_augment.memory_builder=$MEMORY_BUILDER" \
    "+model.memory_augment.enable_online_delta_state=$ONLINE_DELTA_STATE" \
    "+model.memory_augment.delta_rank=$DELTA_RANK" \
    "+model.memory_augment.delta_memory_slots=$DELTA_MEMORY_SLOTS" \
    "+model.memory_augment.delta_seed=$DELTA_SEED" \
    "+model.memory_augment.read_with_current_query=$READ_WITH_CURRENT_QUERY" \
    model.enable_gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    model.strategy=fsdp2 \
    model.lora_rank=0 \
    use_remove_padding=False \
    optim.lr=2e-5 \
    optim.warmup_steps_ratio=0.05 \
    optim.lr_scheduler=cosine \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.project_name='jamel-sft' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    "trainer.logger=[console,wandb]" \
    "+trainer.val_steps=$VAL_STEPS" \
    "+trainer.save_total_limit=$SAVE_TOTAL_LIMIT" \
    "$@"

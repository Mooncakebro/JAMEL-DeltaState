# Trouble-shooting when setting up the environment
1. after:
```bash
uv pip install -r third_party/verl-agent/requirements.txt
```
get:
```bash
...
ModuleNotFoundError: No module named 'torch'<br/><br/>hint: This error likely indicates that `flash-attn@2.8.3` depends on `torch`, but doesn't declare it as a build dependency. If `flash-attn` is a first-party package, consider adding `torch` to its `build-system.requires`. Otherwise, either add it to your `pyproject.toml` under:<br/><br/>[tool.uv.extra-build-dependencies]<br/>flash-attn = ["torch"]<br/><br/>or `uv pip install torch` into the environment and re-run with `--no-build-isolation`.
```
do:
```bash
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

uv pip install ninja packaging setuptools wheel

mkdir -p ./tmp
export TMPDIR=./tmp
export TEMP=./tmp
export TMP=./tmp
export MAX_JOBS=4
uv pip install --no-build-isolation -r third_party/verl-agent/requirements.txt


```

2. get:
```bash
manylinux2014_x86_64.manylinux_2_17_x86_64.whl
  ├─▶ I/O operation failed during extraction
  ╰─▶ Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT (current value: 30s).

hint: `nvidia-nvshmem-cu12` (v3.4.5) was included because `jamel` (v0.1.0) depends on `torch` (v2.11.0+cu128) which depends on `nvidia-nvshmem-cu12`
```
try: 
```bash
UV_HTTP_TIMEOUT=300 uv sync --locked --python 3.10 --extra dev --extra train
```



# Prepare Data for JAMEL-DeltaState (need RAM~400G + A800 GPU)
python jamel/train/memory/prepare_sft_dataset.py \
  --input data/ExplorerSFT-ReAct_Dataset/data/react-text/*/trajectory.parquet \
          data/ExplorerSFT-ReAct_Dataset/data/react-vision/*/trajectory.parquet \
  --output ./data/explorer_sft_online_delta_state \
  --compressor-model ./LLMs/Qwen3-VL-2B-Instruct \
  --memory-hidden-size 2048 \
  --memory-builder online_delta_state \
  --max-memory-items 512 \
  --max-length 8192 \
  --val-ratio 0.02 \
  --compression-batch-size 4 \
  --delta-rank 8 \
  --delta-memory-slots 8 \
  --delta-seed 0


# Train (full STF)
TRAIN_FILE=./data/explorer_sft_online_delta_state/jamel_memory_sft_train.parquet \
VAL_FILE=./data/explorer_sft_online_delta_state/jamel_memory_sft_val.parquet \
BASE_MODEL_PATH=./LLMs/Qwen2.5-VL-7B-Instruct \
COMPRESSOR_MODEL=./LLMs/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=./outputs/explorer_delta_sft_ckpt \
OUTPUT_MODEL_PATH=./outputs/explorer_delta_model \
MEMORY_BUILDER=online_delta_state \
ONLINE_DELTA_STATE=1 \
READ_WITH_CURRENT_QUERY=1 \
MEMORY_MAX_ITEMS=512 \
MEMORY_HIDDEN_SIZE=2048 \
DELTA_RANK=8 \
DELTA_MEMORY_SLOTS=8 \
DELTA_SEED=0 \
bash shell/run_qwen25vl_7b_sft.sh


# Train (LoRA, on 1 A800)
TRAIN_FILE=./data/explorer_sft_online_delta_state/jamel_memory_sft_train.parquet \
VAL_FILE=./data/explorer_sft_online_delta_state/jamel_memory_sft_val.parquet \
BASE_MODEL_PATH=./LLMs/Qwen2.5-VL-7B-Instruct \
COMPRESSOR_MODEL=./LLMs/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=./outputs/explorer_delta_sft_ckpt \
OUTPUT_MODEL_PATH=./outputs/explorer_delta_model \
MEMORY_BUILDER=online_delta_state \
ONLINE_DELTA_STATE=1 \
TOTAL_EPOCHS=2 \
NPROC_PER_NODE=1 \
MAX_LENGTH=8192 \
TRAIN_BATCH_SIZE=4 \
MICRO_BATCH_SIZE_PER_GPU=2 \
bash shell/run_qwen25vl_7b_sft.sh \
    model.lora_rank=64 \
    model.lora_alpha=128 \
    "model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
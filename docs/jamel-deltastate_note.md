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

3. Install Node.js and Chinese font
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18
node --version    # 应输出 v18.x.x
npm --version

npm install -g monocart-coverage-reports monocart-locator

pip install fonttools

# verify
node --version                    # v18+
npm list -g monocart-coverage-reports 2>/dev/null  # 应该显示版本
fc-list :lang=zh | head -3         # 中文字体 (非关键)

```
4. Get:
```
BrowserType.launch: ENOENT: no such file or directory, mkdtemp 'tmp/playwright-artifacts-XXXXXXPWMKVE'
```
do
```bash
mkdir -p tmp
```


# Dataset download  (need data owner's permission)
```bash
pip install modelscope -U

modelscope login

modelscope download --dataset BlitherBoom812/ExplorerSFT-ReAct --local_dir ./data/ExplorerSFT-ReAct_Dataset
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


# Train (LoRA, on 2 A800)
TRAIN_FILE=./data/explorer_sft_online_delta_state/jamel_memory_sft_train.parquet \
VAL_FILE=./data/explorer_sft_online_delta_state/jamel_memory_sft_val.parquet \
BASE_MODEL_PATH=./LLMs/Qwen2.5-VL-7B-Instruct \
COMPRESSOR_MODEL=./LLMs/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=./outputs/explorer_delta_sft_ckpt \
OUTPUT_MODEL_PATH=./outputs/explorer_delta_model \
MEMORY_BUILDER=online_delta_state \
ONLINE_DELTA_STATE=1 \
TOTAL_EPOCHS=2 \
NPROC_PER_NODE=2 \
MAX_LENGTH=8192 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash shell/run_qwen25vl_7b_sft.sh \
    model.lora_rank=64 \
    model.lora_alpha=128 \
    "model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"


# Merge the base model and LoRA blocks (If use LoRA to PEFT)
```bash
# ① 重新合并
CKPT=$(ls -d ./outputs/explorer_delta_sft_ckpt/global_step_* | sort -t_ -k3 -n | tail -1)

python scripts/merge_lora_checkpoint.py \
    --checkpoint "$CKPT" \
    --output "${CKPT}_merged" \
    --lora-alpha 128 \
    --lora-rank 64

python -m jamel.train.memory.package_model \
    --checkpoint "${CKPT}_merged" \
    --compressor-model ./LLMs/Qwen3-VL-2B-Instruct \
    --output-model-path ./outputs/explorer_delta_model_merged

```


# Eval (on test10)
```bash
MODEL_PATH=./outputs/explorer_delta_model_merged \
APPS_MODE=test10 \
MAX_STEPS=50 \
NUM_SESSIONS=1 \
NUM_GPUS=2 \
WORKERS_PER_GPU=1 \
MEMORY_BUILDER=online_delta_state \
EVAL_OUTPUT=./outputs/eval_test10 \
bash shell/run_eval.sh
```

# Eval (on 1 app, e.g. weibo)
```bash
MODEL_PATH=./outputs/explorer_delta_model_merged \
APPS=weibo \
NUM_GPUS=1 \
WORKERS_PER_GPU=1 \
MAX_STEPS=20 \
MEMORY_BUILDER=online_delta_state \
EVAL_OUTPUT=./outputs/eval_weibo \
bash shell/run_eval.sh
```

# Preparation/Train/Eval under hybrid memory 
```bash
python jamel/train/memory/prepare_sft_dataset.py \
  --input /home/spc/.cache/modelscope/hub/datasets/BlitherBoom812/ExplorerSFT-ReAct/data/react-text/*/trajectory.parquet \
          /home/spc/.cache/modelscope/hub/datasets/BlitherBoom812/ExplorerSFT-ReAct/data/react-vision/*/trajectory.parquet \
  --output ./data/explorer_sft_hybrid \
  --compressor-model /home/spc/LLMs/Qwen3-VL-2B-Instruct \
  --memory-hidden-size 2048 \
  --memory-builder hybrid \
  --max-memory-items 512 \
  --hybrid-recent-items 50 \
  --max-length 8192 \
  --val-ratio 0.02 \
  --compression-batch-size 4 \
  --delta-rank 8 \
  --delta-memory-slots 8 \
  --delta-seed 13
```

```bash
TRAIN_FILE=./data/explorer_sft_hybrid/jamel_memory_sft_train.parquet \
VAL_FILE=./data/explorer_sft_hybrid/jamel_memory_sft_val.parquet \
BASE_MODEL_PATH=/home/spc/LLMs/Qwen2.5-VL-7B-Instruct \
COMPRESSOR_MODEL=/home/spc/LLMs/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=./outputs/explorer_hybrid8_sft_ckpt \
OUTPUT_MODEL_PATH=./outputs/explorer_hybrid8_model \
MEMORY_BUILDER=hybrid \
HYBRID_RECENT_ITEMS=50 \
MAX_HYBRID_RECENT_ITEMS=8 \
TOTAL_EPOCHS=8 \
NPROC_PER_NODE=8 \
MAX_LENGTH=8192 \
TRAIN_BATCH_SIZE=16 \
MICRO_BATCH_SIZE_PER_GPU=1 \
bash shell/run_qwen25vl_7b_sft.sh
```

```bash
MODEL_PATH=./outputs/explorer_hybrid8_model \
APPS_MODE=test10 \
MAX_STEPS=50 \
NUM_SESSIONS=1 \
NUM_GPUS=1 \
WORKERS_PER_GPU=1 \
MEMORY_BUILDER=hybrid \
HYBRID_RECENT_ITEMS=50 \
MAX_HYBRID_RECENT_ITEMS=8 \
EVAL_OUTPUT=./outputs/eval_hybrid8_test10 \
bash shell/run_eval.sh
```
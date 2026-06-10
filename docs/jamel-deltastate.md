# Prepare Data for JAMEL-DeltaState
python jamel/train/memory/prepare_sft_dataset.py \
  --input data/ExplorerSFT-ReAct/data/react-text/*/trajectory.parquet \
          data/ExplorerSFT-ReAct/data/react-vision/*/trajectory.parquet \
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


# Train
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
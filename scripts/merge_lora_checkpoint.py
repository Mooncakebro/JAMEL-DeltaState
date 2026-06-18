"""
Merge a LoRA-trained JAMEL checkpoint into a clean, eval-ready checkpoint.

Usage:
    python scripts/merge_lora_checkpoint.py \
        --checkpoint ./outputs/explorer_delta_sft_ckpt/global_step_11728 \
        --output ./outputs/explorer_delta_sft_ckpt/global_step_11728_merged

    python -m jamel.train.memory.package_model \
        --checkpoint ..._merged --compressor-model ./LLMs/Qwen3-VL-2B-Instruct \
        --output-model-path ./outputs/explorer_delta_model_merged

    MODEL_PATH=./outputs/explorer_delta_model_merged APPS=weibo NUM_GPUS=1 \
        WORKERS_PER_GPU=1 MAX_STEPS=20 MEMORY_BUILDER=online_delta_state \
        EVAL_OUTPUT=./outputs/eval_weibo bash shell/run_eval.sh
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


_PEFT_PREFIX = "base_model.model."
_LORA_A_MARKER = ".lora_A.default.weight"
_LORA_B_MARKER = ".lora_B.default.weight"
_BASE_LAYER_RE = re.compile(r"\.base_layer\.(weight|bias)$")

_COPY_FILES = [
    "config.json", "memory_augment_config.json",
    "tokenizer.json", "tokenizer_config.json",
    "preprocessor_config.json", "generation_config.json",
    "special_tokens_map.json", "chat_template.json",
    "vocab.json", "merges.txt",
]


def _strip_peft_prefix(key: str) -> str:
    """llm.base_model.model.X.Y -> llm.X.Y"""
    return key.replace(_PEFT_PREFIX, "")


def merge_lora_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    lora_alpha: float = 128.0,
    lora_rank: int = 64,
) -> None:
    checkpoint = Path(checkpoint)
    output = Path(output)

    # ---- 1. Load state dict ----
    safe_path = checkpoint / "model.safetensors"
    if safe_path.is_file():
        raw_sd = load_file(str(safe_path))
    else:
        index_path = checkpoint / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"No safetensors found in {checkpoint}")
        idx = json.loads(index_path.read_text())
        raw_sd = {}
        for shard in sorted(set(idx.get("weight_map", {}).values())):
            raw_sd.update(load_file(str(checkpoint / shard)))

    # ---- 2. Classify keys ----
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    other: dict[str, torch.Tensor] = {}

    for key, value in raw_sd.items():
        if _LORA_A_MARKER in key:
            lora_a[key.replace(_LORA_A_MARKER, "")] = value.float()
        elif _LORA_B_MARKER in key:
            lora_b[key.replace(_LORA_B_MARKER, "")] = value.float()
        else:
            other[key] = value

    if not lora_a:
        print("No LoRA keys — full fine-tune checkpoint. Copying as-is.")
        shutil.copytree(checkpoint, output, dirs_exist_ok=True)
        return

    print(f"Found {len(lora_a)} LoRA modules.  alpha={lora_alpha}, r={lora_rank}")
    scale = lora_alpha / lora_rank
    print(f"  alpha/r = {scale:.2f}")

    # ---- 3. Build clean state dict ----
    merged: dict[str, torch.Tensor] = {}
    merged_count = 0
    lora_done: set[str] = set()

    # 3a. Merge weights: clean = base_layer.weight + scale * (B @ A)
    for prefix in sorted(lora_a):
        if prefix not in lora_b:
            print(f"  SKIP: no lora_B for ...{prefix[-60:]}")
            continue

        base_w_key = prefix + ".base_layer.weight"
        if base_w_key not in raw_sd:
            print(f"  SKIP: no base_layer for ...{prefix[-60:]}")
            continue

        A = lora_a[prefix]
        B = lora_b[prefix]
        base_w = raw_sd[base_w_key].float()

        delta = scale * (B @ A)
        delta = delta.to(base_w.device, dtype=base_w.dtype)
        merged_w = base_w + delta

        clean_key = _strip_peft_prefix(prefix) + ".weight"
        merged[clean_key] = merged_w
        merged_count += 1
        lora_done.add(prefix)

    print(f"  Merged weights: {merged_count}/{len(lora_a)}")

    # 3b. Rename all non-LoRA keys:  llm.base_model.model.X.Y -> llm.X.Y
    for key, value in other.items():
        if ".base_layer." in key:
            m = _BASE_LAYER_RE.search(key)
            if not m:
                continue
            prefix = key[:m.start()]
            if prefix in lora_done:
                continue  # was merged above
            suffix = m.group(1)
            clean_key = _strip_peft_prefix(prefix) + "." + suffix
        else:
            clean_key = _strip_peft_prefix(key)

        if clean_key not in merged:
            merged[clean_key] = value

    # 3c. Bias keys for LoRA-merged layers (Qwen2.5-VL q/k/v proj have bias)
    for prefix in lora_done:
        base_b_key = prefix + ".base_layer.bias"
        if base_b_key in raw_sd:
            clean_b_key = _strip_peft_prefix(prefix) + ".bias"
            if clean_b_key not in merged:
                merged[clean_b_key] = raw_sd[base_b_key]

    # Safety net
    for key in list(merged.keys()):
        if ".base_layer." in key:
            del merged[key]

    print(f"  Total clean keys: {len(merged)}")

    # ---- 4. Save ----
    output.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output / "model.safetensors"))
    for fn in _COPY_FILES:
        src = checkpoint / fn
        if src.is_file():
            shutil.copy2(src, output / fn)

    print(f"\nMerged -> {output}")
    print(f"Files: {len(list(output.iterdir()))}")
    print(f"\nNext: package & eval.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lora-alpha", type=float, default=128.0)
    p.add_argument("--lora-rank", type=int, default=64)
    args = p.parse_args()
    merge_lora_checkpoint(args.checkpoint, args.output, args.lora_alpha, args.lora_rank)


if __name__ == "__main__":
    main()

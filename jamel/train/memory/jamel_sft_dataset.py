from __future__ import annotations

import io
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask

from jamel.core.env.web.axtree_utils import prune_axtree

logger = logging.getLogger(__name__)


def _decode_png(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _to_numeric_array(value, *, dtype):
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value.astype(dtype, copy=False)
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()
    return np.asarray(value, dtype=dtype)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _build_user_message(prompt: str) -> list[dict]:
    segments = prompt.split("<image>")
    content: list[dict] = []
    for idx, segment in enumerate(segments):
        if segment:
            content.append({"type": "text", "text": segment})
        if idx < len(segments) - 1:
            content.append({"type": "image"})
    return [{"role": "user", "content": content}]


def _build_full_messages(prompt: str, response: str) -> list[dict]:
    messages = _build_user_message(prompt)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return messages


# ── DiscardSentinel ───────────────────────────────────────────────────────
# Returned by _build_model_inputs when a pruned sample still overflows.
# The training loop should skip these.


class DiscardSentinel:
    pass


DISCARD = DiscardSentinel()


# ── Prompt prune wrapper ──────────────────────────────────────────────────
# Uses the SAME prune_axtree from axtree_utils.py as eval, ensuring
# train-inference consistency.  Extracts the AXTree block between
# "Current Observation:" and "Current valid interactive element ids:",
# prunes it, and reconstructs the prompt.


_OBS_MARKER = "Current Observation:"
_END_MARKER = "\nCurrent valid interactive element ids:"


def _prune_prompt(prompt: str, max_chars: int = 8000,
                  max_tokens: int | None = None) -> str:
    """Prune the AXTree block embedded in the training prompt.

    Keeps everything before and after the AXTree block intact (system prompt,
    action space, open pages metadata, valid ids, format instructions).
    Only the AXTree between "Current Observation:" and the end marker is
    compressed via prune_axtree() — the same function used by eval.

    If max_tokens is provided, additionally truncates the pruned AXTree to
    fit within the remaining token budget (chars = tokens * 2 approx).

    Handles escaped AXTree text (\\\\n, \\\\t) that may be present in
    older parquet files.
    """
    obs_start = prompt.find(_OBS_MARKER)
    if obs_start < 0:
        return prompt  # No observation block, nothing to prune

    # Start of AXTree text (after "Current Observation:" + optional newline)
    axtree_start = obs_start + len(_OBS_MARKER)
    if prompt[axtree_start:axtree_start + 1] == "\n":
        axtree_start += 1

    obs_end = prompt.find(_END_MARKER, axtree_start)
    if obs_end < 0:
        return prompt  # No end marker found, can't safely prune

    axtree_text = prompt[axtree_start:obs_end]
    prefix = prompt[:axtree_start]
    suffix = prompt[obs_end:]

    # Unescape \\n → \\n and \\t → \\t (older parquet may store AXTree escaped)
    axtree_text = axtree_text.replace("\\n", "\n").replace("\\t", "\t")

    pruned_axtree = prune_axtree(axtree_text.rstrip("\n"), max_chars=max_chars)

    # Safety truncation: if the pruned AXTree + prefix + suffix may exceed
    # the token budget, truncate the AXTree portion further.
    if max_tokens is not None:
        # Estimate non-AXTree tokens: prefix + suffix chars / 3 (English)
        non_ax_chars = len(prefix) + len(suffix)
        non_ax_tokens_est = non_ax_chars // 3
        vision_est = 1175 if "<image>" in prompt else 0
        chat_tmpl_est = 200
        resp_est = 80  # conservative response estimate
        ax_token_budget = max(0, max_tokens - non_ax_tokens_est - vision_est - chat_tmpl_est - resp_est)
        ax_char_budget = ax_token_budget * 2  # ~2 chars/token for AXTree

        if len(pruned_axtree) > ax_char_budget:
            pruned_axtree = pruned_axtree[:ax_char_budget]

    return prefix + pruned_axtree + suffix


# ── Dataset ───────────────────────────────────────────────────────────────


class JAMELMemoryVLTokenSFTDataset(Dataset):
    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(
            getattr(self.tokenizer, "name_or_path", None) or config.get("processor_path"),
            trust_remote_code=True,
        )
        self.max_length = config.get("max_length", 8192)
        self.prompt_key = config.get("prompt_key", "prompt")
        self.response_key = config.get("response_key", "response")
        self.image_key = config.get("image_key", "current_image_png_bytes")
        self.memory_tokens_key = config.get("memory_tokens_key", "memory_tokens")
        self.memory_attention_mask_key = config.get("memory_attention_mask_key", "memory_attention_mask")
        self.history_memory_tokens_key = config.get("history_memory_tokens_key", "history_memory_tokens")
        self.history_memory_attention_mask_key = config.get(
            "history_memory_attention_mask_key",
            "history_memory_attention_mask",
        )
        self.current_memory_query_tokens_key = config.get(
            "current_memory_query_tokens_key",
            "current_memory_query_tokens",
        )
        self.current_memory_query_attention_mask_key = config.get(
            "current_memory_query_attention_mask_key",
            "current_memory_query_attention_mask",
        )
        self.memory_max_items = config.get("memory_max_items", None)
        self.memory_hidden_size = config.get("memory_hidden_size", None)
        self.use_online_delta_state = _as_bool(config.get("use_online_delta_state", False))
        self.hybrid_recent_items = config.get("hybrid_recent_items", None)
        self.use_shm = config.get("use_shm", False)

        self._download()
        self._read_files()
        self._validate_samples()
        self._sample_saved = False

    def _download(self) -> None:
        for idx, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[idx] = copy_to_local(
                parquet_file,
                verbose=True,
                use_shm=self.use_shm,
            )

    def _read_files(self) -> None:
        frames = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)

    def _validate_samples(self) -> None:
        """Pre-scan all samples with prune + tokenize to identify discards.

        Builds self._valid_indices mapping logical index → raw row index.
        Prints discard statistics.
        """
        total = len(self.dataframe)
        valid_indices: list[int] = []

        # Estimate fixed overhead: chat template + image tokens.
        # Qwen2.5-VL uses ~1175 vision tokens per 1280×720 image.
        # We use a conservative overhead estimate for the text-only check.
        CHAT_TEMPLATE_OVERHEAD = 200   # apply_chat_template tokens
        IMAGE_TOKEN_OVERHEAD = 1175    # Qwen2.5-VL vision tokens per image

        for i in range(total):
            row = self.dataframe.iloc[i]
            prompt = row[self.prompt_key]

            # 1. Apply prune (same prune_axtree as eval, max_chars=8000, safety trunc to max_length)
            pruned = _prune_prompt(prompt, max_tokens=self.max_length)
            # Quick char-based pre-check: if pruned is still huge, discard early
            if len(pruned) > self.max_length * 10:
                continue

            # 2. Estimate token count (text only, no image processing)
            has_image = "<image>" in pruned
            response = row[self.response_key]

            try:
                prompt_tokens = len(self.tokenizer.encode(pruned, add_special_tokens=False))
                response_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
            except Exception:
                logger.warning(f"JAMEL SFT: failed to tokenize sample {i}, discarding.")
                continue

            # 3. Estimate total including processor overhead
            image_overhead = IMAGE_TOKEN_OVERHEAD if has_image else 0
            estimated_total = prompt_tokens + response_tokens + CHAT_TEMPLATE_OVERHEAD + image_overhead

            if estimated_total <= self.max_length:
                valid_indices.append(i)

        self._valid_indices = valid_indices
        discarded = total - len(valid_indices)

        if discarded > 0:
            logger.warning(
                "JAMEL SFT: %d / %d samples (%.1f%%) DISCARDED because pruned prompt "
                "still exceeds max_length=%d. These samples will NOT be used in training.",
                discarded,
                total,
                discarded / total * 100,
                self.max_length,
            )
        logger.info(
            "JAMEL SFT: %d / %d samples (%.1f%%) retained for training (max_length=%d).",
            len(valid_indices),
            total,
            len(valid_indices) / total * 100,
            self.max_length,
        )

    def _save_training_sample(
        self,
        row_idx: int,
        raw_prompt: str,
        pruned_prompt: str,
        full_text: str,
        prompt_text: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        response: str,
        image: Image.Image | None,
    ) -> None:
        """Save the actual sequence fed to the model (post chat-template, post-tokenize).

        Only called for samples that survived `_validate_samples` AND
        `_build_model_inputs` (i.e. never for discarded samples).
        """
        import json
        import os

        sample_dir = os.environ.get(
            "SFT_TRAINING_SAMPLE_DIR",
            "outputs/jamel_sft_training_sample",
        )
        os.makedirs(sample_dir, exist_ok=True)
        # Clean stale files from previous runs to avoid confusion.
        for fn in os.listdir(sample_dir):
            fp = os.path.join(sample_dir, fn)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass

        row = self.dataframe.iloc[row_idx]

        # ── Prune comparison (raw parquet prompt → pruned prompt) ──────────
        # These are pre-chat-template strings. They show what _prune_prompt
        # did to the AXTree block before chat template is applied.
        with open(os.path.join(sample_dir, "prompt_before_prune.txt"), "w") as f:
            f.write(raw_prompt)
        with open(os.path.join(sample_dir, "prompt_after_prune.txt"), "w") as f:
            f.write(pruned_prompt)

        # AXTree lines dropped by prune_axtree (between OBS_MARKER and END_MARKER).
        obs_start = raw_prompt.find(_OBS_MARKER)
        if obs_start >= 0:
            ax_start = obs_start + len(_OBS_MARKER)
            if raw_prompt[ax_start:ax_start + 1] == "\n":
                ax_start += 1
            obs_end = raw_prompt.find(_END_MARKER, ax_start)
            raw_ax = (
                raw_prompt[ax_start:obs_end].replace("\\n", "\n").replace("\\t", "\t")
                if obs_end >= 0 else ""
            )
        else:
            raw_ax = ""
        pruned_lines = set(pruned_prompt.split("\n"))
        raw_lines = raw_ax.split("\n")
        removed = [ln for ln in raw_lines if ln not in pruned_lines]
        with open(os.path.join(sample_dir, "content_removed_by_prune.txt"), "w") as f:
            f.write(
                f"AXTree lines removed by prune_axtree: {len(removed)}/{len(raw_lines)}\n"
                f"raw prompt chars: {len(raw_prompt)}  →  pruned prompt chars: {len(pruned_prompt)}\n"
            )
            f.write("=" * 60 + "\n")
            for line in removed[:500]:
                f.write(line + "\n")
            if len(removed) > 500:
                f.write(f"\n... and {len(removed) - 500} more lines removed\n")

        # ── Actual sequence fed to the model (post chat-template) ──────────
        # 1. The full chat-templated text (prompt + assistant response, including
        #    all special tokens / role markers). This is the string actually
        #    handed to the processor before tokenization.
        with open(os.path.join(sample_dir, "full_text.txt"), "w") as f:
            f.write(full_text)

        # 2. The prompt portion (with the generation prompt suffix appended).
        with open(os.path.join(sample_dir, "prompt_text.txt"), "w") as f:
            f.write(prompt_text)

        # 3. Decoded token sequence (what the model literally sees), special
        #    tokens preserved. Padding tokens are stripped via attention_mask.
        valid_len = int(attention_mask.sum().item())
        valid_ids = input_ids[:valid_len].tolist()
        decoded = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
        with open(os.path.join(sample_dir, "decoded_input_ids.txt"), "w") as f:
            f.write(decoded)

        # 4. Per-token id + piece, including loss_mask flag (1 = contributes to
        #    loss, 0 = ignored). Lets you verify which tokens are supervised.
        loss_flags = loss_mask[:valid_len].tolist()
        with open(os.path.join(sample_dir, "input_ids.txt"), "w") as f:
            f.write("# idx\tloss\ttoken_id\tpiece\n")
            for i, (tid, lm) in enumerate(zip(valid_ids, loss_flags)):
                piece = self.tokenizer.decode([tid], skip_special_tokens=False)
                f.write(f"{i}\t{int(lm)}\t{tid}\t{piece!r}\n")

        # 5. Response region only (tokens with loss_mask == 1).
        response_token_ids = [tid for tid, lm in zip(valid_ids, loss_flags) if lm == 1]
        response_decoded = self.tokenizer.decode(
            response_token_ids, skip_special_tokens=False,
        )
        with open(os.path.join(sample_dir, "response_loss_region.txt"), "w") as f:
            f.write(response_decoded)

        # 6. Original response string (pre-template) for cross-check.
        with open(os.path.join(sample_dir, "response.txt"), "w") as f:
            f.write(response)

        if image is not None:
            image.save(os.path.join(sample_dir, "screenshot.png"))

        if self.history_memory_tokens_key in row and row.get(self.history_memory_tokens_key) is not None:
            mt = _to_numeric_array(row[self.history_memory_tokens_key], dtype=np.float32)
            np.save(os.path.join(sample_dir, "history_memory_tokens.npy"), mt)
            memory_shape = list(mt.shape) if hasattr(mt, "shape") else None
        else:
            mt = _to_numeric_array(row[self.memory_tokens_key], dtype=np.float32)
            np.save(os.path.join(sample_dir, "memory_tokens.npy"), mt)
            memory_shape = list(mt.shape) if hasattr(mt, "shape") else None

        meta = {
            "session_id": str(row.get("session_id", "")),
            "episode_idx": int(row.get("episode_idx", 0)),
            "step_idx": int(row.get("step_idx", 0)),
            "target_app": str(row.get("target_app", "")),
            "reward": float(row.get("reward", 0.0)),
            "max_length": self.max_length,
            "prompt_chars_before_prune": len(raw_prompt),
            "prompt_chars_after_prune": len(pruned_prompt),
            "axtree_lines_removed_by_prune": len(removed),
            "axtree_lines_total": len(raw_lines),
            "valid_seq_len": valid_len,
            "response_token_count": len(response_token_ids),
            "padding_count": int(input_ids.shape[0] - valid_len),
            "has_image": image is not None,
            "image_size": list(image.size) if image is not None else None,
            "memory_tokens_shape": memory_shape,
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            "JAMEL SFT: saved training sample (app=%s, step=%d, seq_len=%d) to %s",
            meta["target_app"], meta["step_idx"], valid_len, sample_dir,
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    def _pad_memory(self, memory_tokens: torch.Tensor, memory_attention_mask: torch.Tensor):
        if memory_tokens.numel() == 0:
            hidden_size = self.memory_hidden_size or 2048
            max_items = self.memory_max_items or 512
            padded_tokens = torch.zeros((max_items, hidden_size), dtype=torch.float32)
            padded_mask = torch.zeros((max_items,), dtype=torch.long)
            return padded_tokens, padded_mask

        if memory_tokens.ndim == 1:
            memory_tokens = memory_tokens.unsqueeze(0)
        if memory_attention_mask.ndim == 0:
            memory_attention_mask = memory_attention_mask.unsqueeze(0)

        max_items = self.memory_max_items or memory_tokens.shape[0]
        if self._has_online_delta_state(row=None) and self.hybrid_recent_items is not None:
            max_items = self.hybrid_recent_items
        hidden_size = memory_tokens.shape[-1]
        padded_tokens = torch.zeros((max_items, hidden_size), dtype=memory_tokens.dtype)
        padded_mask = torch.zeros((max_items,), dtype=torch.long)
        valid_count = min(max_items, memory_tokens.shape[0])
        padded_tokens[:valid_count] = memory_tokens[:valid_count]
        padded_mask[:valid_count] = memory_attention_mask[:valid_count].to(torch.long)
        return padded_tokens, padded_mask

    def _pad_history_memory(self, history_tokens: torch.Tensor, history_attention_mask: torch.Tensor):
        hidden_size = self.memory_hidden_size or 2048
        max_items = self.memory_max_items or (
            int(history_tokens.shape[0]) if history_tokens.ndim >= 2 else 0
        )
        max_items = max(1, int(max_items))

        if history_tokens.numel() == 0:
            return (
                torch.zeros((max_items, int(hidden_size)), dtype=torch.float32),
                torch.zeros((max_items,), dtype=torch.long),
            )
        if history_tokens.ndim == 1:
            history_tokens = history_tokens.unsqueeze(0)
        if history_attention_mask.ndim == 0:
            history_attention_mask = history_attention_mask.unsqueeze(0)
        hidden_size = history_tokens.shape[-1]
        padded_tokens = torch.zeros((max_items, hidden_size), dtype=history_tokens.dtype)
        padded_mask = torch.zeros((max_items,), dtype=torch.long)
        valid_count = min(max_items, history_tokens.shape[0])
        padded_tokens[:valid_count] = history_tokens[-valid_count:]
        padded_mask[:valid_count] = history_attention_mask[-valid_count:].to(torch.long)
        return padded_tokens, padded_mask

    def _pad_current_query_memory(
        self,
        query_tokens: torch.Tensor,
        query_attention_mask: torch.Tensor,
    ):
        hidden_size = self.memory_hidden_size or 2048
        if query_tokens.numel() == 0:
            return (
                torch.zeros((1, int(hidden_size)), dtype=torch.float32),
                torch.zeros((1,), dtype=torch.long),
            )
        if query_tokens.ndim == 1:
            query_tokens = query_tokens.unsqueeze(0)
        if query_attention_mask.ndim == 0:
            query_attention_mask = query_attention_mask.unsqueeze(0)
        return query_tokens.to(torch.float32), query_attention_mask.to(torch.long)

    def _has_online_delta_state(self, row) -> bool:
        if row is None:
            return self.use_online_delta_state
        return (
            self.use_online_delta_state
            or (
                self.history_memory_tokens_key in row
                and self.current_memory_query_tokens_key in row
            )
        )

    def _build_model_inputs(
        self, prompt: str, response: str, image: Image.Image | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict] | DiscardSentinel:
        """Tokenize prompt + response after pruning.

        If the pruned prompt alone exceeds max_length → return DISCARD.
        If the total (prompt + response) exceeds max_length → return DISCARD.
        Otherwise, return (input_ids, attention_mask, position_ids, loss_mask, multi_modal_inputs).
        """
        # 1. Prune the prompt using the same prune_axtree as eval (max_chars=8000)
        pruned_prompt = _prune_prompt(prompt, max_tokens=self.max_length)
        has_image = "<image>" in pruned_prompt

        prompt_messages = _build_user_message(pruned_prompt)
        full_messages = _build_full_messages(pruned_prompt, response)

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )

        # 2. Tokenize.  __getitem__ resizes `image` to WEB_MODEL_IMAGE_SIZE
        # before calling us, so the saved training sample and the model input
        # always agree on what the VLM actually sees.  Browser viewport stays
        # at its original resolution to preserve responsive layout.
        if has_image and image is not None:
            prompt_model_inputs = self.processor(
                text=[prompt_text], images=[image], return_tensors="pt",
            )
            full_model_inputs = self.processor(
                text=[full_text], images=[image], return_tensors="pt",
            )
        else:
            prompt_model_inputs = self.processor(
                text=[prompt_text], return_tensors="pt",
            )
            full_model_inputs = self.processor(
                text=[full_text], return_tensors="pt",
            )
        full_model_inputs.pop("second_per_grid_ts", None)

        full_input_ids = full_model_inputs["input_ids"][0]
        full_attention_mask = full_model_inputs["attention_mask"][0]
        prompt_length = prompt_model_inputs["input_ids"].shape[-1]

        # 3. DISCARD if prompt exceeds max_length (prune wasn't enough)
        if prompt_length > self.max_length:
            return DISCARD

        # 4. DISCARD if total exceeds max_length (response too long after prune)
        total_length = full_input_ids.shape[0]
        if total_length > self.max_length:
            return DISCARD

        # 5. Pad or use as-is
        input_ids = full_input_ids
        attention_mask = full_attention_mask
        response_length = total_length - prompt_length

        if total_length < self.max_length:
            pad_len = self.max_length - total_length
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype),
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros((pad_len,), dtype=attention_mask.dtype),
            ])

        position_ids = compute_position_id_with_mask(attention_mask)
        if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=full_input_ids[: full_attention_mask.sum().item()],
                image_grid_thw=full_model_inputs.get("image_grid_thw"),
                video_grid_thw=full_model_inputs.get("video_grid_thw"),
                second_per_grid_ts=None,
                attention_mask=full_attention_mask[: full_input_ids.shape[-1]],
            )
            valid_mask = attention_mask.bool()
            text_position_ids = torch.ones((1, len(attention_mask)), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            vision_position_ids = vision_position_ids.to(dtype=torch.long)
            if vision_position_ids.shape[-1] != text_position_ids.shape[-1]:
                padded = torch.zeros((vision_position_ids.shape[0], text_position_ids.shape[-1]), dtype=torch.long)
                valid_len = min(vision_position_ids.shape[-1], padded.shape[-1])
                padded[:, :valid_len] = vision_position_ids[:, :valid_len]
                vision_position_ids = padded
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: prompt_length - 1] = 0
        if response_length > 0:
            loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        multi_modal_inputs = {}
        for key, value in full_model_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            if isinstance(value, torch.Tensor):
                if key == "image_grid_thw":
                    multi_modal_inputs[key] = value
                else:
                    multi_modal_inputs[key] = value[0] if value.ndim > 0 and value.shape[0] == 1 else value

        return input_ids, attention_mask, position_ids, loss_mask, multi_modal_inputs

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_idx = self._valid_indices[index]
        row = self.dataframe.iloc[row_idx]
        prompt = row[self.prompt_key]
        response = row[self.response_key]

        has_image = "<image>" in prompt
        image: Image.Image | None = None
        if has_image:
            raw = row.get(self.image_key)
            if raw is not None:
                image = _decode_png(raw)
                # Resize on the model side here so that _build_model_inputs and
                # the saved training sample see the same image fed to the VLM.
                from jamel.train.memory.web_prompt import WEB_MODEL_IMAGE_SIZE
                if image.size != WEB_MODEL_IMAGE_SIZE:
                    image = image.resize(WEB_MODEL_IMAGE_SIZE, Image.BILINEAR)

        # Save the first sample for inspection (once per training run).
        # Done AFTER _build_model_inputs so we never persist a discarded sample.
        result = self._build_model_inputs(prompt=prompt, response=response, image=image)

        if result is DISCARD:
            raise RuntimeError(
                f"Sample at row {row_idx} was not filtered during _validate_samples but "
                f"_build_model_inputs returned DISCARD. This indicates an estimator mismatch. "
                f"Please check prune_prompt and tokenizer consistency."
            )

        input_ids, attention_mask, position_ids, loss_mask, multi_modal_inputs = result

        if not self._sample_saved:
            self._sample_saved = True
            try:
                pruned_prompt = _prune_prompt(prompt, max_tokens=self.max_length)
                prompt_messages = _build_user_message(pruned_prompt)
                full_messages = _build_full_messages(pruned_prompt, response)
                prompt_text = self.processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                )
                full_text = self.processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False,
                )
                self._save_training_sample(
                    row_idx=row_idx,
                    raw_prompt=prompt,
                    pruned_prompt=pruned_prompt,
                    full_text=full_text,
                    prompt_text=prompt_text,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                    response=response,
                    image=image,
                )
            except Exception:
                logger.warning("JAMEL SFT: failed to save training sample.", exc_info=True)

        if self._has_online_delta_state(row):
            history_memory_tokens = torch.tensor(
                _to_numeric_array(row[self.history_memory_tokens_key], dtype=np.float32),
                dtype=torch.float32,
            )
            if (
                self.history_memory_attention_mask_key in row
                and row[self.history_memory_attention_mask_key] is not None
            ):
                history_memory_attention_mask = torch.tensor(
                    _to_numeric_array(row[self.history_memory_attention_mask_key], dtype=np.int64),
                    dtype=torch.long,
                )
            else:
                history_memory_attention_mask = torch.ones(
                    history_memory_tokens.shape[0],
                    dtype=torch.long,
                )
            history_memory_tokens, history_memory_attention_mask = self._pad_history_memory(
                history_memory_tokens,
                history_memory_attention_mask,
            )

            current_memory_query_tokens = torch.tensor(
                _to_numeric_array(row[self.current_memory_query_tokens_key], dtype=np.float32),
                dtype=torch.float32,
            )
            if (
                self.current_memory_query_attention_mask_key in row
                and row[self.current_memory_query_attention_mask_key] is not None
            ):
                current_memory_query_attention_mask = torch.tensor(
                    _to_numeric_array(row[self.current_memory_query_attention_mask_key], dtype=np.int64),
                    dtype=torch.long,
                )
            else:
                current_memory_query_attention_mask = torch.ones(
                    current_memory_query_tokens.shape[0],
                    dtype=torch.long,
                )
            current_memory_query_tokens, current_memory_query_attention_mask = (
                self._pad_current_query_memory(
                    current_memory_query_tokens,
                    current_memory_query_attention_mask,
                )
            )

            if self.memory_tokens_key in row and row.get(self.memory_tokens_key) is not None:
                memory_tokens = torch.tensor(
                    _to_numeric_array(row[self.memory_tokens_key], dtype=np.float32),
                    dtype=torch.float32,
                )
                if self.memory_attention_mask_key in row and row[self.memory_attention_mask_key] is not None:
                    memory_attention_mask = torch.tensor(
                        _to_numeric_array(row[self.memory_attention_mask_key], dtype=np.int64),
                        dtype=torch.long,
                    )
                else:
                    memory_attention_mask = torch.ones(memory_tokens.shape[0], dtype=torch.long)
                memory_tokens, memory_attention_mask = self._pad_memory(memory_tokens, memory_attention_mask)
            else:
                memory_tokens = None
                memory_attention_mask = None

            memory_payload = {
                "history_memory_tokens": history_memory_tokens,
                "history_memory_attention_mask": history_memory_attention_mask,
                "current_memory_query_tokens": current_memory_query_tokens,
                "current_memory_query_attention_mask": current_memory_query_attention_mask,
                # Pure online_delta_state has None here. Hybrid keeps recent
                # original JAMEL tokens here and the actor appends them after
                # the online DeltaState read.
                "memory_tokens": memory_tokens,
                "memory_attention_mask": memory_attention_mask,
            }
        else:
            memory_tokens = torch.tensor(
                _to_numeric_array(row[self.memory_tokens_key], dtype=np.float32), dtype=torch.float32,
            )
            if self.memory_attention_mask_key in row and row[self.memory_attention_mask_key] is not None:
                memory_attention_mask = torch.tensor(
                    _to_numeric_array(row[self.memory_attention_mask_key], dtype=np.int64),
                    dtype=torch.long,
                )
            else:
                memory_attention_mask = torch.ones(memory_tokens.shape[0], dtype=torch.long)
            memory_tokens, memory_attention_mask = self._pad_memory(memory_tokens, memory_attention_mask)
            memory_payload = {
                "memory_tokens": memory_tokens,
                "memory_attention_mask": memory_attention_mask,
            }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
            **memory_payload,
        }

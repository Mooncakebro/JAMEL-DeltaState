"""
Convert augmented browser-session parquet into memory-augmented SFT parquet.

Each output row uses the canonical web-agent prompt format defined in
``jamel.train.memory.web_prompt`` (see ``docs/TRAINING.md``). Prompts
are rebuilt from atomic columns; the upstream ``prompt`` column (which may
contain ReAct/JSON memory blocks) is intentionally ignored. ``<think>`` is
stripped from responses; long-term context flows through ``memory_tokens``,
not through prompt text.

Required input columns:
    session_id, episode_idx, step_idx, target_app, start_url,
    before_observation_str, before_open_pages_urls,
    response, action, reward, before_screenshot (or screenshot)

Memory semantics:
    For step (session_id, episode_idx, step_idx), memory = all steps in the
    same session with strictly smaller (episode_idx, step_idx), ordered by
    (episode_idx, step_idx).  This matches inference: memory is never cleared
    across episode resets within a session.

Compressor reuse:
    Each (session_id, episode_idx, step_idx) step is compressed exactly once.
    Processing steps in (episode_idx, step_idx) order within a session lets the
    OnlineHistoryMemoryBuilder's per-record _cached_memory_token cache reuse
    results from earlier steps without recomputation.

Usage:
    python prepare_sft_dataset.py \\
        --input  outputs/.../augmented_accepted_samples.parquet \\
        --output data/jamel_sft_data \\
        --compressor-model /path/to/Qwen3-VL-2B-Instruct \\
        --max-memory-items 512 \\
        --max-length 8192 \\
        --val-ratio 0.05
"""
from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def _decode_png(image_bytes: bytes | None) -> Image.Image | None:
    if image_bytes is None:
        return None
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def _get_screenshot(row) -> bytes | None:
    """Try before_screenshot first (old schema), then screenshot (new schema)."""
    v = row.get("before_screenshot")
    if v is not None:
        return v
    return row.get("screenshot")


def build_dataset(args: argparse.Namespace) -> None:
    from jamel.core.env.web.axtree_utils import prune_axtree
    from jamel.train.memory.delta_state_encoder import MODEL_NAME, make_history_memory_builder
    from jamel.train.memory.web_prompt import (
        build_web_prompt,
        extract_axtree_from_observation_str,
        strip_think,
    )

    # Multi-file input: --input can be one or more parquet paths. Each path is
    # loaded and concatenated. Used for full-scale ReAct training where every
    # app has its own trajectory.parquet under data/react/react-vision/<app>/.
    inputs = args.input if isinstance(args.input, list) else [args.input]
    if len(inputs) == 1:
        df = pd.read_parquet(inputs[0])
    else:
        frames = []
        for p in inputs:
            sub = pd.read_parquet(p)
            print(f"  loaded {len(sub):>5} rows from {p}")
            frames.append(sub)
        df = pd.concat(frames, ignore_index=True)
        print(f"Concatenated {len(inputs)} parquet files → {len(df)} rows total")

    # ── Validate schema ──────────────────────────────────────────────────────
    required_cols = {
        "session_id", "episode_idx", "step_idx",
        "target_app", "start_url",
        "before_observation_str", "before_open_pages_urls",
        "response", "action",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input parquet is missing required columns: {missing}\n"
            "Expected augmented session-schema parquet with atomic observation "
            "fields (before_observation_str, before_open_pages_urls)."
        )

    # ── Sort by canonical order ──────────────────────────────────────────────
    df = df.sort_values(["target_app", "session_id", "episode_idx", "step_idx"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows | sessions={df['session_id'].nunique()} | "
          f"apps={df['target_app'].nunique()} | positive={(df['reward'] > 0).sum()}")

    # ── Build specs in session order (critical for cache reuse) ──────────────
    # We do NOT shuffle before compression.  Shuffling happens after all memory
    # tokens are computed, so the _cached_memory_token entries written by earlier
    # steps in a session are available when later steps are processed.

    all_specs: list[dict[str, Any]] = []

    for session_id, session_df in df.groupby("session_id", sort=False):
        session_df = session_df.sort_values(["episode_idx", "step_idx"]).reset_index(drop=True)

        # Build shared trajectory_records for this session.
        # These Python dicts are referenced by multiple history_slice lists;
        # _cached_memory_token is written back into them by the builder.
        trajectory_records: list[dict[str, Any]] = []
        for _, row in session_df.iterrows():
            img_bytes = _get_screenshot(row)
            img = _decode_png(img_bytes)
            action = str(row.get("action") or "")
            trajectory_records.append({"image_obs": img, "action": action})

        for step_pos, (_, row) in enumerate(session_df.iterrows()):
            img_bytes = _get_screenshot(row)
            if img_bytes is None:
                continue

            action = str(row.get("action") or "")
            if not action:
                continue

            # History = ALL steps in this session before the current one,
            # ordered by (episode_idx, step_idx).  No sliding-window truncation.
            # The builder will internally cap at max_memory_items if set.
            history_slice = trajectory_records[:step_pos]

            obs_str = str(row.get("before_observation_str") or "")
            axtree_raw = extract_axtree_from_observation_str(obs_str)
            pruned_axtree = prune_axtree(axtree_raw, max_chars=8000)

            prompt_text = build_web_prompt(
                step_idx=int(row.get("step_idx", 0)),
                target_app=str(row.get("target_app", "")),
                start_url=str(row.get("start_url", "")),
                open_urls=row.get("before_open_pages_urls"),
                pruned_axtree=pruned_axtree,
            )
            response_text = strip_think(str(row.get("response", "")))

            spec: dict[str, Any] = {
                "session_id": session_id,
                "episode_idx": int(row.get("episode_idx", 0)),
                "step_idx": int(row.get("step_idx", 0)),
                "target_app": str(row.get("target_app", "")),
                "start_url": str(row.get("start_url", "")),
                "prompt": prompt_text,
                "response": response_text,
                "reward": float(row.get("reward", 0.0)),
                "current_image_png_bytes": img_bytes,
                "history_records": history_slice,  # list of shared record dicts
            }
            all_specs.append(spec)

    print(f"Total specs: {len(all_specs)}")
    if not all_specs:
        raise RuntimeError("No valid specs found.")

    # ── Compress memory tokens (in session order → max cache reuse) ──────────
    builder = make_history_memory_builder(
        memory_builder=args.memory_builder,
        compressor_model_name=args.compressor_model,
        memory_hidden_size=args.memory_hidden_size,
        # history_window: upper bound on records passed; use max_memory_items
        # as the effective window so the builder truncates to max_memory_items
        # most-recent history steps if session history exceeds that limit.
        history_window=args.max_memory_items,
        max_memory_items=args.max_memory_items,
        torch_dtype="bfloat16",
        device_map="auto",
        cache_history_memory=True,
        delta_rank=args.delta_rank,
        delta_memory_slots=args.delta_memory_slots,
        delta_seed=args.delta_seed,
        hybrid_recent_items=args.hybrid_recent_items,
    )
    compressor = builder.compressor
    tokenizer = getattr(getattr(compressor, "processor", None), "tokenizer", None)
    if tokenizer is not None:
        tokenizer.add_eos_token = True
    print(
        f"Memory builder: {args.memory_builder} ({MODEL_NAME if args.memory_builder == 'delta_state' else 'JAMEL'})  "
        f"hidden_size: {builder.memory_hidden_size}  max_memory_items: {args.max_memory_items}  "
        f"delta_rank: {args.delta_rank}  delta_slots: {args.delta_memory_slots}"
    )

    finalized_rows: list[dict[str, Any]] = []
    batch_size = args.compression_batch_size
    for start in range(0, len(all_specs), batch_size):
        batch = all_specs[start: start + batch_size]
        memory_tokens, memory_mask = builder.build_memory_inputs(
            batch_size=len(batch),
            history_records=[s["history_records"] for s in batch],
        )
        for i, spec in enumerate(batch):
            row = {k: v for k, v in spec.items() if k != "history_records"}
            row["memory_tokens"] = memory_tokens[i].tolist()
            row["memory_attention_mask"] = memory_mask[i].tolist()
            finalized_rows.append(row)

        if (start // batch_size + 1) % 10 == 0:
            print(f"  Compressed {start + len(batch)}/{len(all_specs)} samples")

    # ── Shuffle and split ────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(finalized_rows)

    dataset_df = pd.DataFrame(finalized_rows)
    if args.val_ratio <= 0.0:
        # Use ALL rows for training. val parquet is a 1-row dummy so verl's
        # required val_files path stays valid; train scripts should set
        # val_steps high enough that val never runs intra-epoch.
        train_df = dataset_df.reset_index(drop=True)
        val_df = train_df.iloc[:1].copy()
    else:
        split_idx = max(1, int(len(dataset_df) * (1.0 - args.val_ratio)))
        train_df = dataset_df.iloc[:split_idx].reset_index(drop=True)
        val_df = dataset_df.iloc[split_idx:].reset_index(drop=True)
        if len(val_df) == 0:
            val_df = train_df.iloc[:1].copy()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "jamel_memory_sft_train.parquet"
    val_path = output_dir / "jamel_memory_sft_val.parquet"
    train_df.to_parquet(train_path, row_group_size=4000)
    val_df.to_parquet(val_path, row_group_size=4000)

    print(f"\nDataset written:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Val:   {val_path}  ({len(val_df)} rows)")
    print(f"  Memory hidden size: {builder.memory_hidden_size}")
    print(f"  Max memory items:   {args.max_memory_items}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, nargs="+",
                   help="One or more augmented session-schema SFT parquet "
                        "files (session_id / episode_idx / step_idx). "
                        "Multiple paths are concatenated — used for full-scale "
                        "ReAct runs.")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--compressor-model", required=True, help="Local Qwen3-VL-2B compressor model directory")
    p.add_argument("--memory-hidden-size", default="auto")
    p.add_argument(
        "--memory-builder",
        choices=["online_tokens", "delta_state", "hybrid"],
        default="online_tokens",
        help="History compressor used to produce memory_tokens. delta_state is JAMEL-DeltaState.",
    )
    p.add_argument("--delta-rank", type=int, default=8, help="JAMEL-DeltaState associative rank.")
    p.add_argument("--delta-memory-slots", type=int, default=8, help="Number of state-derived memory tokens.")
    p.add_argument("--delta-seed", type=int, default=13, help="Deterministic projection seed for JAMEL-DeltaState.")
    p.add_argument(
        "--hybrid-recent-items",
        type=int,
        default=32,
        help="For memory_builder=hybrid, append this many recent original JAMEL tokens after DeltaState tokens.",
    )
    p.add_argument(
        "--max-memory-items", type=int, default=512,
        help="Maximum number of history steps to keep in memory.  "
             "History is all previous steps in the session (no sliding window); "
             "if a session exceeds this limit, the oldest steps are dropped.  "
             "Set large enough to cover the longest session in your dataset.",
    )
    p.add_argument(
        "--max-length", type=int, default=8192,
        help="Target context length (tokens) for the main model.  "
             "Must match the MAX_LENGTH used during training.  "
             "This is passed through to dataset metadata; actual prompt prune "
             "happens at training time via prune_prompt().  "
             "Default: 8192.",
    )
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--compression-batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())

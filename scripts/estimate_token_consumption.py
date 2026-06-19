"""
Estimate input token consumption from eval trajectory parquets.

Usage (after running eval):
    python scripts/estimate_token_consumption.py \
        --eval-dir ./outputs/eval_test10 \
        --tokenizer-path ./LLMs/Qwen2.5-VL-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def estimate_tokens(
    eval_dir: str | Path,
    tokenizer_path: str,
    max_steps_per_session: int = 50,
) -> None:
    eval_dir = Path(eval_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    app_stats: dict[str, dict] = {}

    for app_dir in sorted(eval_dir.iterdir()):
        if not app_dir.is_dir():
            continue
        app_name = app_dir.name

        session_prompt_tokens: list[int] = []
        session_response_tokens: list[int] = []
        session_total_tokens: list[int] = []
        all_steps: list[dict] = []

        # Find all trajectory parquets under this app dir.
        # Two possible layouts:
        #   app_dir/trajectory_*.parquet            (NUM_SESSIONS=1, flat)
        #   app_dir/session_XX/trajectory_*.parquet (NUM_SESSIONS>1, nested)
        traj_files = sorted(app_dir.glob("trajectory_*.parquet"))
        if not traj_files:
            for session_dir in sorted(app_dir.iterdir()):
                if session_dir.is_dir():
                    traj_files.extend(sorted(session_dir.glob("trajectory_*.parquet")))

        if not traj_files:
            continue

        for traj_path in traj_files:
            df = pd.read_parquet(traj_path)
            prompt_tokens = 0
            response_tokens = 0

            for _, row in df.iterrows():
                prompt = str(row.get("prompt", ""))
                response = str(row.get("response", ""))
                action = str(row.get("action", ""))

                p_tok = len(tokenizer.encode(prompt, add_special_tokens=False))
                # Response = raw model output (includes <action> tags etc.)
                r_tok = len(tokenizer.encode(response, add_special_tokens=False))

                prompt_tokens += p_tok
                response_tokens += r_tok

                all_steps.append({
                    "step": int(row.get("step_idx", 0)),
                    "episode": int(row.get("episode_idx", 0)),
                    "action": action,
                    "prompt_tokens": p_tok,
                    "response_tokens": r_tok,
                    "reward": float(row.get("reward", 0.0)),
                })

            session_prompt_tokens.append(prompt_tokens)
            session_response_tokens.append(response_tokens)
            session_total_tokens.append(prompt_tokens + response_tokens)

        if not session_prompt_tokens:
            continue

        app_stats[app_name] = {
            "sessions": len(session_prompt_tokens),
            "steps": len(all_steps),
            "prompt_tokens_total": sum(session_prompt_tokens),
            "prompt_tokens_per_session": sum(session_prompt_tokens) / len(session_prompt_tokens),
            "response_tokens_total": sum(session_response_tokens),
            "prompt_tokens_per_step": sum(p["prompt_tokens"] for p in all_steps) / max(1, len(all_steps)),
            "total_tokens": sum(session_total_tokens),
            "steps_detail": all_steps,
        }

    # ── Print report ──
    print(f"\n{'='*80}")
    print(f"Input Token Consumption over {len(app_stats)} apps ({max_steps_per_session} steps/session)")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"{'='*80}\n")
    print(f"{'App':<18} {'Steps':>6} {'Prompt/Step':>12} {'Prompt/Session':>16} {'Total Prompt':>14} {'Total All':>12}")
    print("-" * 80)

    total_prompt = 0
    total_all = 0
    total_steps = 0

    for app_name in sorted(app_stats):
        s = app_stats[app_name]
        total_prompt += s["prompt_tokens_total"]
        total_all += s["total_tokens"]
        total_steps += s["steps"]
        print(
            f"{app_name:<18} {s['steps']:>6} "
            f"{s['prompt_tokens_per_step']:>10.0f}  "
            f"{s['prompt_tokens_per_session']:>14.0f}  "
            f"{s['prompt_tokens_total']:>14}  "
            f"{s['total_tokens']:>12}"
        )

    print("-" * 80)
    print(
        f"{'TOTAL':<18} {total_steps:>6} "
        f"{total_prompt / max(1, total_steps):>10.0f}  "
        f"{'':>14}  "
        f"{total_prompt:>14}  "
        f"{total_all:>12}"
    )

    # ── Save JSON ──
    summary = {
        "tokenizer": tokenizer_path,
        "max_steps_per_session": max_steps_per_session,
        "apps": len(app_stats),
        "total_prompt_tokens": total_prompt,
        "total_response_tokens": total_all - total_prompt,
        "total_tokens": total_all,
        "total_steps": total_steps,
        "avg_prompt_per_step": total_prompt / max(1, total_steps),
        "per_app": {
            name: {
                k: v for k, v in stats.items() if k != "steps_detail"
            }
            for name, stats in app_stats.items()
        },
    }
    out_path = eval_dir / "token_consumption.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True, help="Eval output directory")
    p.add_argument("--tokenizer-path", required=True, help="Path to tokenizer (base model dir)")
    p.add_argument("--max-steps", type=int, default=50)
    args = p.parse_args()
    estimate_tokens(args.eval_dir, args.tokenizer_path, args.max_steps)


if __name__ == "__main__":
    main()

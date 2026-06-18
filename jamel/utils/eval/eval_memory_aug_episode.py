"""
eval_memory_aug_episode.py
Run one BrowserGym session with the trained MemoryAugmentedCausalLM checkpoint.
Saves a trajectory parquet aligned with the training data schema, including V8
JS coverage collection, coverage reward computation, and full observation fields.

Terminology (matches docs/weak_model_sft_labeling_workflow.md):
  session  — the entire eval run for one app: one model load, one memory buffer,
             max_steps total steps.  Memory is NOT reset between episodes.
  episode  — one reset-bounded sub-run within a session.  Begins at env.reset()
             and ends when the agent emits reset() or max_steps is exhausted.
             episode_idx counts how many resets have occurred so far.

Usage:
    python eval_memory_aug_episode.py \\
        --checkpoint outputs/jamel_sft_ckpt/global_step_2361 \\
        --env-id weibo \\
        --max-steps 20 \\
        --output outputs/eval_trajectory

Environment:
    export TRANSFORMERS_OFFLINE=1
    export PYTHONPATH=<JAMEL>:<JAMEL>/third_party/verl-agent

Inference hyperparameters (fixed):
    - do_sample=False  (greedy decoding)
    - max_new_tokens=256
    - memory_max_items=512  (matches SFT v2 training)
    - memory_hidden_size=2048  (Qwen3-VL-2B compressor hidden size)
    - prompt format: text-only, no <image> tag (matches SFT v2 training data)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal as _signal
import sys
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────
# Default downloaded environment layout:
#   JAMEL/env/browser_env/scalewob-env
# Override via SCALEWOB_ROOT env var or --scalewob-root CLI arg.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RELEASE_SCALEWOB_ROOT = _REPO_ROOT / "env" / "browser_env" / "scalewob-env"
_DEFAULT_SCALEWOB_ROOT = Path(
    os.environ.get(
        "SCALEWOB_ROOT",
        str(_DEFAULT_RELEASE_SCALEWOB_ROOT),
    )
)

# ── JAMEL imports ───────────────────────────────────────────────────
from browsergym.core.env import BrowserEnv
from browsergym.core.task import AbstractBrowserTask
from browsergym.utils.obs import flatten_axtree_to_str

from jamel.core.env.web.axtree_utils import prune_observation_text
from jamel.core.env.web.observer import Observer
from jamel.core.env.web.coverage import start_coverage, save_coverage
from jamel.core.reward.web.utils import compute_monocart_coverage_reward_details
from jamel.coverage_artifact import build_coverage_artifact_fields


def _timeout_handler(signum, frame):
    raise TimeoutError("timed out after 120s")


def compute_coverage_details(
    current_path: Path | None,
    history_paths: list[Path],
    previous_score: int = 0,
) -> dict:
    """Compute reward through the shared Monocart cumulative coverage interface."""
    return compute_monocart_coverage_reward_details(
        current_path=current_path,
        baseline_paths=history_paths,
        previous_score=previous_score,
    )

# ── prompt builder ─────────────────────────────────────────────────────────
# Inference uses the same `build_web_prompt` as training (see web_prompt.py).
# Action space, prompt template, and AXTree pruning live in that single module
# so eval and SFT cannot drift apart.

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
THINK_RE  = re.compile(r"<think>(.*?)</think>",  re.DOTALL)


# ── ScaleWoB BrowserGym task (local static server) ─────────────────────────
class LocalStaticServer:
    def __init__(self, root_dir: Path, host: str = "127.0.0.1", port: int = 8790):
        self.root_dir = Path(root_dir)
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        handler = lambda *a, **kw: SimpleHTTPRequestHandler(
            *a, directory=str(self.root_dir), **kw
        )
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="scalewob-http", daemon=True
        )
        self._thread.start()
        print(f"[server] Serving {self.root_dir} at http://{self.host}:{self.port}/")

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None


class ScaleWoBTask(AbstractBrowserTask):
    @classmethod
    def get_task_id(cls):
        return "scalewob.eval-episode"

    def __init__(self, seed: int, env_id: str, port: int = 8790,
                 viewport_width: int = 1280, viewport_height: int = 720,
                 scalewob_root: str | Path | None = None) -> None:
        super().__init__(seed)
        self.env_id = env_id
        self.port = port
        self.server = LocalStaticServer(
            Path(scalewob_root) if scalewob_root is not None else _DEFAULT_SCALEWOB_ROOT,
            port=port,
        )
        self.viewport = {"width": viewport_width, "height": viewport_height}
        self.slow_mo = 0
        self.timeout = 15_000

    @property
    def start_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/{self.env_id}/index.html"

    def setup(self, page) -> tuple[str, dict]:
        self.server.start()

        def _handle_route(route, request):
            url = request.url
            if "fonts.googleapis.com" in url or "fonts.gstatic.com" in url:
                route.abort()
            else:
                route.continue_()

        page.route("**/*", _handle_route)
        try:
            page.goto(self.start_url, wait_until="load", timeout=self.timeout)
        except Exception:
            # Some apps never fire "load" (e.g. dongchedi, zol); fall back to domcontentloaded
            page.goto(self.start_url, wait_until="domcontentloaded", timeout=self.timeout)
        try:
            page.wait_for_load_state("networkidle", timeout=self.timeout)
        except Exception:
            pass
        page.wait_for_timeout(2000)

        goal = f"Explore the {self.env_id} web app to maximize novel JavaScript execution coverage."
        return goal, {"env_id": self.env_id, "start_url": self.start_url}

    def validate(self, page, chat_messages) -> tuple[float, bool, str, dict]:
        return 0.0, False, "", {}

    def teardown(self) -> None:
        self.server.stop()


# ── MemoryAugmentedCausalLM inference ──────────────────────────────────────
class MemoryAugAgent:
    """
    Wraps MemoryAugmentedCausalLM for step-by-step session inference.
    Maintains cumulative memory_tokens across episodes (not reset on reset()).
    """

    def __init__(self, checkpoint: str, compressor_model: str, memory_max_items: int = 512, device: str = "cuda",
                 temperature: float = 0.8, top_p: float = 0.9,
                 model_image_size: tuple[int, int] | None = None,
                 max_input_tokens: int = 8192,
                 max_new_tokens: int = 256,
                 memory_builder: str = "online_tokens",
                 delta_rank: int = 8,
                 delta_memory_slots: int = 8,
                 delta_seed: int = 13,
                 hybrid_recent_items: int = 32):
        from transformers import AutoProcessor
        from jamel.train.memory.modeling import MemoryAugmentedCausalLM
        from jamel.train.memory.delta_state_encoder import MODEL_NAME, make_history_memory_builder
        from jamel.train.memory.web_prompt import WEB_MODEL_IMAGE_SIZE

        print(f"[model] Loading MemoryAugmentedCausalLM from {checkpoint} ...")
        # SFT checkpoints saved by fsdp_sft_trainer only persist tokenizer files.
        # Fall back to the base model (recorded in memory_augment_config.json) for
        # the image processor / vision tower preprocessor.
        try:
            self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        except (OSError, ValueError):
            import json as _json, os as _os
            cfg_path = _os.path.join(checkpoint, "memory_augment_config.json")
            base = None
            if _os.path.isfile(cfg_path):
                try:
                    base = _json.load(open(cfg_path)).get("base_model_name_or_path")
                except Exception:
                    base = None
            base = os.environ.get("JAMEL_BASE_MODEL") or os.environ.get("BASE_MODEL_PATH") or base
            base = base or "Qwen/Qwen2.5-VL-7B-Instruct"
            print(
                f"[model] WARNING: preprocessor_config.json not found in checkpoint\n"
                f"        {checkpoint}\n"
                f"        Falling back to base model processor: {base}\n"
                f"        This is safe only if SFT did not modify image preprocessing.\n"
                f"        Checkpoints saved by the updated fsdp_sft_trainer.py include\n"
                f"        processor files and will not trigger this fallback.",
                flush=True,
            )
            self.processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        self.model = MemoryAugmentedCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        self.model.aligner = self.model.aligner.to(dtype=torch.bfloat16)
        self.model.eval()
        self.device = device
        self.memory_max_items = memory_max_items
        self.temperature = temperature
        self.top_p = top_p
        # Image fed to the VLM is always resized to this; viewport stays larger to preserve layout.
        self.model_image_size = model_image_size if model_image_size is not None else WEB_MODEL_IMAGE_SIZE
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.memory_builder_type = str(memory_builder).strip().lower().replace("-", "_")
        self.delta_rank = int(delta_rank)
        self.delta_memory_slots = int(delta_memory_slots)
        self.delta_seed = int(delta_seed)
        self.hybrid_recent_items = int(hybrid_recent_items)
        print(f"[model] eval image size aligned to {self.model_image_size[0]}x{self.model_image_size[1]} (resize before processor)")

        mem_cfg_path = Path(checkpoint) / "memory_augment_config.json"
        if mem_cfg_path.exists():
            mem_cfg = json.loads(mem_cfg_path.read_text())
            self.memory_hidden_size = mem_cfg.get("memory_hidden_size", 2048)
        else:
            self.memory_hidden_size = 2048
        self.use_online_delta_state = getattr(self.model, "delta_state_memory", None) is not None
        builder_type = self.memory_builder_type
        if self.use_online_delta_state and builder_type not in {"online_delta_state", "hybrid"}:
            builder_type = "online_delta_state"
        print(
            f"[model] memory_hidden_size={self.memory_hidden_size}, memory_max_items={memory_max_items}, "
            f"memory_builder={builder_type}, delta_rank={self.delta_rank}, "
            f"delta_slots={self.delta_memory_slots}"
        )

        self.memory_builder = make_history_memory_builder(
            memory_builder=builder_type,
            compressor_model_name=compressor_model,
            memory_hidden_size=self.memory_hidden_size,
            history_window=memory_max_items,
            max_memory_items=memory_max_items,
            torch_dtype="bfloat16",
            device_map=device,
            cache_history_memory=True,
            delta_rank=self.delta_rank,
            delta_memory_slots=self.delta_memory_slots,
            delta_seed=self.delta_seed,
            hybrid_recent_items=self.hybrid_recent_items,
        )
        self.memory_builder_type = builder_type
        if self.memory_builder_type in {"online_delta_state", "hybrid"}:
            print(f"[model] using {MODEL_NAME}")
        compressor = self.memory_builder.compressor
        tokenizer = getattr(getattr(compressor, "processor", None), "tokenizer", None)
        if tokenizer is not None:
            tokenizer.add_eos_token = True
        # Session-level history: NOT reset between episodes (only reset() clears browser, not memory)
        self._history_records: list[dict[str, Any]] = []

    def reset_session(self):
        """Full reset — start of a new session. Clears memory and step counter."""
        self._history_records = []
        self._session_step_idx = 0

    def _build_memory_inputs(self, *, current_image: Image.Image | None = None) -> dict[str, torch.Tensor]:
        if self.use_online_delta_state:
            history_tokens, history_mask = self.memory_builder.build_step_token_inputs(
                batch_size=1,
                history_records=[self._history_records],
            )
            recent_tokens = None
            recent_mask = None
            if self.memory_builder_type == "hybrid":
                recent_tokens, recent_mask = self.memory_builder.build_recent_memory_inputs(
                    batch_size=1,
                    history_records=[self._history_records],
                )
            if current_image is None:
                query_tokens = torch.zeros((1, 1, self.memory_hidden_size), dtype=torch.float32)
                query_mask = torch.zeros((1, 1), dtype=torch.long)
            else:
                query_tokens, query_mask = self.memory_builder.build_current_query_inputs(
                    images=[current_image],
                )
            payload = {
                "history_memory_tokens": history_tokens,
                "history_memory_attention_mask": history_mask,
                "current_memory_query_tokens": query_tokens,
                "current_memory_query_attention_mask": query_mask,
            }
            if recent_tokens is not None and recent_mask is not None:
                payload["memory_tokens"] = recent_tokens
                payload["memory_attention_mask"] = recent_mask
            return payload
        memory_tokens, memory_mask = self.memory_builder.build_memory_inputs(
            batch_size=1,
            history_records=[self._history_records],
        )
        return {
            "memory_tokens": memory_tokens,
            "memory_attention_mask": memory_mask,
        }

    def _build_prompt(self, obs_dict: dict, target_url: str, start_url: str, max_steps: int) -> str:
        from jamel.train.memory.web_prompt import (
            build_web_prompt,
            extract_axtree_from_observation_str,
        )
        from jamel.core.env.web.axtree_utils import prune_axtree
        import urllib.parse as _urlparse

        obs_text = Observer.get_observation(obs_dict)
        axtree_raw = extract_axtree_from_observation_str(obs_text)
        pruned_axtree = prune_axtree(axtree_raw, max_chars=8000)
        path_parts = _urlparse.urlparse(start_url).path.strip("/").split("/")
        target_app = path_parts[0] if path_parts else "app"
        return build_web_prompt(
            step_idx=int(self._session_step_idx),
            target_app=target_app,
            start_url=start_url,
            open_urls=obs_dict.get("open_pages_urls", (start_url,)),
            pruned_axtree=pruned_axtree,
        )

    @torch.inference_mode()
    def decide_action(self, obs_dict: dict, target_url: str, start_url: str, max_steps: int) -> dict[str, str]:
        """Returns dict with keys: action, think, raw_response, prompt."""
        prompt = self._build_prompt(obs_dict, target_url, start_url, max_steps)

        screenshot_arr = obs_dict.get("screenshot")
        if screenshot_arr is None:
            raise RuntimeError("Web prompt requires a screenshot in obs_dict; got None.")
        query_image = Image.fromarray(screenshot_arr.astype(np.uint8))
        image = query_image
        if self.model_image_size is not None and image.size != self.model_image_size:
            image = image.resize(self.model_image_size, Image.BILINEAR)
        # Split prompt on the single <image> tag to interleave text/image content.
        segments = prompt.split("<image>")
        content: list[dict] = []
        for idx, seg in enumerate(segments):
            if seg:
                content.append({"type": "text", "text": seg})
            if idx < len(segments) - 1:
                content.append({"type": "image"})
        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs.pop("second_per_grid_ts", None)

        # Truncate to max_input_tokens — must match training max_length (run_qwen25vl_7b_sft.sh
        # MAX_LENGTH) so eval never sees prompts longer than what the model trained on.
        # Safety net only; viewport+pruning should keep prompt well under this.
        _orig_len = inputs["input_ids"].shape[1]
        if _orig_len > self.max_input_tokens:
            for _k in list(inputs.keys()):
                if hasattr(inputs[_k], "shape") and inputs[_k].shape[-1] == _orig_len:
                    inputs[_k] = inputs[_k][..., -self.max_input_tokens:]

        memory_kwargs = {
            key: value.to(
                self.device,
                dtype=torch.bfloat16 if value.dtype.is_floating_point else torch.long,
            )
            for key, value in self._build_memory_inputs(current_image=query_image).items()
        }

        input_len = inputs["input_ids"].shape[1]

        # Daemon thread kills the process if generate() hangs >180s (SIGALRM can't interrupt CUDA)
        _kill_timer = threading.Timer(180, lambda: os.kill(os.getpid(), _signal.SIGKILL))
        _kill_timer.daemon = True
        _kill_timer.start()
        try:
            generated = self.model.generate(
                **inputs,
                **memory_kwargs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p if self.temperature > 0 else None,
            )
        finally:
            _kill_timer.cancel()
        new_ids = generated[:, input_len:]
        raw = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        action_m = ACTION_RE.search(raw)
        think_m  = THINK_RE.search(raw)
        action = action_m.group(1).strip().split("\n")[0] if action_m else ""
        think  = think_m.group(1).strip() if think_m else ""
        return {"action": action, "think": think, "raw_response": raw, "prompt": prompt}

    def record_step(self, obs_dict: dict, action: str):
        """Append step to session-level history (survives browser reset)."""
        screenshot_arr = obs_dict.get("screenshot")
        image = (
            Image.fromarray(screenshot_arr.astype(np.uint8))
            if screenshot_arr is not None
            else None
        )
        self._history_records.append({"image_obs": image, "action": action})
        if len(self._history_records) > self.memory_max_items:
            self._history_records = self._history_records[-self.memory_max_items:]
        self._session_step_idx += 1


def _obs_to_bytes(obs: dict | None, key: str) -> bytes | None:
    """Encode screenshot numpy array → PNG bytes."""
    if obs is None:
        return None
    arr = obs.get(key)
    if arr is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _to_json_str(obj) -> str | None:
    """Serialize arbitrary Python objects to JSON string for parquet storage."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _build_row(
    *,
    step_idx: int,
    episode_idx: int,
    obs: dict,
    next_obs: dict | None,
    info: dict,
    next_info: dict | None,
    prompt: str,
    raw_response: str,
    think: str,
    action: str,
    reward_details: dict,
    coverage_artifact: dict,
    target_app: str,
    target_url: str,
    start_url: str,
    timestamp: str,
    error: str = "",
    episode_boundary: bool = False,
) -> dict[str, Any]:
    """Build a trajectory row aligned with training parquet schema."""
    before_screenshot = _obs_to_bytes(obs, "screenshot")
    after_screenshot  = _obs_to_bytes(next_obs, "screenshot") if next_obs else None

    def _axtree(o):
        if o is None:
            return ""
        try:
            return flatten_axtree_to_str(o.get("axtree_object", {}))
        except Exception:
            return ""

    return {
        # ── observation fields ───────────────────────────────────────────
        "before_screenshot":        before_screenshot,
        "after_screenshot":         after_screenshot,
        "before_axtree_object":     _to_json_str(obs.get("axtree_object")),
        "after_axtree_object":      _to_json_str(next_obs.get("axtree_object") if next_obs else None),
        "before_dom_object":        _to_json_str(obs.get("dom_object")),
        "after_dom_object":         _to_json_str(next_obs.get("dom_object") if next_obs else None),
        "before_last_action":       obs.get("last_action"),
        "after_last_action":        next_obs.get("last_action") if next_obs else None,
        "before_last_action_error": obs.get("last_action_error") or "",
        "after_last_action_error":  next_obs.get("last_action_error") or "" if next_obs else "",
        "before_open_pages_urls":   _to_json_str(obs.get("open_pages_urls")),
        "after_open_pages_urls":    _to_json_str(next_obs.get("open_pages_urls") if next_obs else None),
        "before_open_pages_titles": _to_json_str(obs.get("open_pages_titles")),
        "after_open_pages_titles":  _to_json_str(next_obs.get("open_pages_titles") if next_obs else None),
        "before_active_page_index": obs.get("active_page_index"),
        "after_active_page_index":  next_obs.get("active_page_index") if next_obs else None,
        "before_info":              _to_json_str(info),
        "after_info":               _to_json_str(next_info),
        "before_observation_str":   _axtree(obs),
        "after_observation_str":    _axtree(next_obs),
        # ── model fields ────────────────────────────────────────────────
        "prompt":                   prompt,
        "response":                 raw_response,
        "think":                    think,
        "action":                   action,
        "raw_content":              raw_response,
        "memory_content":           think,
        "parsed_content":           _to_json_str({"action": action, "think": think}),
        "result":                   None,
        "action_format_valid":      bool(action),
        "action_execution_valid":   not bool(error),
        "action_validation_error":  None,
        # ── reward / coverage ───────────────────────────────────────────
        "reward":                   reward_details.get("reward", 0.0),
        "coverage_delta_score":     reward_details.get("delta_score", 0),
        "coverage_previous_score":  reward_details.get("previous_score", 0),
        "coverage_current_score":   reward_details.get("current_score", 0),
        "coverage_skip_reason":     reward_details.get("skip_reason"),
        **coverage_artifact,
        # ── session schema ──────────────────────────────────────────────
        "target_app":               target_app,
        "target_url":               target_url,
        "start_url":                start_url,
        "timestamp":                timestamp,
        "episode_idx":              episode_idx,
        "step_idx":                 step_idx,
        "episode_boundary":         episode_boundary,
        "extra_fields":             _to_json_str({"error": error} if error else {}),
    }


# ── session runner ──────────────────────────────────────────────────────────
def run_session(args: argparse.Namespace, agent: "MemoryAugAgent | None" = None) -> dict:
    """Run one app session (max_steps total across all episodes).

    A session holds one memory buffer shared across all episodes.  An episode
    ends when the agent emits reset() or max_steps is exhausted.

    If agent is provided, reuse it (model already loaded).
    """
    output_dir = Path(args.output)
    coverage_dir = output_dir / "coverage"
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)

    target_url = f"http://127.0.0.1:{args.port}/{args.env_id}/index.html"
    start_url  = target_url

    # ── load model (only if not provided) ──
    if agent is None:
        agent = MemoryAugAgent(
            checkpoint=args.checkpoint,
            compressor_model=args.compressor_model,
            memory_max_items=args.memory_max_items,
            device=args.device,
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 1.0),
            model_image_size=(args.model_image_width, args.model_image_height),
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            memory_builder=getattr(args, "memory_builder", "online_tokens"),
            delta_rank=getattr(args, "delta_rank", 8),
            delta_memory_slots=getattr(args, "delta_memory_slots", 8),
            delta_seed=getattr(args, "delta_seed", 13),
            hybrid_recent_items=getattr(args, "hybrid_recent_items", 32),
        )
    agent.reset_session()

    # ── build BrowserEnv ──
    print(f"\n[env] Starting BrowserGym env: {args.env_id}")
    env = BrowserEnv(
        task_entrypoint=ScaleWoBTask,
        task_kwargs={
                "env_id": args.env_id, "port": args.port,
                "viewport_width": args.viewport_width, "viewport_height": args.viewport_height,
                "scalewob_root": args.scalewob_root,
            },
        headless=True,
        viewport={"width": args.viewport_width, "height": args.viewport_height},
    )

    trajectory: list[dict] = []
    cdp_session = None

    def _start_coverage_on_page():
        nonlocal cdp_session
        try:
            page = env.unwrapped.page
            cdp_session = start_coverage(page)
        except Exception as e:
            print(f"  [coverage] Failed to start coverage: {e}")
            cdp_session = None

    def _save_step_coverage(global_step: int) -> Path | None:
        if cdp_session is None:
            return None
        cov_path = coverage_dir / f"step_{global_step:04d}.json"
        try:
            save_coverage(cdp_session, str(cov_path))
            return cov_path
        except Exception as e:
            print(f"  [coverage] Save failed at step {global_step}: {e}")
            return None

    def _reset_with_timeout(seed=None):
        _kt = threading.Timer(600, lambda: os.kill(os.getpid(), _signal.SIGKILL))
        _kt.daemon = True; _kt.start()
        try:
            result = env.reset(seed=seed)
        finally:
            _kt.cancel()
        return result

    try:
        obs, info = _reset_with_timeout(seed=args.seed)
        _start_coverage_on_page()
        print(f"[env] Reset done. Start URL: {target_url}\n")

        cumulative_reward = 0.0
        last_coverage_score = 0
        history_cov_paths: list[Path] = []   # coverage paths of all normal steps so far
        episode_idx = 0
        global_step = 0  # monotonically increasing across all episodes

        for step_idx_in_session in range(args.max_steps):
            print(f"── Step {step_idx_in_session + 1}/{args.max_steps} (ep {episode_idx}) " + "─" * 30)
            ts = datetime.now().isoformat()

            # ── inference ──
            result = agent.decide_action(obs, target_url, start_url, args.max_steps)
            action_str = result["action"]
            think_str  = result["think"]
            raw_resp   = result["raw_response"]
            prompt_str = result["prompt"]

            print(f"  think:  {think_str[:120]}{'...' if len(think_str) > 120 else ''}")
            print(f"  action: {action_str}")

            if not action_str:
                print("  [WARN] Empty action, inserting noop()")
                action_str = "noop()"

            # ── reset() branch: browser reset, memory retained ──
            if action_str.strip() == "reset()":
                print("  [reset] Browser reset — memory retained across episode boundary.")
                agent.record_step(obs, action_str)

                cov_path = _save_step_coverage(global_step)
                cov_artifact = build_coverage_artifact_fields(cov_path)

                row = _build_row(
                    step_idx=global_step,
                    episode_idx=episode_idx,
                    obs=obs,
                    next_obs=None,
                    info=info,
                    next_info=None,
                    prompt=prompt_str,
                    raw_response=raw_resp,
                    think=think_str,
                    action=action_str,
                    reward_details={"reward": 0.0, "delta_score": 0, "previous_score": last_coverage_score,
                                    "current_score": last_coverage_score, "skip_reason": "reset_action"},
                    coverage_artifact=cov_artifact,
                    target_app=args.env_id,
                    target_url=target_url,
                    start_url=start_url,
                    timestamp=ts,
                    episode_boundary=True,
                )
                trajectory.append(row)

                # Save before-screenshot to disk
                before_bytes = _obs_to_bytes(obs, "screenshot")
                if before_bytes:
                    (output_dir / f"step_{step_idx_in_session+1:03d}_before.png").write_bytes(before_bytes)

                global_step += 1
                episode_idx += 1
                obs, info = _reset_with_timeout(seed=args.seed)
                _start_coverage_on_page()
                print(f"  [reset] Done. Back at: {target_url}")
                continue

            # ── normal step ──
            before_bytes = _obs_to_bytes(obs, "screenshot")
            if before_bytes:
                (output_dir / f"step_{step_idx_in_session+1:03d}_before.png").write_bytes(before_bytes)

            try:
                _kt = threading.Timer(600, lambda: os.kill(os.getpid(), _signal.SIGKILL))
                _kt.daemon = True; _kt.start()
                try:
                    next_obs, _raw_reward, terminated, truncated, next_info = env.step(action_str)
                finally:
                    _kt.cancel()
            except Exception as e:
                print(f"  [ERROR] env.step failed: {e}")
                agent.record_step(obs, action_str)
                cov_path = _save_step_coverage(global_step)
                cov_artifact = build_coverage_artifact_fields(cov_path)
                row = _build_row(
                    step_idx=global_step,
                    episode_idx=episode_idx,
                    obs=obs,
                    next_obs=None,
                    info=info,
                    next_info=None,
                    prompt=prompt_str,
                    raw_response=raw_resp,
                    think=think_str,
                    action=action_str,
                    reward_details={"reward": 0.0, "delta_score": 0, "previous_score": last_coverage_score,
                                    "current_score": last_coverage_score, "skip_reason": "step_error"},
                    coverage_artifact=cov_artifact,
                    target_app=args.env_id,
                    target_url=target_url,
                    start_url=start_url,
                    timestamp=ts,
                    error=str(e),
                )
                trajectory.append(row)
                if cov_path and cov_path.exists():
                    history_cov_paths.append(cov_path)
                global_step += 1
                continue

            # ── save coverage and compute reward ──
            cov_path = _save_step_coverage(global_step)
            cov_artifact = build_coverage_artifact_fields(cov_path)
            reward_details = compute_coverage_details(
                cov_path,
                history_cov_paths,
                previous_score=last_coverage_score,
            )
            if cov_path and cov_path.exists():
                history_cov_paths.append(cov_path)
            cumulative_reward += reward_details["reward"]
            last_coverage_score = int(reward_details.get("current_score", last_coverage_score) or 0)

            # after-screenshot
            after_bytes = _obs_to_bytes(next_obs, "screenshot")
            if after_bytes:
                (output_dir / f"step_{step_idx_in_session+1:03d}_after.png").write_bytes(after_bytes)

            print(
                f"  reward: {reward_details['reward']:+.4f}"
                f"  Δcov={reward_details.get('delta_score', 0)}"
                f"  (cumulative: {cumulative_reward:.4f})"
            )

            row = _build_row(
                step_idx=global_step,
                episode_idx=episode_idx,
                obs=obs,
                next_obs=next_obs,
                info=info,
                next_info=next_info,
                prompt=prompt_str,
                raw_response=raw_resp,
                think=think_str,
                action=action_str,
                reward_details=reward_details,
                coverage_artifact=cov_artifact,
                target_app=args.env_id,
                target_url=target_url,
                start_url=start_url,
                timestamp=ts,
            )
            trajectory.append(row)

            agent.record_step(obs, action_str)
            obs, info = next_obs, next_info
            global_step += 1

            if terminated or truncated:
                print("\n[env] Episode finished early (terminated/truncated).")
                break

    finally:
        env.close()

    # ── save trajectory ──
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_path = output_dir / f"trajectory_{args.env_id}_{ts_str}.parquet"
    df = pd.DataFrame(trajectory)
    df.to_parquet(traj_path)
    print(f"\n[save] Trajectory: {traj_path}  ({len(df)} rows)")

    summary = {
        "checkpoint":        args.checkpoint,
        "env_id":            args.env_id,
        "steps":             len(trajectory),
        "episodes":          episode_idx + 1,
        "cumulative_reward": cumulative_reward,
        "inference_hyperparams": {
            "do_sample":          args.temperature > 0,
            "temperature":        args.temperature,
            "top_p":              args.top_p,
            "max_new_tokens":     agent.max_new_tokens,
            "memory_max_items":   args.memory_max_items,
            "memory_hidden_size": agent.memory_hidden_size,
            "prompt_format":      "web_prompt",
        },
        "actions": [t["action"] for t in trajectory],
        "rewards": [t["reward"] for t in trajectory],
        "coverage_delta_scores": [t.get("coverage_delta_score", 0) for t in trajectory],
        "trajectory_path": str(traj_path),
    }
    summary_path = output_dir / f"summary_{args.env_id}_{ts_str}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[save] Summary:    {summary_path}")

    print("\n=== Session Summary ===")
    for i, t in enumerate(trajectory):
        cov = t.get("coverage_delta_score", 0)
        ep  = t.get("episode_idx", 0)
        bnd = " [RESET]" if t.get("episode_boundary") else ""
        ef = t.get("extra_fields") or {}
        if isinstance(ef, str):
            try: ef = json.loads(ef)
            except Exception: ef = {}
        err = f"  ERR: {ef.get('error','')}" if ef.get("error") else ""
        print(f"  step {i+1:2d} ep{ep} | r={t['reward']:+.4f} Δcov={cov:4d} | {t['action'][:70]}{bnd}{err}")
    print(f"\nTotal reward: {cumulative_reward:.4f}  Episodes: {episode_idx + 1}")
    return summary


def run_all_apps(args: argparse.Namespace) -> None:
    """Load model once, evaluate all apps in args.env_ids sequentially."""
    import copy

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    num_sessions = max(1, int(getattr(args, "num_sessions", 1)))
    # Per-worker done file to avoid race conditions in parallel runs.
    # Port is unique per worker in shell/run_eval.sh.
    done_suffix = f"_gpu{args.gpu_id}_port{args.port}" if args.gpu_id is not None else f"_port{args.port}"
    done_file = output_base / f".done_apps{done_suffix}"
    done_apps: set[str] = set(done_file.read_text().splitlines()) if done_file.exists() else set()

    # Load model once
    agent = MemoryAugAgent(
        checkpoint=args.checkpoint,
        compressor_model=args.compressor_model,
        memory_max_items=args.memory_max_items,
        device=args.device,
        temperature=getattr(args, "temperature", 0.0),
        top_p=getattr(args, "top_p", 1.0),
        model_image_size=(args.model_image_width, args.model_image_height),
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        memory_builder=getattr(args, "memory_builder", "online_tokens"),
        delta_rank=getattr(args, "delta_rank", 8),
        delta_memory_slots=getattr(args, "delta_memory_slots", 8),
        delta_seed=getattr(args, "delta_seed", 13),
        hybrid_recent_items=getattr(args, "hybrid_recent_items", 32),
    )

    all_results = []
    port = args.port
    for app in args.env_ids:
        if app in done_apps:
            print(f"\n[skip] {app} already done")
            continue

        try:
            session_summaries = []
            for session_idx in range(num_sessions):
                print(f"\n{'='*50}")
                print(
                    f"Evaluating: {app}  ({args.env_ids.index(app)+1}/{len(args.env_ids)})  "
                    f"session {session_idx + 1}/{num_sessions}"
                )
                print(f"{'='*50}")

                app_args = copy.copy(args)
                app_args.env_id = app
                if num_sessions > 1:
                    session_output = output_base / app / f"session_{session_idx:02d}"
                else:
                    session_output = output_base / app
                app_args.output = str(session_output)
                app_args.port = port
                app_args.seed = args.seed + session_idx
                app_args.session_idx = session_idx

                session_summary = run_session(app_args, agent=agent)
                session_summary["session_idx"] = session_idx
                session_summaries.append(session_summary)

            if num_sessions == 1:
                summary = session_summaries[0]
            else:
                app_output_dir = output_base / app
                app_output_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "checkpoint": args.checkpoint,
                    "env_id": app,
                    "num_sessions": num_sessions,
                    "steps": sum(s["steps"] for s in session_summaries),
                    "episodes": sum(s["episodes"] for s in session_summaries),
                    "cumulative_reward": sum(s["cumulative_reward"] for s in session_summaries),
                    "inference_hyperparams": session_summaries[0].get("inference_hyperparams", {}),
                    "actions": [a for s in session_summaries for a in s.get("actions", [])],
                    "rewards": [r for s in session_summaries for r in s.get("rewards", [])],
                    "coverage_delta_scores": [
                        c for s in session_summaries for c in s.get("coverage_delta_scores", [])
                    ],
                    "trajectory_paths": [s.get("trajectory_path") for s in session_summaries],
                    "session_summaries": session_summaries,
                }
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_path = app_output_dir / f"summary_{app}_{ts_str}.json"
                summary["summary_path"] = str(summary_path)
                summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
                print(f"[save] App summary: {summary_path}")

            all_results.append(summary)
            done_apps.add(app)
            done_file.write_text("\n".join(sorted(done_apps)))
        except Exception as e:
            print(f"[ERROR] {app} failed: {e}")
            import traceback; traceback.print_exc()

        port += 1
        if port > 8900:
            port = args.port

    # ── aggregate report ──
    print("\n" + "="*60)
    print("FULL EVAL RESULTS")
    print("="*60)
    all_results.sort(key=lambda x: -x["cumulative_reward"])
    print(f"{'App':<22} {'Reward':>8} {'TotalCov':>10} {'Episodes':>10} {'Steps':>7}")
    print("-"*60)
    for s in all_results:
        cov = sum(s["coverage_delta_scores"])
        print(f"{s['env_id']:<22} {s['cumulative_reward']:>8.1f} {cov:>10} {s['episodes']:>10} {s['steps']:>7}")
    total_r = sum(s["cumulative_reward"] for s in all_results)
    total_c = sum(sum(s["coverage_delta_scores"]) for s in all_results)
    print("-"*60)
    print(f"{'TOTAL':<22} {total_r:>8.1f} {total_c:>10} {'':>10} {sum(s['steps'] for s in all_results):>7}")

    agg_path = output_base / "aggregate_results.json"
    agg_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "prompt_format": "web_prompt",
        "memory_builder": args.memory_builder,
        "delta_rank": args.delta_rank,
        "delta_memory_slots": args.delta_memory_slots,
        "max_steps_per_session": args.max_steps,
        "sessions_per_app": num_sessions,
        "total_apps": len(all_results),
        "total_reward": total_r,
        "total_coverage": total_c,
        "per_app": all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\n[save] Aggregate: {agg_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval one session (multi-episode) with MemoryAugmentedCausalLM")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("MODEL_PATH"),
        help="Path to trained checkpoint",
    )
    p.add_argument(
        "--compressor-model",
        default=os.environ.get("COMPRESSOR_MODEL"),
        help="Local Qwen3-VL-2B compressor model directory (hidden_size=2048)",
    )
    p.add_argument("--env-id", default="weibo", help="Single ScaleWoB env id (ignored when --env-ids is set)")
    p.add_argument("--env-ids", nargs="+", default=None, help="Multiple env ids for batch eval (model loaded once)")
    p.add_argument("--gpu-id", type=int, default=None, help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--num-sessions", type=int, default=1, help="Independent sessions to run per app")
    p.add_argument("--memory-max-items", type=int, default=512)
    p.add_argument(
        "--memory-builder",
        choices=["online_tokens", "online_delta_state", "hybrid"],
        default=os.environ.get("MEMORY_BUILDER", "online_tokens"),
        help="History compressor for memory tokens. online_delta_state and hybrid compute DeltaState inside the actor.",
    )
    p.add_argument("--delta-rank", type=int, default=int(os.environ.get("DELTA_RANK", "8")))
    p.add_argument("--delta-memory-slots", type=int, default=int(os.environ.get("DELTA_MEMORY_SLOTS", "8")))
    p.add_argument("--delta-seed", type=int, default=int(os.environ.get("DELTA_SEED", "13")))
    p.add_argument("--hybrid-recent-items", type=int, default=int(os.environ.get("HYBRID_RECENT_ITEMS", "32")))
    p.add_argument("--output", default=str(_REPO_ROOT / "outputs" / "eval_trajectory"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=8790, help="Local HTTP port for ScaleWoB")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0=greedy)")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    p.add_argument("--viewport-width", type=int, default=1280,
                   help="Browser viewport width in px. Affects DOM layout and raw screenshot size.")
    p.add_argument("--viewport-height", type=int, default=720,
                   help="Browser viewport height in px. Affects DOM layout and raw screenshot size.")
    p.add_argument("--model-image-width", type=int, default=640,
                   help="Image width fed to the VLM (px). MUST match training (web_prompt.WEB_MODEL_IMAGE_SIZE).")
    p.add_argument("--model-image-height", type=int, default=360,
                   help="Image height fed to the VLM (px). MUST match training (web_prompt.WEB_MODEL_IMAGE_SIZE).")
    p.add_argument("--max-input-tokens", type=int, default=8192,
                   help="Max input token length; truncate prompt if exceeded. MUST match training MAX_LENGTH.")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max tokens to generate per step.")
    p.add_argument("--scalewob-root", type=str, default=None,
                   help="Path to scalewob-env static files directory. "
                        "Defaults to SCALEWOB_ROOT or env/browser_env/scalewob-env in the release tree.")
    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    _args = parse_args()
    if not _args.checkpoint:
        import sys
        sys.exit("ERROR: --checkpoint is required (or set MODEL_PATH env var)")
    if not _args.compressor_model:
        import sys
        sys.exit("ERROR: --compressor-model is required. Prefer MODEL_PATH=/path/to/jamel_model via shell/run_eval.sh.")
    if not Path(_args.compressor_model).is_dir():
        import sys
        sys.exit(f"ERROR: --compressor-model must be a local directory: {_args.compressor_model}")
    # CUDA_VISIBLE_DEVICES is set by the shell launcher before process start.
    # Setting it inside the process (os.environ[...] = ...) is too late —
    # torch lazy-init reads it on first CUDA call, which would overwrite
    # the shell's per-worker assignment and force all workers onto one GPU.
    if _args.gpu_id is not None:
        _args.device = "cuda"
    if _args.env_ids:
        run_all_apps(_args)
    else:
        run_session(_args)

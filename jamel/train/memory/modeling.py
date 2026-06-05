from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.modeling_outputs import TokenClassifierOutput

from jamel.arch.qwen3vl_compressor.memory_injector import (
    DimensionAligner,
    MemoryInjector,
)
from jamel.arch.qwen3vl_compressor.config import resolve_torch_dtype
from jamel.train.memory.delta_state_encoder import (
    DeltaStateMemoryConfig,
    DeltaStateMemoryModule,
)


_MEMORY_CONFIG_NAME = "memory_augment_config.json"
_SAFE_WEIGHTS_NAME = "model.safetensors"
_TORCH_WEIGHTS_NAME = "pytorch_model.bin"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
_TORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"


def _is_memory_augmented_checkpoint(path: str | Path) -> bool:
    checkpoint_path = Path(path)
    return checkpoint_path.is_dir() and (checkpoint_path / _MEMORY_CONFIG_NAME).is_file()


def _is_self_contained_checkpoint(path: str | Path) -> bool:
    checkpoint_path = Path(path)
    has_config = (checkpoint_path / "config.json").is_file()
    has_weights = (
        (checkpoint_path / _SAFE_WEIGHTS_NAME).is_file()
        or (checkpoint_path / _TORCH_WEIGHTS_NAME).is_file()
        or (checkpoint_path / _SAFE_WEIGHTS_INDEX_NAME).is_file()
        or (checkpoint_path / _TORCH_WEIGHTS_INDEX_NAME).is_file()
        or any(checkpoint_path.glob("model-*.safetensors"))
        or any(checkpoint_path.glob("pytorch_model-*.bin"))
    )
    return checkpoint_path.is_dir() and has_config and has_weights


def _load_memory_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path) / _MEMORY_CONFIG_NAME
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _resolve_local_model_path(model_name_or_path: str | Path, *, base_dir: Path | None = None) -> str:
    """Resolve ModelScope cache aliases for offline HuggingFace-style model ids."""
    value = str(model_name_or_path)
    path = Path(value).expanduser()
    if base_dir is not None and not path.is_absolute():
        relative_candidate = (base_dir / path).resolve()
        if relative_candidate.exists():
            return str(relative_candidate)
    if path.exists():
        return str(path.resolve())

    override = os.environ.get("JAMEL_BASE_MODEL") or os.environ.get("BASE_MODEL_PATH")
    if override and Path(override).exists():
        return str(Path(override))

    cache_root = os.environ.get("MODELSCOPE_CACHE")
    if not cache_root or "/" not in value:
        return value

    namespace, name = value.split("/", 1)
    candidates = [
        Path(cache_root) / "models" / namespace / name,
        Path(cache_root) / "models" / namespace / name.replace(".", "___"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return value


def _load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    safe_path = checkpoint_path / _SAFE_WEIGHTS_NAME
    if safe_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safe_path))
    safe_index_path = checkpoint_path / _SAFE_WEIGHTS_INDEX_NAME
    if safe_index_path.is_file():
        from safetensors.torch import load_file

        index = json.loads(safe_index_path.read_text(encoding="utf-8"))
        state_dict: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(index.get("weight_map", {}).values())):
            state_dict.update(load_file(str(checkpoint_path / shard_name)))
        return state_dict
    torch_path = checkpoint_path / _TORCH_WEIGHTS_NAME
    if torch_path.is_file():
        return torch.load(torch_path, map_location="cpu", weights_only=False)
    torch_index_path = checkpoint_path / _TORCH_WEIGHTS_INDEX_NAME
    if torch_index_path.is_file():
        index = json.loads(torch_index_path.read_text(encoding="utf-8"))
        state_dict: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(index.get("weight_map", {}).values())):
            state_dict.update(torch.load(checkpoint_path / shard_name, map_location="cpu", weights_only=False))
        return state_dict
    raise FileNotFoundError(
        f"No memory-augmented weights found under {checkpoint_path}. "
        f"Expected {_SAFE_WEIGHTS_NAME}, {_TORCH_WEIGHTS_NAME}, or a sharded weight index."
    )


def _save_checkpoint_state_dict(
    state_dict: dict[str, torch.Tensor],
    save_directory: str | Path,
    *,
    safe_serialization: bool,
) -> None:
    save_path = Path(save_directory)
    if safe_serialization:
        from safetensors.torch import save_file

        save_file(state_dict, str(save_path / _SAFE_WEIGHTS_NAME))
    else:
        torch.save(state_dict, save_path / _TORCH_WEIGHTS_NAME)


def _copy_optional_model_file(src_dir: str | Path, dst_dir: str | Path, filename: str) -> None:
    src = Path(src_dir) / filename
    if src.is_file():
        shutil.copy2(src, Path(dst_dir) / filename)


def _infer_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model does not expose a config object.")
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError("Unable to infer hidden_size from model config.")


def _repeat_batch(tensor: torch.Tensor, target_batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == target_batch_size:
        return tensor
    if target_batch_size % tensor.shape[0] != 0:
        raise ValueError(
            f"Cannot expand tensor batch from {tensor.shape[0]} to {target_batch_size}."
        )
    repeat_factor = target_batch_size // tensor.shape[0]
    return tensor.repeat_interleave(repeat_factor, dim=0)


def _profile_enabled() -> bool:
    return os.environ.get("JAMEL_PROFILE", "0") == "1"


def _profile_log(event: str, *, start_time: float | None = None, **kwargs: Any) -> None:
    if not _profile_enabled():
        return
    payload = {key: value for key, value in kwargs.items()}
    if start_time is not None:
        payload["elapsed_s"] = round(time.perf_counter() - start_time, 6)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    extra = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"[PROFILE {timestamp}] {event} {extra}".rstrip(), flush=True)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _online_delta_state_enabled(memory_augment_config: dict[str, Any]) -> bool:
    builder = str(memory_augment_config.get("memory_builder", "")).strip().lower().replace("-", "_")
    return bool(
        _as_bool(memory_augment_config.get("enable_online_delta_state", False))
        or _as_bool(memory_augment_config.get("online_delta_state", False))
        or builder == "online_delta_state"
    )


def _normalize_multimodal_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
) -> torch.Tensor | None:
    if position_ids is None or position_ids.ndim != 3:
        return position_ids
    if position_ids.shape[1] == batch_size:
        return position_ids.contiguous()
    if position_ids.shape[0] == batch_size:
        return position_ids.transpose(0, 1).contiguous()
    raise ValueError(
        "Rank-3 multimodal position_ids must be native Qwen-VL style [C, B, S] or "
        f"batch-first [B, C, S]. Got shape={tuple(position_ids.shape)} with batch_size={batch_size}."
    )


def _masked_rms(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    while mask.ndim < hidden_states.ndim:
        mask = mask.unsqueeze(-1)
    squared = hidden_states.float().pow(2) * mask.float()
    denom = (mask.float().sum(dim=(-1, -2)) * hidden_states.shape[-1]).clamp_min(1.0)
    return torch.sqrt(squared.sum(dim=(-1, -2)) / denom).to(hidden_states.dtype)


class _MemoryAugmentedBase(nn.Module):
    def __init__(
        self,
        *,
        base_model_name_or_path: str,
        memory_hidden_size: int | str | None,
        torch_dtype: str | torch.dtype | None = None,
        trust_remote_code: bool = True,
        llm: Optional[nn.Module] = None,
        memory_augment_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.memory_augment_config = dict(memory_augment_config or {})
        self.base_model_name_or_path = base_model_name_or_path
        self.llm = llm or self._load_backbone(
            base_model_name_or_path=base_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.config = self.llm.config
        _nsm = getattr(self.llm, "_no_split_modules", None)
        self._no_split_modules = list(_nsm) if _nsm is not None else None
        self.llm_hidden_size = _infer_hidden_size(self.llm)
        if memory_hidden_size is None or memory_hidden_size == "auto":
            self.memory_hidden_size = self.llm_hidden_size
        else:
            self.memory_hidden_size = int(memory_hidden_size)
        self.memory_augment_config.setdefault("memory_hidden_size", self.memory_hidden_size)
        self.aligner = DimensionAligner(
            src_dim=self.memory_hidden_size,
            tgt_dim=self.llm_hidden_size,
        )
        self.injector = MemoryInjector()
        self.delta_state_memory: DeltaStateMemoryModule | None = None
        if _online_delta_state_enabled(self.memory_augment_config):
            self.delta_state_memory = DeltaStateMemoryModule(
                memory_hidden_size=self.memory_hidden_size,
                delta_config=DeltaStateMemoryConfig(
                    rank=int(self.memory_augment_config.get("delta_rank", 8)),
                    memory_slots=int(self.memory_augment_config.get("delta_memory_slots", 8)),
                    seed=int(self.memory_augment_config.get("delta_seed", 13)),
                    beta_bias=float(self.memory_augment_config.get("delta_beta_bias", -1.5)),
                    value_layer_norm=_as_bool(
                        self.memory_augment_config.get("delta_value_layer_norm", True),
                        default=True,
                    ),
                ),
                read_with_current_query=_as_bool(
                    self.memory_augment_config.get("read_with_current_query", True),
                    default=True,
                ),
            )

    @staticmethod
    def _load_backbone(
        *,
        base_model_name_or_path: str,
        torch_dtype: str | torch.dtype | None,
        trust_remote_code: bool,
    ) -> nn.Module:
        resolved_dtype = resolve_torch_dtype(torch_dtype)
        try:
            return AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=resolved_dtype,
                trust_remote_code=trust_remote_code,
            )
        except ValueError as causal_error:
            if "Unrecognized configuration class" not in str(causal_error):
                raise
        return AutoModelForImageTextToText.from_pretrained(
            base_model_name_or_path,
            torch_dtype=resolved_dtype,
            trust_remote_code=trust_remote_code,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        memory_hidden_size: int | str | None = None,
        memory_augment_config: Optional[dict[str, Any]] = None,
        config=None,
        torch_dtype: str | torch.dtype | None = None,
        trust_remote_code: bool = True,
        **_: Any,
    ):
        memory_augment_config = memory_augment_config or {}
        checkpoint_path = Path(pretrained_model_name_or_path)
        metadata: dict[str, Any] = {}
        if _is_memory_augmented_checkpoint(checkpoint_path):
            metadata = _load_memory_metadata(checkpoint_path)
        metadata_memory_config = metadata.get("memory_augment_config") or {}
        if metadata_memory_config:
            memory_augment_config = {
                **metadata_memory_config,
                **memory_augment_config,
            }

        resolved_hidden_size = (
            memory_hidden_size
            or memory_augment_config.get("memory_hidden_size")
            or metadata.get("memory_hidden_size")
        )
        if resolved_hidden_size is None and config is not None:
            resolved_hidden_size = getattr(config, "memory_hidden_size", None)

        if metadata and _is_self_contained_checkpoint(checkpoint_path):
            base_model_name_or_path = str(checkpoint_path)
        else:
            base_model_name_or_path = _resolve_local_model_path(
                metadata.get("base_model_name_or_path", pretrained_model_name_or_path),
                base_dir=checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent,
            )
        model = cls(
            base_model_name_or_path=base_model_name_or_path,
            memory_hidden_size=resolved_hidden_size,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            memory_augment_config=memory_augment_config,
        )

        if metadata:
            state_dict = _load_checkpoint_state_dict(checkpoint_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            allowed_missing_suffixes = {"lm_head.weight"}
            unexpected_keys = [key for key in unexpected_keys if not key.startswith("_metadata")]
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys when loading memory checkpoint: {unexpected_keys[:20]}")
            important_missing = [
                key for key in missing_keys
                if not any(key.endswith(suffix) for suffix in allowed_missing_suffixes)
                and not key.startswith("aligner.")
            ]
            if important_missing:
                raise RuntimeError(f"Missing keys when loading memory checkpoint: {important_missing[:20]}")
            model.base_model_name_or_path = str(base_model_name_or_path)
        return model

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def can_generate(self) -> bool:
        if hasattr(self.llm, "can_generate"):
            return bool(self.llm.can_generate())
        return hasattr(self.llm, "generate")

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.llm, "prepare_inputs_for_generation"):
            return self.llm.prepare_inputs_for_generation(*args, **kwargs)
        raise AttributeError(f"{self.__class__.__name__} does not support prepare_inputs_for_generation")

    @property
    def generation_config(self):
        return getattr(self.llm, "generation_config", None)

    @generation_config.setter
    def generation_config(self, value):
        self.llm.generation_config = value

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            return self.llm.gradient_checkpointing_enable(*args, **kwargs)
        return None

    def enable_input_require_grads(self):
        if hasattr(self.llm, "enable_input_require_grads"):
            return self.llm.enable_input_require_grads()
        embed_layer = self.get_input_embeddings()

        def _make_inputs_require_grads(_, __, output):
            output.requires_grad_(True)

        embed_layer.register_forward_hook(_make_inputs_require_grads)

    def _prepare_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
    ):
        if memory_tokens is None:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

        if input_ids.ndim != 2:
            raise ValueError("Memory-augmented models require padded rank-2 input_ids.")
        if attention_mask is None:
            raise ValueError("memory_tokens require an attention_mask.")

        if memory_tokens.ndim == 2:
            memory_tokens = memory_tokens.unsqueeze(1)
        if memory_tokens.ndim != 3:
            raise ValueError("memory_tokens must have shape [B, H] or [B, N, H].")

        memory_tokens = _repeat_batch(memory_tokens, input_ids.shape[0])
        position_ids = _normalize_multimodal_position_ids(
            position_ids,
            batch_size=input_ids.shape[0],
        )
        if memory_attention_mask is not None:
            if memory_attention_mask.ndim == 1:
                memory_attention_mask = memory_attention_mask.unsqueeze(0)
            memory_attention_mask = _repeat_batch(memory_attention_mask, input_ids.shape[0])

        embed_weight = self.get_input_embeddings().weight
        input_ids = input_ids.to(embed_weight.device)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        memory_tokens = memory_tokens.to(device=embed_weight.device, dtype=embed_weight.dtype)
        aligned_memory = self.aligner(memory_tokens)
        if memory_attention_mask is None:
            memory_attention_mask = torch.ones(
                aligned_memory.shape[:2],
                dtype=torch.long,
                device=aligned_memory.device,
            )
        else:
            memory_attention_mask = memory_attention_mask.to(
                device=aligned_memory.device,
                dtype=torch.long,
            )

        text_attention_mask = attention_mask
        if text_attention_mask is None:
            text_attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.long,
                device=aligned_memory.device,
            )
        else:
            text_attention_mask = text_attention_mask.to(
                device=aligned_memory.device,
                dtype=torch.long,
            )

        stats_start = time.perf_counter()
        with torch.no_grad():
            target_rms = _masked_rms(inputs_embeds.detach(), text_attention_mask)
            memory_rms = _masked_rms(aligned_memory.detach(), memory_attention_mask)
            scale = (target_rms / memory_rms.clamp_min(1e-6)).clamp(max=1.0)
        aligned_memory = aligned_memory * scale.view(-1, 1, 1)
        with torch.no_grad():
            memory_rms_after = _masked_rms(aligned_memory.detach(), memory_attention_mask)
        _profile_log(
            "memory.inject_stats",
            start_time=stats_start,
            batch_size=input_ids.shape[0],
            prefix_length=aligned_memory.shape[1],
            text_rms=round(float(target_rms.mean().item()), 6),
            memory_rms_before=round(float(memory_rms.mean().item()), 6),
            memory_rms_after=round(float(memory_rms_after.mean().item()), 6),
            downscaled=int((scale < 0.999).sum().item()),
        )
        injected_inputs = self.injector.inject(
            model=self.llm,
            memory_tokens=aligned_memory,
            memory_attention_mask=memory_attention_mask,
            inputs_embeds=inputs_embeds,
            attention_mask=text_attention_mask,
            position_ids=position_ids,
        )
        return {
            "inputs_embeds": injected_inputs.inputs_embeds,
            "attention_mask": injected_inputs.attention_mask,
            "position_ids": injected_inputs.position_ids,
            "prefix_length": injected_inputs.prefix_length,
        }

    def _resolve_memory_inputs(
        self,
        *,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        history_memory_tokens: Optional[torch.Tensor] = None,
        history_memory_attention_mask: Optional[torch.Tensor] = None,
        current_memory_query_tokens: Optional[torch.Tensor] = None,
        current_memory_query_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if history_memory_tokens is None:
            return memory_tokens, memory_attention_mask
        if self.delta_state_memory is None:
            raise ValueError(
                "history_memory_tokens were provided, but online DeltaState memory is not enabled. "
                "Set model.memory_augment.enable_online_delta_state=True."
            )
        return self.delta_state_memory.tokens_to_state_memory_batch(
            history_tokens=history_memory_tokens,
            history_attention_mask=history_memory_attention_mask,
            current_query_tokens=current_memory_query_tokens,
            current_query_attention_mask=current_memory_query_attention_mask,
        )

    @staticmethod
    def _prefix_labels(
        labels: torch.Tensor,
        prefix_length: int,
        device: torch.device,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        labels = labels.to(device)
        prefix = torch.full(
            (labels.shape[0], prefix_length),
            ignore_index,
            dtype=labels.dtype,
            device=device,
        )
        return torch.cat([prefix, labels], dim=1)

    def _save_memory_metadata(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        metadata = {
            "format": "jamel_memory_augmented_lm",
            "format_version": 1,
            "self_contained": True,
            "base_model_name_or_path": self.base_model_name_or_path,
            "memory_hidden_size": self.memory_hidden_size,
            "class_name": self.__class__.__name__,
            "memory_augment_config": self.memory_augment_config,
        }
        (save_path / _MEMORY_CONFIG_NAME).write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def _save_pretrained_common(
        self,
        save_directory: str | Path,
        state_dict=None,
        *,
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        if state_dict is None and safe_serialization:
            from safetensors.torch import save_model

            save_model(self, str(save_path / _SAFE_WEIGHTS_NAME))
        else:
            state_dict = state_dict or self.state_dict()
            state_dict = {key: value.detach().cpu() for key, value in state_dict.items()}
            _save_checkpoint_state_dict(
                state_dict,
                save_path,
                safe_serialization=safe_serialization,
            )
        self.config.save_pretrained(save_path)
        if self.generation_config is not None:
            self.generation_config.save_pretrained(save_path)
        self._save_memory_metadata(save_path)

        processing_class = kwargs.get("processing_class") or kwargs.get("processor") or kwargs.get("tokenizer")
        if processing_class is not None and hasattr(processing_class, "save_pretrained"):
            processing_class.save_pretrained(save_path)

        for filename in (
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.json",
            "vocab.json",
            "merges.txt",
        ):
            _copy_optional_model_file(self.base_model_name_or_path, save_path, filename)


class MemoryAugmentedCausalLM(_MemoryAugmentedBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        history_memory_tokens: Optional[torch.Tensor] = None,
        history_memory_attention_mask: Optional[torch.Tensor] = None,
        current_memory_query_tokens: Optional[torch.Tensor] = None,
        current_memory_query_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        memory_tokens, memory_attention_mask = self._resolve_memory_inputs(
            memory_tokens=memory_tokens,
            memory_attention_mask=memory_attention_mask,
            history_memory_tokens=history_memory_tokens,
            history_memory_attention_mask=history_memory_attention_mask,
            current_memory_query_tokens=current_memory_query_tokens,
            current_memory_query_attention_mask=current_memory_query_attention_mask,
        )
        model_inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            memory_tokens=memory_tokens,
            memory_attention_mask=memory_attention_mask,
        )
        prefix_length = model_inputs.pop("prefix_length", 0)
        if labels is not None and prefix_length > 0:
            labels = self._prefix_labels(
                labels=labels,
                prefix_length=prefix_length,
                device=model_inputs["inputs_embeds"].device,
            )
        if labels is not None:
            model_inputs["labels"] = labels
        output = self.llm(**model_inputs, **kwargs)
        try:
            output.memory_prefix_length = prefix_length
        except Exception:
            pass
        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        history_memory_tokens: Optional[torch.Tensor] = None,
        history_memory_attention_mask: Optional[torch.Tensor] = None,
        current_memory_query_tokens: Optional[torch.Tensor] = None,
        current_memory_query_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        memory_tokens, memory_attention_mask = self._resolve_memory_inputs(
            memory_tokens=memory_tokens,
            memory_attention_mask=memory_attention_mask,
            history_memory_tokens=history_memory_tokens,
            history_memory_attention_mask=history_memory_attention_mask,
            current_memory_query_tokens=current_memory_query_tokens,
            current_memory_query_attention_mask=current_memory_query_attention_mask,
        )
        model_inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            memory_tokens=memory_tokens,
            memory_attention_mask=memory_attention_mask,
        )
        model_inputs.pop("prefix_length", None)
        outputs = self.llm.generate(**model_inputs, **kwargs)

        prompt_ids = input_ids
        if hasattr(outputs, "sequences"):
            sequences = outputs.sequences
            if sequences.shape[0] != prompt_ids.shape[0]:
                prompt_ids = _repeat_batch(prompt_ids, sequences.shape[0])
            if sequences.shape[1] <= kwargs.get("max_new_tokens", 0):
                outputs.sequences = torch.cat([prompt_ids.to(sequences.device), sequences], dim=1)
            return outputs

        if outputs.shape[0] != prompt_ids.shape[0]:
            prompt_ids = _repeat_batch(prompt_ids, outputs.shape[0])
        if outputs.shape[1] <= kwargs.get("max_new_tokens", 0):
            return torch.cat([prompt_ids.to(outputs.device), outputs], dim=1)
        return outputs

    def save_pretrained(self, save_directory: str | Path, state_dict=None, **kwargs: Any) -> None:
        self._save_pretrained_common(save_directory, state_dict=state_dict, **kwargs)


class MemoryAugmentedValueModel(_MemoryAugmentedBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.score = nn.Linear(self.llm_hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_attention_mask: Optional[torch.Tensor] = None,
        history_memory_tokens: Optional[torch.Tensor] = None,
        history_memory_attention_mask: Optional[torch.Tensor] = None,
        current_memory_query_tokens: Optional[torch.Tensor] = None,
        current_memory_query_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        memory_tokens, memory_attention_mask = self._resolve_memory_inputs(
            memory_tokens=memory_tokens,
            memory_attention_mask=memory_attention_mask,
            history_memory_tokens=history_memory_tokens,
            history_memory_attention_mask=history_memory_attention_mask,
            current_memory_query_tokens=current_memory_query_tokens,
            current_memory_query_attention_mask=current_memory_query_attention_mask,
        )
        model_inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            memory_tokens=memory_tokens,
            memory_attention_mask=memory_attention_mask,
        )
        model_inputs.pop("prefix_length", None)
        outputs = self.llm(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.hidden_states[-1]
        logits = self.score(hidden_states)
        return TokenClassifierOutput(logits=logits)

    def save_pretrained(self, save_directory: str | Path, state_dict=None, **kwargs: Any) -> None:
        self._save_pretrained_common(save_directory, state_dict=state_dict, **kwargs)

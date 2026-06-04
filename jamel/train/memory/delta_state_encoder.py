from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from jamel.arch.qwen3vl_compressor.screen_compressor import ScreenCompressor
else:
    ScreenCompressor = Any


MODEL_NAME = "JAMEL-DeltaState"
VALID_MEMORY_BUILDERS = ("online_tokens", "delta_state", "hybrid")


def _normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=eps)


def _as_memory_rows(token: torch.Tensor, *, hidden_size: int) -> torch.Tensor:
    if token.ndim == 1:
        token = token.unsqueeze(0)
    if token.ndim != 2 or token.shape[-1] != hidden_size:
        raise ValueError(
            f"Expected memory token rows [N, {hidden_size}], got {tuple(token.shape)}."
        )
    return token.detach().float().cpu()


def _projection_matrix(rows: int, cols: int, *, seed: int, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
    matrix = matrix / max(1.0, float(cols) ** 0.5)
    return matrix * float(scale)


def _to_pil_image(image: Any):
    from PIL import Image
    import numpy as np

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = image if image.dtype == np.uint8 else image.astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported image type for memory compression: {type(image)!r}")


@dataclass(frozen=True)
class DeltaStateMemoryConfig:
    """Configuration for the level-1 JAMEL-DeltaState memory compressor.

    This is intentionally a global session-memory compressor, not the full
    layer-wise delta-mem module. It consumes per-step compressor embeddings,
    updates a compact associative state, and emits fixed-count JAMEL memory
    tokens compatible with the existing actor path.
    """

    rank: int = 8
    memory_slots: int = 8
    seed: int = 13
    beta_bias: float = -1.5
    value_layer_norm: bool = True

    def __post_init__(self) -> None:
        if int(self.rank) < 1:
            raise ValueError("rank must be >= 1")
        if int(self.memory_slots) < 1:
            raise ValueError("memory_slots must be >= 1")


class DeltaStateHistoryMemoryBuilder:
    """Build fixed-size memory tokens from a global delta-rule state.

    For each historical browser step, we first reuse JAMEL's ScreenCompressor to
    produce a step embedding x_t. Then we update a value-wide associative state:

        pred_t = S @ k_t
        S_t = lambda_t * S - beta_t * outer(pred_t, k_t) + beta_t * outer(v_t, k_t)

    where k_t is a low-rank key and v_t is the step embedding. The final state is
    read with deterministic slot queries to produce K prefix memory tokens.
    """

    builder_name = "delta_state"

    def __init__(
        self,
        *,
        compressor_model_name: str,
        memory_hidden_size: int | str | None,
        history_window: int = 512,
        max_memory_items: int | None = None,
        history_action_prefix: str = "Previous action:",
        torch_dtype: str | torch.dtype | None = "auto",
        device_map: str | dict | None = "auto",
        cache_history_memory: bool = True,
        compressor: Optional[ScreenCompressor] = None,
        delta_config: DeltaStateMemoryConfig | None = None,
        delta_rank: int | None = None,
        delta_memory_slots: int | None = None,
        delta_seed: int | None = None,
    ) -> None:
        self.history_window = max(1, int(history_window))
        self.max_memory_items = None if max_memory_items is None else max(1, int(max_memory_items))
        self.history_action_prefix = history_action_prefix
        self.cache_history_memory = bool(cache_history_memory)
        if compressor is None:
            from jamel.arch.qwen3vl_compressor.screen_compressor import ScreenCompressor as _ScreenCompressor

            compressor = _ScreenCompressor(
                model_name=compressor_model_name,
                hidden_size=memory_hidden_size,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        self.compressor = compressor
        self.memory_hidden_size = int(self.compressor.hidden_size)
        base_config = delta_config or DeltaStateMemoryConfig()
        self.delta_config = DeltaStateMemoryConfig(
            rank=int(delta_rank if delta_rank is not None else base_config.rank),
            memory_slots=int(delta_memory_slots if delta_memory_slots is not None else base_config.memory_slots),
            seed=int(delta_seed if delta_seed is not None else base_config.seed),
            beta_bias=base_config.beta_bias,
            value_layer_norm=base_config.value_layer_norm,
        )
        self._init_fixed_projections()

    def _init_fixed_projections(self) -> None:
        hidden = self.memory_hidden_size
        rank = self.delta_config.rank
        seed = self.delta_config.seed
        self.q_proj = _projection_matrix(hidden, rank, seed=seed + 1)
        self.k_proj = _projection_matrix(hidden, rank, seed=seed + 2)
        self.beta_proj = _projection_matrix(hidden, 1, seed=seed + 3)
        slot_queries = _projection_matrix(self.delta_config.memory_slots, rank, seed=seed + 4)
        self.slot_queries = _normalize_rows(slot_queries)

    def _history_limit(self) -> int:
        limit = self.history_window
        if self.max_memory_items is not None:
            limit = min(limit, self.max_memory_items)
        return limit

    def _prepare_step_tokens(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> list[list[torch.Tensor]]:
        history_records = history_records or [[] for _ in range(batch_size)]
        flat_images = []
        flat_texts: list[str] = []
        sample_refs: list[list[torch.Tensor | tuple[int, int]]] = []
        flat_record_refs: list[dict[str, Any]] = []
        limit = self._history_limit()

        for sample_idx in range(batch_size):
            records = list(history_records[sample_idx])[-limit:]
            refs: list[torch.Tensor | tuple[int, int]] = []
            for record in records:
                cached = record.get("_cached_delta_state_input_token") if self.cache_history_memory else None
                if isinstance(cached, torch.Tensor):
                    refs.append(_as_memory_rows(cached, hidden_size=self.memory_hidden_size))
                    continue
                action = str(record.get("action", "")).strip() or "unknown"
                image_value = record.get("image_obs")
                if image_value is None:
                    continue
                flat_images.append(_to_pil_image(image_value))
                flat_texts.append(f"{self.history_action_prefix} {action}")
                flat_record_refs.append(record)
                refs.append((len(flat_images) - 1, len(flat_record_refs) - 1))
            sample_refs.append(refs)

        computed_memory: torch.Tensor | None = None
        if flat_images:
            computed_memory = self.compressor.compress_batch(flat_images, flat_texts).detach().cpu().float()

        resolved: list[list[torch.Tensor]] = []
        for refs in sample_refs:
            rows: list[torch.Tensor] = []
            for ref in refs:
                if isinstance(ref, torch.Tensor):
                    rows.append(ref)
                    continue
                flat_idx, record_ref_idx = ref
                assert computed_memory is not None
                token = _as_memory_rows(computed_memory[flat_idx], hidden_size=computed_memory.shape[-1])
                rows.append(token)
                if self.cache_history_memory:
                    flat_record_refs[record_ref_idx]["_cached_delta_state_input_token"] = token.clone()
            resolved.append(rows)
        return resolved

    def _tokens_to_state_memory(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.memory_hidden_size
        rank = self.delta_config.rank
        state = torch.zeros((hidden, rank), dtype=torch.float32)
        if tokens.numel() == 0:
            return torch.zeros((self.delta_config.memory_slots, hidden), dtype=torch.float32)

        values = tokens.float()
        if self.delta_config.value_layer_norm:
            values = F.layer_norm(values, (hidden,))
        q_seq = _normalize_rows(values @ self.q_proj)
        k_seq = _normalize_rows(values @ self.k_proj)
        beta_seq = torch.sigmoid(values @ self.beta_proj + float(self.delta_config.beta_bias)).squeeze(-1)

        for idx in range(values.shape[0]):
            k_t = k_seq[idx]
            v_t = values[idx]
            beta_t = beta_seq[idx].clamp(0.0, 1.0)
            lambda_t = 1.0 - beta_t
            pred_t = state @ k_t
            state = (
                lambda_t * state
                - beta_t * torch.outer(pred_t, k_t)
                + beta_t * torch.outer(v_t, k_t)
            )

        memory_tokens = torch.einsum("hr,sr->sh", state, self.slot_queries)
        source_rms = values.float().pow(2).mean().sqrt().clamp_min(1e-6)
        memory_rms = memory_tokens.float().pow(2).mean().sqrt().clamp_min(1e-6)
        memory_tokens = memory_tokens * (source_rms / memory_rms).clamp(max=1.0)
        return memory_tokens.float()

    def build_memory_inputs(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_rows = self._prepare_step_tokens(
            batch_size=batch_size,
            history_records=history_records,
        )
        memory_tokens = torch.zeros(
            (batch_size, self.delta_config.memory_slots, self.memory_hidden_size),
            dtype=torch.float32,
        )
        memory_mask = torch.zeros(
            (batch_size, self.delta_config.memory_slots),
            dtype=torch.long,
        )
        for sample_idx, rows in enumerate(sample_rows):
            if not rows:
                continue
            tokens = torch.cat(rows, dim=0)
            memory_tokens[sample_idx] = self._tokens_to_state_memory(tokens)
            memory_mask[sample_idx] = 1
        return memory_tokens, memory_mask


class HybridHistoryMemoryBuilder:
    """Concatenate JAMEL-DeltaState tokens with recent original JAMEL tokens."""

    builder_name = "hybrid"

    def __init__(
        self,
        *,
        compressor_model_name: str,
        memory_hidden_size: int | str | None,
        history_window: int = 512,
        max_memory_items: int | None = None,
        hybrid_recent_items: int = 32,
        history_action_prefix: str = "Previous action:",
        torch_dtype: str | torch.dtype | None = "auto",
        device_map: str | dict | None = "auto",
        cache_history_memory: bool = True,
        delta_rank: int = 8,
        delta_memory_slots: int = 8,
        delta_seed: int = 13,
        compressor: Optional[ScreenCompressor] = None,
    ) -> None:
        if compressor is None:
            from jamel.arch.qwen3vl_compressor.screen_compressor import ScreenCompressor as _ScreenCompressor

            compressor = _ScreenCompressor(
                model_name=compressor_model_name,
                hidden_size=memory_hidden_size,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        shared_compressor = compressor
        self.delta_builder = DeltaStateHistoryMemoryBuilder(
            compressor_model_name=compressor_model_name,
            memory_hidden_size=memory_hidden_size,
            history_window=history_window,
            max_memory_items=max_memory_items,
            history_action_prefix=history_action_prefix,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_history_memory=cache_history_memory,
            compressor=shared_compressor,
            delta_rank=delta_rank,
            delta_memory_slots=delta_memory_slots,
            delta_seed=delta_seed,
        )
        recent_limit = max(1, int(hybrid_recent_items))
        from jamel.train.memory.encoder import OnlineHistoryMemoryBuilder

        self.online_builder = OnlineHistoryMemoryBuilder(
            compressor_model_name=compressor_model_name,
            memory_hidden_size=memory_hidden_size,
            history_window=recent_limit,
            max_memory_items=recent_limit,
            history_action_prefix=history_action_prefix,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_history_memory=cache_history_memory,
            compressor=shared_compressor,
        )
        self.compressor = shared_compressor
        self.memory_hidden_size = int(shared_compressor.hidden_size)
        self.hybrid_recent_items = recent_limit

    def build_memory_inputs(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_tokens, delta_mask = self.delta_builder.build_memory_inputs(
            batch_size=batch_size,
            history_records=history_records,
        )
        online_tokens, online_mask = self.online_builder.build_memory_inputs(
            batch_size=batch_size,
            history_records=history_records,
        )
        if online_tokens.numel() == 0:
            return delta_tokens, delta_mask
        return (
            torch.cat([delta_tokens, online_tokens], dim=1),
            torch.cat([delta_mask, online_mask], dim=1),
        )


def make_history_memory_builder(
    *,
    memory_builder: str,
    compressor_model_name: str,
    memory_hidden_size: int | str | None,
    history_window: int = 512,
    max_memory_items: int | None = None,
    history_action_prefix: str = "Previous action:",
    torch_dtype: str | torch.dtype | None = "auto",
    device_map: str | dict | None = "auto",
    cache_history_memory: bool = True,
    delta_rank: int = 8,
    delta_memory_slots: int = 8,
    delta_seed: int = 13,
    hybrid_recent_items: int = 32,
    compressor: Optional[ScreenCompressor] = None,
):
    normalized = str(memory_builder).strip().lower().replace("-", "_")
    if normalized not in VALID_MEMORY_BUILDERS:
        raise ValueError(
            f"Unsupported memory_builder={memory_builder!r}; expected one of {VALID_MEMORY_BUILDERS}."
        )
    common = dict(
        compressor_model_name=compressor_model_name,
        memory_hidden_size=memory_hidden_size,
        history_window=history_window,
        max_memory_items=max_memory_items,
        history_action_prefix=history_action_prefix,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_history_memory=cache_history_memory,
        compressor=compressor,
    )
    if normalized == "online_tokens":
        from jamel.train.memory.encoder import OnlineHistoryMemoryBuilder

        return OnlineHistoryMemoryBuilder(**common)
    if normalized == "delta_state":
        return DeltaStateHistoryMemoryBuilder(
            **common,
            delta_rank=delta_rank,
            delta_memory_slots=delta_memory_slots,
            delta_seed=delta_seed,
        )
    return HybridHistoryMemoryBuilder(
        **common,
        delta_rank=delta_rank,
        delta_memory_slots=delta_memory_slots,
        delta_seed=delta_seed,
        hybrid_recent_items=hybrid_recent_items,
    )

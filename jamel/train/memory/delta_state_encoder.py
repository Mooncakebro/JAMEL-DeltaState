from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TYPE_CHECKING

import torch
from torch import nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from jamel.arch.qwen3vl_compressor.screen_compressor import ScreenCompressor
else:
    ScreenCompressor = Any


MODEL_NAME = "JAMEL-DeltaState"
VALID_MEMORY_BUILDERS = ("online_tokens", "online_delta_state", "hybrid")
CURRENT_MEMORY_QUERY_TEXT = "Current browser state for memory read."


def _as_memory_rows(token: torch.Tensor, *, hidden_size: int) -> torch.Tensor:
    if token.ndim == 1:
        token = token.unsqueeze(0)
    if token.ndim != 2 or token.shape[-1] != hidden_size:
        raise ValueError(
            f"Expected memory token rows [N, {hidden_size}], got {tuple(token.shape)}."
        )
    return token.detach().float().cpu()


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


class DeltaStateMemoryModule(nn.Module):
    """Differentiable DeltaState memory over compressed browser-step embeddings.

    Each x_t is one compressed browser step, not one LLM token. The sequence
    length T is the number of remembered history steps. Trainable memory
    projections produce low-rank q_t/k_t, full-width v_t, and per-rank beta_t
    gates. We update a value-wide associative state:

        pred_t = S @ k_t
        S_t = S * (1 - beta_t) + outer(v_t - pred_t, beta_t * k_t)

    where k_t is a low-rank key and v_t is the projected step value. At read
    time, slot queries can be conditioned on q_current so prefix memory remains
    fixed-width while the read becomes current-input dependent.
    """

    def __init__(
        self,
        *,
        memory_hidden_size: int,
        delta_config: DeltaStateMemoryConfig | None = None,
        delta_rank: int | None = None,
        delta_memory_slots: int | None = None,
        delta_seed: int | None = None,
        read_with_current_query: bool = True,
    ) -> None:
        super().__init__()
        self.memory_hidden_size = int(memory_hidden_size)
        self.read_with_current_query = bool(read_with_current_query)
        base_config = delta_config or DeltaStateMemoryConfig()
        self.delta_config = DeltaStateMemoryConfig(
            rank=int(delta_rank if delta_rank is not None else base_config.rank),
            memory_slots=int(delta_memory_slots if delta_memory_slots is not None else base_config.memory_slots),
            seed=int(delta_seed if delta_seed is not None else base_config.seed),
            beta_bias=base_config.beta_bias,
            value_layer_norm=base_config.value_layer_norm,
        )
        hidden = self.memory_hidden_size
        rank = self.delta_config.rank
        slots = self.delta_config.memory_slots
        self.W_q = nn.Linear(hidden, rank, bias=False)
        self.W_k = nn.Linear(hidden, rank, bias=False)
        self.W_v = nn.Linear(hidden, hidden, bias=False)
        self.W_beta = nn.Linear(hidden, rank, bias=True)
        self.slot_queries = nn.Parameter(torch.empty(slots, rank, dtype=torch.float32))
        self._reset_memory_parameters()

    def _reset_memory_parameters(self) -> None:
        hidden = self.memory_hidden_size
        rank = self.delta_config.rank
        seed = int(self.delta_config.seed)

        def randn_like_parameter(parameter: torch.Tensor, *, local_seed: int, fan_in: int) -> torch.Tensor:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(local_seed)
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
            return values / max(1.0, float(fan_in) ** 0.5)

        with torch.no_grad():
            self.W_q.weight.copy_(
                randn_like_parameter(self.W_q.weight, local_seed=seed + 1, fan_in=hidden)
            )
            self.W_k.weight.copy_(
                randn_like_parameter(self.W_k.weight, local_seed=seed + 2, fan_in=hidden)
            )
            self.W_v.weight.zero_()
            self.W_v.weight.diagonal().fill_(1.0)
            self.W_beta.weight.zero_()
            self.W_beta.bias.fill_(float(self.delta_config.beta_bias))
            self.slot_queries.copy_(
                randn_like_parameter(self.slot_queries, local_seed=seed + 4, fan_in=rank)
            )

    def _project_to_memory(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project step embeddings into DeltaState q/k/v/beta memory coordinates."""
        q = F.normalize(torch.tanh(self.W_q(x)), dim=-1, eps=1e-6)
        k = F.normalize(torch.tanh(self.W_k(x)), dim=-1, eps=1e-6)
        v = self.W_v(x)
        if self.delta_config.value_layer_norm:
            v = F.layer_norm(v, (self.memory_hidden_size,))
        beta = torch.sigmoid(self.W_beta(x))
        return q, k, v, beta

    def _project_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(torch.tanh(self.W_q(x)), dim=-1, eps=1e-6)

    def _current_read_queries(self, current_query_tokens: torch.Tensor | None) -> torch.Tensor:
        if (
            current_query_tokens is None
            or current_query_tokens.numel() == 0
            or not self.read_with_current_query
        ):
            return self.slot_queries
        if current_query_tokens.ndim == 1:
            current_query_tokens = current_query_tokens.unsqueeze(0)
        current_query_tokens = current_query_tokens.to(
            device=self.W_q.weight.device,
            dtype=self.W_q.weight.dtype,
        )
        q_current = self._project_query(current_query_tokens).mean(dim=0)
        q_current = F.normalize(q_current, dim=-1, eps=1e-6)
        return F.normalize(self.slot_queries + q_current.unsqueeze(0), dim=-1, eps=1e-6)

    def _read_state(
        self,
        state: torch.Tensor,
        *,
        current_query_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        read_queries = self._current_read_queries(current_query_tokens)
        return (state @ read_queries.T).T

    def _tokens_to_state_memory(
        self,
        tokens: torch.Tensor,
        *,
        current_query_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.memory_hidden_size
        rank = self.delta_config.rank
        parameter = self.W_q.weight
        device = parameter.device
        dtype = parameter.dtype
        state = torch.zeros((hidden, rank), dtype=dtype, device=device)
        if tokens.numel() == 0:
            return torch.zeros(
                (self.delta_config.memory_slots, hidden),
                dtype=torch.float32,
                device=device,
            )

        x_seq = tokens.to(device=device, dtype=dtype)
        _q_seq, k_seq, values, beta_seq = self._project_to_memory(x_seq)

        for idx in range(values.shape[0]):
            k_t = k_seq[idx]
            v_t = values[idx]
            beta_t = beta_seq[idx].clamp(0.0, 1.0)
            lambda_t = 1.0 - beta_t
            pred_t = state @ k_t
            state = lambda_t.unsqueeze(0) * state + torch.outer(v_t - pred_t, beta_t * k_t)

        memory_tokens = self._read_state(
            state,
            current_query_tokens=current_query_tokens,
        )
        source_rms = values.float().pow(2).mean().sqrt().clamp_min(1e-6)
        memory_rms = memory_tokens.float().pow(2).mean().sqrt().clamp_min(1e-6)
        memory_tokens = memory_tokens * (source_rms / memory_rms).clamp(max=1.0)
        return memory_tokens.float()

    def tokens_to_state_memory_batch(
        self,
        history_tokens: torch.Tensor,
        history_attention_mask: torch.Tensor | None = None,
        current_query_tokens: torch.Tensor | None = None,
        current_query_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_tokens.ndim != 3:
            raise ValueError(
                f"history_tokens must have shape [B, T, H], got {tuple(history_tokens.shape)}."
            )
        batch_size, _, hidden = history_tokens.shape
        if hidden != self.memory_hidden_size:
            raise ValueError(
                f"history_tokens hidden size must be {self.memory_hidden_size}, got {hidden}."
            )
        device = self.W_q.weight.device
        if history_attention_mask is None:
            history_attention_mask = torch.ones(
                history_tokens.shape[:2],
                dtype=torch.long,
                device=history_tokens.device,
            )
        if current_query_tokens is not None and current_query_tokens.ndim == 2:
            current_query_tokens = current_query_tokens.unsqueeze(1)
        if current_query_attention_mask is not None and current_query_attention_mask.ndim == 1:
            current_query_attention_mask = current_query_attention_mask.unsqueeze(1)

        sample_memory_tokens: list[torch.Tensor] = []
        sample_memory_masks: list[torch.Tensor] = []
        for sample_idx in range(batch_size):
            valid_history = history_attention_mask[sample_idx].to(dtype=torch.bool)
            sample_history = history_tokens[sample_idx][valid_history]
            sample_query = None
            if current_query_tokens is not None:
                sample_query_tokens = current_query_tokens[sample_idx]
                if current_query_attention_mask is not None:
                    valid_query = current_query_attention_mask[sample_idx].to(dtype=torch.bool)
                    sample_query_tokens = sample_query_tokens[valid_query]
                if sample_query_tokens.numel() > 0:
                    sample_query = sample_query_tokens
            if sample_history.numel() == 0:
                sample_memory_tokens.append(
                    torch.zeros(
                        (self.delta_config.memory_slots, self.memory_hidden_size),
                        dtype=torch.float32,
                        device=device,
                    )
                )
                sample_memory_masks.append(
                    torch.zeros(
                        (self.delta_config.memory_slots,),
                        dtype=torch.long,
                        device=device,
                    )
                )
                continue
            sample_memory_tokens.append(
                self._tokens_to_state_memory(
                    sample_history,
                    current_query_tokens=sample_query,
                )
            )
            sample_memory_masks.append(
                torch.ones(
                    (self.delta_config.memory_slots,),
                    dtype=torch.long,
                    device=device,
                )
            )
        return torch.stack(sample_memory_tokens, dim=0), torch.stack(sample_memory_masks, dim=0)


class DeltaStateHistoryMemoryBuilder(DeltaStateMemoryModule):
    """Build fixed-size memory tokens from a global delta-rule state."""

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
        read_with_current_query: bool = True,
    ) -> None:
        if compressor is None:
            from jamel.arch.qwen3vl_compressor.screen_compressor import ScreenCompressor as _ScreenCompressor

            compressor = _ScreenCompressor(
                model_name=compressor_model_name,
                hidden_size=memory_hidden_size,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        super().__init__(
            memory_hidden_size=int(compressor.hidden_size),
            delta_config=delta_config,
            delta_rank=delta_rank,
            delta_memory_slots=delta_memory_slots,
            delta_seed=delta_seed,
            read_with_current_query=read_with_current_query,
        )
        self.history_window = max(1, int(history_window))
        self.max_memory_items = None if max_memory_items is None else max(1, int(max_memory_items))
        self.history_action_prefix = history_action_prefix
        self.cache_history_memory = bool(cache_history_memory)
        object.__setattr__(self, "compressor", compressor)

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

    def build_step_token_rows(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> list[list[torch.Tensor]]:
        return self._prepare_step_tokens(
            batch_size=batch_size,
            history_records=history_records,
        )

    def build_step_token_inputs(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_rows = self.build_step_token_rows(
            batch_size=batch_size,
            history_records=history_records,
        )
        max_rows = max((sum(row.shape[0] for row in rows) for rows in sample_rows), default=0)
        max_rows = min(max_rows, self._history_limit())
        tokens = torch.zeros(
            (batch_size, max_rows, self.memory_hidden_size),
            dtype=torch.float32,
        )
        mask = torch.zeros((batch_size, max_rows), dtype=torch.long)
        if max_rows == 0:
            return tokens, mask
        for sample_idx, rows in enumerate(sample_rows):
            if not rows:
                continue
            sample_tokens = torch.cat(rows, dim=0)[-max_rows:]
            valid_count = min(max_rows, sample_tokens.shape[0])
            tokens[sample_idx, :valid_count] = sample_tokens[:valid_count]
            mask[sample_idx, :valid_count] = 1
        return tokens, mask

    def build_current_query_inputs(
        self,
        *,
        images: Sequence[Any],
        texts: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not images:
            return (
                torch.zeros((0, 1, self.memory_hidden_size), dtype=torch.float32),
                torch.zeros((0, 1), dtype=torch.long),
            )
        query_texts = list(texts) if texts is not None else [CURRENT_MEMORY_QUERY_TEXT] * len(images)
        if len(query_texts) != len(images):
            raise ValueError("current query images and texts must have the same batch size.")
        pil_images = [_to_pil_image(image) for image in images]
        query_tokens = self.compressor.compress_batch(pil_images, query_texts).detach().cpu().float()
        if query_tokens.ndim == 2:
            query_tokens = query_tokens.unsqueeze(1)
        query_mask = torch.ones(query_tokens.shape[:2], dtype=torch.long)
        return query_tokens, query_mask

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
        parameter = self.W_q.weight
        device = parameter.device
        sample_memory_tokens: list[torch.Tensor] = []
        sample_memory_masks: list[torch.Tensor] = []
        for sample_idx, rows in enumerate(sample_rows):
            if not rows:
                sample_memory_tokens.append(
                    torch.zeros(
                        (self.delta_config.memory_slots, self.memory_hidden_size),
                        dtype=torch.float32,
                        device=device,
                    )
                )
                sample_memory_masks.append(
                    torch.zeros(
                        (self.delta_config.memory_slots,),
                        dtype=torch.long,
                        device=device,
                    )
                )
                continue
            tokens = torch.cat(rows, dim=0)
            sample_memory_tokens.append(self._tokens_to_state_memory(tokens))
            sample_memory_masks.append(
                torch.ones(
                    (self.delta_config.memory_slots,),
                    dtype=torch.long,
                    device=device,
                )
            )
        return torch.stack(sample_memory_tokens, dim=0), torch.stack(sample_memory_masks, dim=0)


class HybridHistoryMemoryBuilder:
    """Online DeltaState history plus recent original JAMEL memory tokens.

    Hybrid no longer computes offline DeltaState tokens. It exposes the same
    online history/query tensors as ``online_delta_state`` and additionally
    packs the most recent step embeddings as ordinary JAMEL memory tokens.
    The actor computes the DeltaState read online, then concatenates the recent
    tokens in ``MemoryAugmentedCausalLM._resolve_memory_inputs``.
    """

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
        self.compressor = shared_compressor
        self.memory_hidden_size = int(shared_compressor.hidden_size)
        self.hybrid_recent_items = max(1, int(hybrid_recent_items))

    def build_step_token_rows(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> list[list[torch.Tensor]]:
        return self.delta_builder.build_step_token_rows(
            batch_size=batch_size,
            history_records=history_records,
        )

    def build_step_token_inputs(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.delta_builder.build_step_token_inputs(
            batch_size=batch_size,
            history_records=history_records,
        )

    def build_current_query_inputs(
        self,
        *,
        images: Sequence[Any],
        texts: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.delta_builder.build_current_query_inputs(images=images, texts=texts)

    def build_recent_memory_inputs(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_rows = self.build_step_token_rows(
            batch_size=batch_size,
            history_records=history_records,
        )
        max_rows = self.hybrid_recent_items
        tokens = torch.zeros(
            (batch_size, max_rows, self.memory_hidden_size),
            dtype=torch.float32,
        )
        mask = torch.zeros((batch_size, max_rows), dtype=torch.long)
        for sample_idx, rows in enumerate(sample_rows):
            if not rows:
                continue
            sample_tokens = torch.cat(rows, dim=0)[-max_rows:]
            valid_count = min(max_rows, sample_tokens.shape[0])
            tokens[sample_idx, :valid_count] = sample_tokens[:valid_count]
            mask[sample_idx, :valid_count] = 1
        return tokens, mask

    def build_memory_inputs(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise ValueError(
            "memory_builder='hybrid' requires online DeltaState. Use "
            "build_step_token_inputs/build_current_query_inputs plus "
            "build_recent_memory_inputs, and pass all tensors to the actor."
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
    if normalized == "online_delta_state":
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

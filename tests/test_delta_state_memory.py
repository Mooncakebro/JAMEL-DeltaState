from __future__ import annotations

import torch

from jamel.train.memory.delta_state_encoder import (
    DeltaStateMemoryModule,
    DeltaStateHistoryMemoryBuilder,
    HybridHistoryMemoryBuilder,
    make_history_memory_builder,
)


class FakeCompressor:
    hidden_size = 6

    def compress_batch(self, images, texts):
        rows = []
        for idx, text in enumerate(texts):
            value = float(len(str(text)) + idx + 1)
            rows.append(torch.arange(1, self.hidden_size + 1, dtype=torch.float32) * value)
        return torch.stack(rows, dim=0)


def _record(action: str, scale: float = 1.0):
    token = torch.arange(1, 7, dtype=torch.float32) * scale
    return {"action": action, "_cached_delta_state_input_token": token}


def test_delta_state_builder_empty_history_masks_all_slots():
    builder = DeltaStateHistoryMemoryBuilder(
        compressor_model_name="fake",
        memory_hidden_size=6,
        compressor=FakeCompressor(),
        delta_rank=3,
        delta_memory_slots=4,
    )

    tokens, mask = builder.build_memory_inputs(batch_size=2, history_records=[[], []])

    assert tokens.shape == (2, 4, 6)
    assert mask.shape == (2, 4)
    assert torch.all(mask == 0)
    assert torch.all(tokens == 0)


def test_delta_state_builder_produces_fixed_slots_for_variable_history():
    builder = DeltaStateHistoryMemoryBuilder(
        compressor_model_name="fake",
        memory_hidden_size=6,
        compressor=FakeCompressor(),
        delta_rank=3,
        delta_memory_slots=2,
        delta_seed=7,
    )

    tokens, mask = builder.build_memory_inputs(
        batch_size=2,
        history_records=[
            [_record("click('a')", 1.0)],
            [_record("click('a')", 1.0), _record("scroll(0, 400)", 2.0)],
        ],
    )

    assert tokens.shape == (2, 2, 6)
    assert torch.all(mask == 1)
    assert not torch.allclose(tokens[0], tokens[1])


def test_delta_state_builder_uses_trainable_memory_projections():
    builder = DeltaStateHistoryMemoryBuilder(
        compressor_model_name="fake",
        memory_hidden_size=6,
        compressor=FakeCompressor(),
        delta_rank=3,
        delta_memory_slots=2,
        delta_seed=7,
    )

    parameter_names = {name for name, _ in builder.named_parameters()}

    assert parameter_names == {
        "W_q.weight",
        "W_k.weight",
        "W_v.weight",
        "W_beta.weight",
        "W_beta.bias",
        "slot_queries",
    }
    assert all(parameter.requires_grad for parameter in builder.parameters())

    x = torch.arange(1, 13, dtype=torch.float32).reshape(2, 6)
    q, k, v, beta = builder._project_to_memory(x)

    assert q.shape == (2, 3)
    assert k.shape == (2, 3)
    assert v.shape == (2, 6)
    assert beta.shape == (2, 3)
    assert builder._tokens_to_state_memory(x).shape == (2, 6)


def test_delta_state_online_current_query_trains_w_q():
    module = DeltaStateMemoryModule(
        memory_hidden_size=6,
        delta_rank=3,
        delta_memory_slots=2,
        delta_seed=7,
    )
    history = torch.arange(1, 19, dtype=torch.float32).reshape(1, 3, 6)
    history_mask = torch.ones((1, 3), dtype=torch.long)
    current_query = torch.arange(1, 7, dtype=torch.float32).reshape(1, 1, 6)
    current_query_mask = torch.ones((1, 1), dtype=torch.long)

    tokens, mask = module.tokens_to_state_memory_batch(
        history_tokens=history,
        history_attention_mask=history_mask,
        current_query_tokens=current_query,
        current_query_attention_mask=current_query_mask,
    )
    loss = tokens.pow(2).sum()
    loss.backward()

    assert tokens.shape == (1, 2, 6)
    assert mask.shape == (1, 2)
    assert module.W_q.weight.grad is not None
    assert module.W_q.weight.grad.abs().sum() > 0


def test_factory_rejects_offline_delta_state_name():
    try:
        make_history_memory_builder(
            memory_builder="delta-state",
            compressor_model_name="fake",
            memory_hidden_size=6,
            compressor=FakeCompressor(),
        )
    except ValueError as exc:
        assert "Unsupported memory_builder" in str(exc)
    else:
        raise AssertionError("offline delta_state should not be a public memory builder")


def test_factory_hybrid_is_online_delta_state_plus_recent_tokens():
    builder = make_history_memory_builder(
        memory_builder="hybrid",
        compressor_model_name="fake",
        memory_hidden_size=6,
        compressor=FakeCompressor(),
        delta_rank=3,
        delta_memory_slots=2,
        hybrid_recent_items=1,
    )

    assert isinstance(builder, HybridHistoryMemoryBuilder)

    history_records = [[_record("a", 1.0), _record("b", 2.0)]]
    history_tokens, history_mask = builder.build_step_token_inputs(
        batch_size=1,
        history_records=history_records,
    )
    recent_tokens, recent_mask = builder.build_recent_memory_inputs(
        batch_size=1,
        history_records=history_records,
    )

    assert history_tokens.shape == (1, 2, 6)
    assert torch.all(history_mask == 1)
    assert recent_tokens.shape == (1, 1, 6)
    assert recent_mask.shape == (1, 1)
    assert torch.all(recent_mask == 1)
    assert torch.allclose(recent_tokens[0, 0], torch.arange(1, 7, dtype=torch.float32) * 2.0)


def test_hybrid_prefix_is_online_delta_tokens_plus_recent_tokens():
    module = DeltaStateMemoryModule(
        memory_hidden_size=6,
        delta_rank=3,
        delta_memory_slots=2,
    )
    history = torch.arange(1, 13, dtype=torch.float32).reshape(1, 2, 6)
    history_mask = torch.ones((1, 2), dtype=torch.long)
    current_query = torch.arange(1, 7, dtype=torch.float32).reshape(1, 1, 6)
    current_query_mask = torch.ones((1, 1), dtype=torch.long)
    recent = torch.full((1, 1, 6), 3.0)
    recent_mask = torch.ones((1, 1), dtype=torch.long)

    delta_tokens, delta_mask = module.tokens_to_state_memory_batch(
        history_tokens=history,
        history_attention_mask=history_mask,
        current_query_tokens=current_query,
        current_query_attention_mask=current_query_mask,
    )
    memory_tokens = torch.cat([delta_tokens, recent], dim=1)
    memory_mask = torch.cat([delta_mask, recent_mask], dim=1)

    assert memory_tokens.shape == (1, 3, 6)
    assert memory_mask.shape == (1, 3)
    assert torch.all(memory_mask == 1)
    assert torch.allclose(memory_tokens[:, -1], recent[:, 0])

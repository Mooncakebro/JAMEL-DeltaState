from __future__ import annotations

import torch

from jamel.train.memory.delta_state_encoder import (
    DeltaStateHistoryMemoryBuilder,
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


def test_factory_accepts_delta_state_name():
    builder = make_history_memory_builder(
        memory_builder="delta-state",
        compressor_model_name="fake",
        memory_hidden_size=6,
        compressor=FakeCompressor(),
    )

    assert isinstance(builder, DeltaStateHistoryMemoryBuilder)

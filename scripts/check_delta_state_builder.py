from __future__ import annotations

import torch

from jamel.train.memory.delta_state_encoder import (
    DeltaStateHistoryMemoryBuilder,
    make_history_memory_builder,
)


class FakeCompressor:
    hidden_size = 6

    def compress_batch(self, images, texts):
        raise AssertionError("This check uses cached step tokens and should not compress images.")


def rec(action: str, scale: float):
    token = torch.arange(1, 7, dtype=torch.float32) * scale
    return {"action": action, "_cached_delta_state_input_token": token}


def main() -> None:
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
            [rec("click", 1.0)],
            [rec("click", 1.0), rec("scroll", 2.0)],
        ],
    )
    assert tokens.shape == (2, 2, 6)
    assert torch.all(mask == 1)
    assert not torch.allclose(tokens[0], tokens[1])

    builder = make_history_memory_builder(
        memory_builder="delta-state",
        compressor_model_name="fake",
        memory_hidden_size=6,
        compressor=FakeCompressor(),
    )
    assert isinstance(builder, DeltaStateHistoryMemoryBuilder)
    print("delta_state_builder_ok")


if __name__ == "__main__":
    main()

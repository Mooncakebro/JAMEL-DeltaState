__all__ = [
    "MemoryAugmentedCausalLM",
    "MemoryAugmentedValueModel",
    "MemoryTokenSFTDataset",
    "JAMELMemoryVLTokenSFTDataset",
    "OnlineHistoryMemoryBuilder",
    "DeltaStateHistoryMemoryBuilder",
    "HybridHistoryMemoryBuilder",
    "make_history_memory_builder",
]

_IMPORTS = {
    "MemoryAugmentedCausalLM": (".modeling", "MemoryAugmentedCausalLM"),
    "MemoryAugmentedValueModel": (".modeling", "MemoryAugmentedValueModel"),
    "MemoryTokenSFTDataset": (".sft_dataset", "MemoryTokenSFTDataset"),
    "JAMELMemoryVLTokenSFTDataset": (".jamel_sft_dataset", "JAMELMemoryVLTokenSFTDataset"),
    "OnlineHistoryMemoryBuilder": (".encoder", "OnlineHistoryMemoryBuilder"),
    "DeltaStateHistoryMemoryBuilder": (".delta_state_encoder", "DeltaStateHistoryMemoryBuilder"),
    "HybridHistoryMemoryBuilder": (".delta_state_encoder", "HybridHistoryMemoryBuilder"),
    "make_history_memory_builder": (".delta_state_encoder", "make_history_memory_builder"),
}


def __getattr__(name):
    if name not in _IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

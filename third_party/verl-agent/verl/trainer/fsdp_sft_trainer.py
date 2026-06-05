# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    fsdp2_clip_grad_norm_
)
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.tracking import Tracking
from verl.utils.custom_model import (
    has_custom_model,
    instantiate_custom_model,
    resolve_memory_augment_config,
)
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


if is_cuda_available:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def _multimodal_collate_fn(data_list: list[dict]) -> dict:
    """Collate for batches that may contain a 'multi_modal_inputs' dict.

    pixel_values and image_grid_thw are concatenated (cat) across samples
    rather than stacked, because each image has a different number of patches.
    All other tensors are stacked normally.
    """
    batch: dict = {}
    # Collect all keys
    all_keys = list(data_list[0].keys())

    mm_keys_to_cat = {"pixel_values", "image_grid_thw"}

    for key in all_keys:
        vals = [sample[key] for sample in data_list]
        if key == "multi_modal_inputs":
            # vals is a list of dicts
            combined: dict = {}
            if vals[0]:
                for mm_key in vals[0].keys():
                    tensors = [v[mm_key] for v in vals if isinstance(v.get(mm_key), torch.Tensor)]
                    if tensors:
                        if mm_key in mm_keys_to_cat:
                            combined[mm_key] = torch.cat(tensors, dim=0)
                        else:
                            try:
                                combined[mm_key] = torch.stack(tensors, dim=0)
                            except Exception:
                                combined[mm_key] = torch.cat(tensors, dim=0)
            batch[key] = combined
        elif isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals, dim=0)
        else:
            import numpy as np
            batch[key] = np.array(vals, dtype=object)

    return batch


def _apply_lora_to_custom_model(model: nn.Module, config) -> nn.Module:
    if config.model.get("lora_rank", 0) <= 0:
        return model

    target_model = getattr(model, "llm", None) or getattr(model, "backbone", None)
    if target_model is None:
        target_model = model

    if hasattr(target_model, "enable_input_require_grads"):
        target_model.enable_input_require_grads()

    lora_config = {
        "task_type": TaskType.CAUSAL_LM,
        "r": config.model.lora_rank,
        "lora_alpha": config.model.lora_alpha,
        "target_modules": convert_to_regular_types(config.model.target_modules),
        "bias": "none",
    }
    peft_model = get_peft_model(target_model, LoraConfig(**lora_config))
    if target_model is model:
        return peft_model
    if getattr(model, "llm", None) is target_model:
        model.llm = peft_model
    elif getattr(model, "backbone", None) is target_model:
        model.backbone = peft_model
    return model


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=_multimodal_collate_fn,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=_multimodal_collate_fn,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        custom_model_enabled = has_custom_model(self.config.model.get("custom_cls", None))
        with init_context():
            if custom_model_enabled:
                if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                    raise ValueError("Custom memory-augmented SFT models currently require use_remove_padding=False and ulysses_sequence_parallel_size=1.")
                self.model = instantiate_custom_model(
                    custom_cfg=self.config.model.custom_cls,
                    pretrained_model_name_or_path=local_model_path,
                    model_config=config,
                    torch_dtype=torch.float32,
                    trust_remote_code=trust_remote_code,
                    extra_kwargs={
                        "memory_augment_config": resolve_memory_augment_config(
                            self.config.model.get("memory_augment", None)
                        )
                    },
                )
            else:
                self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch.float32,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                )

                if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                    from verl.models.transformers.monkey_patch import apply_monkey_patch

                    apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

                # Apply Liger kernel if use_liger is enabled
                if self.config.model.get("use_liger", False):
                    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                    _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                if custom_model_enabled:
                    self.model = _apply_lora_to_custom_model(self.model, self.config)
                else:
                    self.model.enable_input_require_grads()
                    lora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "bias": "none",
                    }
                    self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                             cast_forward_inputs=True)

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True, multi_modal_inputs=None):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        memory_tokens = batch.get("memory_tokens")
        memory_attention_mask = batch.get("memory_attention_mask")
        history_memory_tokens = batch.get("history_memory_tokens")
        history_memory_attention_mask = batch.get("history_memory_attention_mask")
        current_memory_query_tokens = batch.get("current_memory_query_tokens")
        current_memory_query_attention_mask = batch.get("current_memory_query_attention_mask")
        # multi_modal_inputs is passed as a kwarg to avoid TensorDict batch-dim validation
        if multi_modal_inputs is None:
            multi_modal_inputs = batch.get("multi_modal_inputs")
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        has_online_delta_state = history_memory_tokens is not None
        if (memory_tokens is not None or has_online_delta_state) and use_sp:
            raise ValueError("memory inputs are not supported with remove padding / sequence parallel SFT.")
        if memory_tokens is not None:
            memory_tokens = memory_tokens.to(self.device_name)
            if memory_attention_mask is not None:
                memory_attention_mask = memory_attention_mask.to(self.device_name)
        if history_memory_tokens is not None:
            history_memory_tokens = history_memory_tokens.to(self.device_name)
            if history_memory_attention_mask is not None:
                history_memory_attention_mask = history_memory_attention_mask.to(self.device_name)
        if current_memory_query_tokens is not None:
            current_memory_query_tokens = current_memory_query_tokens.to(self.device_name)
            if current_memory_query_attention_mask is not None:
                current_memory_query_attention_mask = current_memory_query_attention_mask.to(self.device_name)
        if multi_modal_inputs is not None:
            multi_modal_inputs = {
                key: value.to(self.device_name) if isinstance(value, torch.Tensor) else value
                for key, value in multi_modal_inputs.items()
            }
        else:
            multi_modal_inputs = {}

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    memory_tokens=memory_tokens,
                    memory_attention_mask=memory_attention_mask,
                    history_memory_tokens=history_memory_tokens,
                    history_memory_attention_mask=history_memory_attention_mask,
                    current_memory_query_tokens=current_memory_query_tokens,
                    current_memory_query_attention_mask=current_memory_query_attention_mask,
                    use_cache=False,
                    **multi_modal_inputs,
                )
                logits = output.logits

                # If memory tokens were prepended as a prefix, the logits include
                # extra positions for those prefix tokens.  Strip them so that
                # logits and labels have the same sequence length.
                prefix_len = int(getattr(output, "memory_prefix_length", 0) or 0)
                if prefix_len <= 0 and memory_tokens is not None:
                    prefix_len = memory_tokens.shape[1]
                if prefix_len > 0:
                    logits = logits[:, prefix_len:, :]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                # Use logits last dim as vocab_size (works for any wrapped model)
                _vocab_size = shift_logits.shape[-1]
                shift_logits = shift_logits.view(-1, _vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict, multi_modal_inputs=None):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for i, micro_batch in enumerate(micro_batches):
            # Slice multi_modal_inputs per micro-batch if present.
            # pixel_values and image_grid_thw are concatenated across samples by
            # the collate fn, so when splitting into micro-batches we must slice
            # them to match the current micro-batch's samples.
            mm = multi_modal_inputs
            if mm is not None and n_micro_batches > 1 and "pixel_values" in mm:
                micro_bs = self.config.data.micro_batch_size_per_gpu
                start_sample = i * micro_bs
                end_sample = start_sample + micro_bs
                ig = mm.get("image_grid_thw")
                if ig is not None:
                    patches_per_sample = (ig[:, 0] * ig[:, 1] * ig[:, 2]).tolist()
                    cumsum = [0]
                    for p in patches_per_sample:
                        cumsum.append(cumsum[-1] + p)
                    pv_start = cumsum[start_sample]
                    pv_end = cumsum[end_sample]
                    mm = {
                        **mm,
                        "pixel_values": mm["pixel_values"][pv_start:pv_end],
                        "image_grid_thw": ig[start_sample:end_sample],
                    }
            loss = self._compute_loss_and_backward(batch=micro_batch, multi_modal_inputs=mm) / n_micro_batches
            step_loss += loss.item()

        if self.config.model.strategy == 'fsdp':
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == 'fsdp2':
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.ulysses_device_mesh.size(0)
        grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        return {
            'train/loss': step_loss.detach().item(),
            'train/lr(1e-3)': lr * 1e3,
            'train/grad_norm': grad_norm_val,
        }

    def validation_step(self, batch: TensorDict, multi_modal_inputs=None):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False, multi_modal_inputs=multi_modal_inputs)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.ulysses_device_mesh.size(0)
        return loss

    def _save_processor_if_available(self, path):
        """Persist the multimodal processor alongside the model so eval can
        load the checkpoint standalone. VL models need preprocessor_config.json
        which the tokenizer alone does not provide. We resolve the base model
        from model_config._name_or_path or partial_pretrain and copy its
        processor files."""
        try:
            from transformers import AutoProcessor
        except Exception:
            return
        base = (getattr(self.model_config, "_name_or_path", None)
                or getattr(self.config.model, "partial_pretrain", None))
        if not base:
            return
        try:
            processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
            processor.save_pretrained(path)
        except Exception as e:
            print(f"[ckpt] WARN: failed to save processor from {base}: {e}")

    def save_checkpoint(self, step):
        # save checkpoint
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            # FSDP1 checkpoint saving
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)
                self._save_processor_if_available(path)
        elif fsdp_strategy == "fsdp2":
            # FSDP2 checkpoint saving
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            # Get full state dict with FSDP2; cast to bfloat16 to halve disk usage (~15GB vs ~31GB)
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)
            state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            # save huggingface model sharded (max_shard_size="5GB" keeps each shard manageable)
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict, max_shard_size="5GB")
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
                self._save_processor_if_available(path)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        # Rotate old checkpoints, keeping only the most recent `save_total_limit` ones.
        save_total_limit = getattr(self.config.trainer, 'save_total_limit', 0)
        if self.device_mesh.get_rank() == 0 and save_total_limit and save_total_limit > 0:
            import re
            import shutil
            base = self.config.trainer.default_local_dir
            existing = []
            for name in os.listdir(base):
                m = re.fullmatch(r"global_step_(\d+)", name)
                full = os.path.join(base, name)
                if m and os.path.isdir(full):
                    existing.append((int(m.group(1)), full))
            existing.sort(key=lambda x: x[0])
            for _, old_path in existing[:-save_total_limit]:
                shutil.rmtree(old_path, ignore_errors=True)

        torch.distributed.barrier()
        return path

    def package_final_model(self, checkpoint_path):
        output_model_path = (
            os.environ.get("OUTPUT_MODEL_PATH")
            or getattr(self.config.trainer, "output_model_path", None)
        )
        if not output_model_path:
            return

        compressor_model = (
            os.environ.get("COMPRESSOR_MODEL")
            or getattr(self.config.trainer, "compressor_model", None)
        )
        if not compressor_model:
            raise RuntimeError(
                "OUTPUT_MODEL_PATH was set, but no COMPRESSOR_MODEL was provided. "
                "Set COMPRESSOR_MODEL to the local Qwen3-VL compressor model directory."
            )

        if self.device_mesh.get_rank() != 0:
            return

        from jamel.train.memory.package_model import package_jamel_model

        output = package_jamel_model(
            checkpoint=checkpoint_path,
            compressor_model=compressor_model,
            output_model_path=output_model_path,
        )
        print(f"[jamel] Packaged final model: {output}")

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.

        # val_steps: run validation every N steps (in addition to end-of-epoch).
        # Configured via trainer.val_steps (default 0 = epoch-only).
        val_steps = getattr(self.config.trainer, 'val_steps', 0)

        def _run_validation(epoch_num):
            val_losses = []
            for val_data in self.val_dataloader:
                _val_mm = val_data.pop("multi_modal_inputs", None)
                val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(self.device_name)
                val_losses.append(self.validation_step(val_data, multi_modal_inputs=_val_mm))
            if rank == 0:
                avg = torch.mean(torch.stack(val_losses))
                tracking.log(data={
                    "val/loss": avg.detach().item(),
                    "val/epoch": epoch_num,
                    "val/global_step": global_step,
                }, step=global_step)
            torch.distributed.barrier()

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            epoch_step = 0
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0
            ):
                global_step += 1
                epoch_step += 1
                _mm_inputs = data.pop("multi_modal_inputs", None)
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data, multi_modal_inputs=_mm_inputs)
                if rank == 0:
                    metric['train/epoch'] = epoch + epoch_step / self.steps_per_epoch
                    metric['train/epoch_step'] = epoch_step
                    metric['train/global_step'] = global_step
                    tracking.log(data=metric, step=global_step)

                # intra-epoch validation every val_steps steps
                if val_steps > 0 and epoch_step % val_steps == 0:
                    _run_validation(epoch + epoch_step / self.steps_per_epoch)

                # early exit
                if global_step >= self.total_training_steps:
                    _run_validation(epoch + 1)
                    checkpoint_path = self.save_checkpoint(step=global_step)
                    self.package_final_model(checkpoint_path)
                    return

            # end-of-epoch validation + checkpoint
            _run_validation(epoch + 1)
            checkpoint_path = self.save_checkpoint(step=global_step)
            if epoch + 1 >= self.config.trainer.total_epochs:
                self.package_final_model(checkpoint_path)


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

    trainer.fit()


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()

import torch
from contextlib import contextmanager
from torch import nn
from pathlib import Path
from typing import Optional, Any, Dict
import copy
from vllm.tokenformer.tokenformer_surgeon import (
    TokenformerSurgeon,
)
from vllm.model_executor.models import SupportsLoRA, supports_tokenformer
from vllm.lora.utils import get_adapter_absolute_path, get_lora_id
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

import os

logger = init_logger(__name__)

# Weights that vLLM shards on dim 0 (output_dim) across TP ranks
_TP_SHARD_DIM0 = ("q_proj", "k_proj", "v_proj", "embed_tokens", "lm_head", "gate_proj", "up_proj")
# Weights that vLLM shards on dim 1 (input_dim) across TP ranks
_TP_SHARD_DIM1 = ("o_proj", "down_proj")


def _shard_for_tp(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Shard a full unsharded tensor for the current TP rank."""
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return tensor

    if any(k in key for k in _TP_SHARD_DIM0):
        shard_size = tensor.shape[0] // tp_size
        start = tp_rank * shard_size
        return tensor[start:start + shard_size, :]

    if any(k in key for k in _TP_SHARD_DIM1):
        shard_size = tensor.shape[1] // tp_size
        start = tp_rank * shard_size
        return tensor[:, start:start + shard_size]

    # Replicated weight (norms, tokenformer params, etc.) — no sharding needed
    return tensor


class TokenformerModel:
    """A tokenformer pre-trained model."""

    def __init__(self, tokenformers: Dict[str, torch.Tensor]) -> None:
        self.id = get_lora_id()
        self.tokenformers = tokenformers

    @classmethod
    def from_local_checkpoint(
        cls, model_dir: str, device: torch.device
    ) -> "TokenformerModel":
        files = list(Path(model_dir).glob("*.pt"))

        if len(files) == 0:
            raise FileNotFoundError(f"No .pt file found in {model_dir}")

        checkpoint_file = files[0]

        tokenformers = {}
        state_dict = torch.load(checkpoint_file, map_location=device)
        module_state_dict = state_dict['model_state_dict']
        for module, tensor in module_state_dict.items():
            logger.info(f"Loading {module} from {checkpoint_file}")
            tokenformers[module] = tensor.to(device)

        return cls(tokenformers)


class TokenformerModelManager:
    """A manager that manages tokenformer models."""

    def __init__(
        self,
        model: SupportsLoRA,
        device: torch.device,
    ):
        if supports_tokenformer(model):
            self.model = TokenformerSurgeon(model, device).insert_adapter_modules()
        else:
            self.model = model

        self._registered_adapters: Dict[int, Any] = {}
        self._active_adapter: Any = None
        self.tokenformer_model_cls = TokenformerModel
        self.dtype = next(self.model.parameters()).dtype
        self.device = device
        # Stores the original (already TP-sharded) parameter data before an
        # adapter overwrites it, so we can restore it on deactivation without
        # going through load_weights again.
        self.original_tensors = {}
        self._lru_adaptor_ids = []

    def activate_adapter(self, adapter_id: int) -> bool:
        assert adapter_id in self._registered_adapters, f"Adapter {adapter_id} not found"

        if adapter_id == self._active_adapter:
            logger.info(f"Tokenformer {adapter_id} is already active")
            return False

        self.update_lru_position(adapter_id)
        logger.info(f"Activating Tokenformer - {adapter_id}")

        tokenformers = self._registered_adapters[adapter_id].tokenformers

        # Save the current (sharded) param values before we overwrite them,
        # so _deactivate_adapter can restore them directly without re-sharding.
        for name, param in self.model.named_parameters():
            if name in tokenformers and name not in self.original_tensors:
                logger.info(f"Saving original tensor {name} before loading adapter {adapter_id}")
                self.original_tensors[name] = param.data.clone()

        # Build a full model state dict, then overwrite only the adapter keys
        # after sharding them to match the current TP rank.
        model_state_dict = self.model.state_dict()

        for key, value in tokenformers.items():
            if 'lora' in key:
                continue
            logger.info(f"Loading {key} from adapter {adapter_id}")
            model_state_dict[key] = _shard_for_tp(key, value)

        self.model.load_weights(model_state_dict.items())
        process_weights_after_loading(self.model, self.model.model_config, self.device)

        self._active_adapter = adapter_id
        return True

    def _deactivate_adapter(self, adapter_id: int):
        logger.info(f"Deactivating Tokenformer - {adapter_id}")

        tokenformers = self._registered_adapters[adapter_id].tokenformers

        # Restore original param data directly — these are already TP-sharded
        # so we bypass load_weights entirely to avoid double-sharding.
        for name, param in self.model.named_parameters():
            if name in tokenformers:
                if "tokenformer_p" in name:
                    nn.init.zeros_(param)
                elif name in self.original_tensors:
                    logger.info(f"Restoring original tensor {name}")
                    param.data.copy_(self.original_tensors[name])

        process_weights_after_loading(self.model, self.model.model_config, self.device)
        self._active_adapter = None

    def update_lru_position(self, adapter_id: int) -> None:
        if adapter_id in self._lru_adaptor_ids:
            self._lru_adaptor_ids.remove(adapter_id)
        self._lru_adaptor_ids.append(adapter_id)

    def deactivate_adapter(self, adapter_id: int) -> bool:
        return self._deactivate_adapter(adapter_id)

    def add_adapter(self, request) -> bool:
        lora_path = get_adapter_absolute_path(request.lora_path)
        tokenformer = self.tokenformer_model_cls.from_local_checkpoint(
            lora_path, device=self.device
        )

        if len(self._registered_adapters) >= self.capacity:
            lru_adapter_id = self._lru_adaptor_ids.pop(0)
            self.remove_adapter(lru_adapter_id)

        self._registered_adapters[request.adapter_id] = tokenformer
        self._lru_adaptor_ids.append(request.adapter_id)

        logger.info(f"Adapter {request.adapter_id} added")
        return True

    def supports_tower_connector_lora(self):
        return False

    def set_active_adapters(self, lora_requests, lora_mapping):
        if len(lora_requests) == 0:
            self.deactivate_all_adapters()
        else:
            for request in lora_requests:
                self.activate_adapter(request.adapter_id)

    def set_adapter_mapping(self, mapping: Any) -> None:
        pass

    def remove_adapter(self, adapter_id: int) -> bool:
        return remove_adapter(
            adapter_id, self._registered_adapters, self._remove_adapter
        )

    def _remove_adapter(self, adapter_id: int) -> None:
        if adapter_id not in self._registered_adapters:
            logger.warning(f"Adapter {adapter_id} not found")
            return

        if adapter_id == self._active_adapter:
            self.deactivate_adapter(adapter_id)

        del self._registered_adapters[adapter_id]
        logger.info(f"Adapter {adapter_id} removed")

    def deactivate_all_adapters(self) -> None:
        if self._active_adapter is not None:
            self.deactivate_adapter(self._active_adapter)
        self._active_adapter = None

    def remove_all_adapters(self) -> None:
        for id in self._registered_adapters:
            self.deactivate_adapter(id)
        self._registered_adapters.clear()
        self._active_adapter = None

    def get_adapter(self, adapter_id: int) -> Optional[Any]:
        get_adapter(adapter_id, self._registered_adapters)

    def list_adapters(self) -> Dict[int, Any]:
        return list_adapters(self._registered_adapters)

    def pin_adapter(self, adapter_id: int) -> bool:
        pass

    @property
    def capacity(self) -> int:
        return int(os.getenv("TOKENFORMER_CACHE_CAPACITY", "4"))

    @property
    def adapter_slots(self) -> int:
        pass

    @contextmanager
    def dummy_lora_cache(self):
        yield

    def add_dummy_lora(self, lora_request, rank: int = 8):
        logger.debug(f"Adding dummy LoRA {lora_request.lora_name} with rank {rank} (no-op for tokenformer)")
        pass


def add_adapter(adapter: Any, registered_adapters: dict[int, Any],
                capacity: int, add_func: callable) -> bool:
    if adapter.id not in registered_adapters:
        if len(registered_adapters) >= capacity:
            raise RuntimeError('No free adapter slots.')
        add_func(adapter)
        registered_adapters[adapter.id] = adapter
        return True
    return False


def deactivate_adapter(adapter_id: int, active_adapters: dict[int, None],
                       deactivate_func: callable) -> bool:
    if adapter_id in active_adapters:
        deactivate_func(adapter_id)
        active_adapters.pop(adapter_id)
        return True
    return False


def remove_adapter(adapter_id: int, registered_adapters: dict[int, Any],
                   deactivate_func: callable) -> bool:
    deactivate_func(adapter_id)
    return bool(registered_adapters.pop(adapter_id, None))


def list_adapters(registered_adapters: dict[int, Any]) -> dict[int, Any]:
    return dict(registered_adapters)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-0 symbol-drift assertion.

The fork's `.pt`/Tokenformer adapter layer depends on a set of core-vLLM
symbols (ADR 0005; migration plan Phase 0). Every one exists today; this test
imports each and asserts presence + the specific attribute/signature the adapter
layer relies on. Green now — it turns a future rebase's silent "AttributeError at
model load" into a red unit test the moment a bump removes or reshapes one.

Pure imports + introspection: no model, no torch tensors, no GPU. Milliseconds.
"""

from __future__ import annotations

import inspect

import pytest


def test_lora_utils_symbols():
    from vllm.lora.utils import get_adapter_absolute_path, get_lora_id
    assert callable(get_adapter_absolute_path)
    assert callable(get_lora_id)


def test_worker_manager_symbol():
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    assert inspect.isclass(LRUCacheWorkerLoRAManager)


def test_lora_model_symbol():
    from vllm.lora.lora_model import LoRAModel
    assert inspect.isclass(LoRAModel)
    # The fork loads adapters via this constructor path.
    assert hasattr(LoRAModel, "from_lora_tensors")


def test_peft_helper_fields():
    from vllm.lora.peft_helper import PEFTHelper
    # The fork reads these off .pt metadata; a rename here breaks alpha/rslora
    # handling silently.
    for field in ("r", "lora_alpha", "use_rslora", "vllm_lora_scaling_factor"):
        assert field in PEFTHelper.__dataclass_fields__, (
            f"PEFTHelper lost field '{field}'")


def test_process_weights_after_loading_symbol():
    from vllm.model_executor.model_loader.utils import (
        process_weights_after_loading)
    assert callable(process_weights_after_loading)


def test_supports_lora_interface():
    from vllm.model_executor.models import SupportsLoRA
    assert inspect.isclass(SupportsLoRA)


def test_initialize_model_symbol():
    from vllm.model_executor.model_loader.utils import initialize_model
    sig = inspect.signature(initialize_model)
    assert "vllm_config" in sig.parameters


def test_engine_args_create_engine_config():
    from vllm.engine.arg_utils import EngineArgs
    assert hasattr(EngineArgs, "create_engine_config")


def test_adapter_layer_entrypoints():
    # The fork's own two-pass normalization surface the harness drives.
    from vllm.tokenformer.adapter_format import normalize_lora_key
    from vllm.tokenformer.hybrid_adapter_manager import PTWorkerLoRAManager
    assert callable(normalize_lora_key)
    assert hasattr(PTWorkerLoRAManager, "_renormalize_lora_sd_for_model")
    assert hasattr(PTWorkerLoRAManager, "_detect_model_layers_prefix")

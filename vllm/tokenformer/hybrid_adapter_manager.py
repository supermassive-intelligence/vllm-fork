# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HybridAdapterManager — dispatches adapter operations across a
Tokenformer sub-manager and a LoRA sub-manager.

Current state (phase 2 skeleton): the Tokenformer sub-manager is real;
the LoRA sub-manager is a placeholder. `add_adapter` classifies the
incoming `.pt` file and, if it's pure Tokenformer, delegates. Pure-LoRA
and hybrid adapters raise NotImplementedError until the LoRA-from-.pt
loader lands (option C in the rollout plan).

Once option C is in, this class will:
 1. Split the loaded state dict via split_adapter_state_dict.
 2. Register the Tokenformer tensors with TokenformerModelManager.
 3. Register the LoRA tensors with the LoRA worker manager.
 4. At `set_active_adapters` time, fan out to both sub-managers.

See `docs/design/hybrid_lora_tokenformer.md`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.tokenformer.adapter_format import AdapterKind, load_adapter_from_pt
from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = init_logger(__name__)


class HybridAdapterManager:
    """Manager that composes a Tokenformer sub-manager and a LoRA
    sub-manager behind the same interface the runner mixin expects.

    Phase 2 skeleton: only the Tokenformer half is wired. LoRA/hybrid
    adapters raise NotImplementedError until the LoRA-from-.pt loader
    is in place.
    """

    def __init__(
        self,
        model: "nn.Module",
        device: "torch.device",
    ):
        self._tokenformer = TokenformerModelManager(model=model, device=device)
        # LoRA sub-manager placeholder. Will be instantiated alongside the
        # Tokenformer one once the LoRA-from-.pt loader (option C) lands.
        self._lora: Any = None
        # adapter_id -> AdapterKind, so remove/activate can route correctly.
        self._kinds: dict[int, AdapterKind] = {}

    # --- model handle exposed to the runner -----------------------------

    @property
    def model(self) -> "nn.Module":
        # Today this is the Tokenformer-wrapped model. When the LoRA
        # sub-manager arrives, order will be: apply LoRA layer
        # replacement first, then Tokenformer surgeon on top — so this
        # property will return the model after both passes. The
        # Tokenformer sub-manager already holds a reference to the
        # post-surgeon model, so reading from it Just Works.
        return self._tokenformer.model

    # --- adapter lifecycle ---------------------------------------------

    def add_adapter(self, lora_request) -> bool:
        loaded = load_adapter_from_pt(lora_request.lora_path)
        kind = loaded.kind
        self._kinds[lora_request.adapter_id] = kind

        if kind == "tokenformer":
            # TokenformerModelManager.add_adapter re-loads the file
            # itself. That's a redundant torch.load call since we just
            # did one in load_adapter_from_pt, but the correctness
            # story is clean and phase 2 isn't perf-sensitive. We'll
            # tighten this to pass the pre-loaded dict through in a
            # later step.
            return self._tokenformer.add_adapter(lora_request)

        raise NotImplementedError(
            f"Adapter {lora_request.adapter_id} at {loaded.source_path} "
            f"has kind '{kind}', but LoRA and hybrid handling are not "
            f"wired yet. See docs/design/hybrid_lora_tokenformer.md."
        )

    def remove_adapter(self, adapter_id: int) -> bool:
        kind = self._kinds.pop(adapter_id, "tokenformer")
        if kind in ("tokenformer", "hybrid"):
            self._tokenformer.remove_adapter(adapter_id)
        if kind in ("lora", "hybrid") and self._lora is not None:
            self._lora.remove_adapter(adapter_id)
        return True

    def remove_all_adapters(self) -> None:
        self._kinds.clear()
        self._tokenformer.remove_all_adapters()
        if self._lora is not None:
            self._lora.remove_all_adapters()

    def pin_adapter(self, adapter_id: int) -> bool:
        # TokenformerModelManager.pin_adapter is a no-op pass today,
        # so pinning is a no-op regardless of kind. When the LoRA
        # sub-manager lands, route LoRA ids to its pin.
        return True

    def list_adapters(self) -> Any:
        return self._tokenformer.list_adapters()

    # --- per-step activation -------------------------------------------

    def set_active_adapters(self, lora_requests, lora_mapping) -> None:
        # Phase 2 skeleton: every registered id is Tokenformer, so
        # forward wholesale. When LoRA ids coexist, we'll partition
        # `lora_requests` by `self._kinds[id]` and route each half.
        self._tokenformer.set_active_adapters(lora_requests, lora_mapping)

    def activate_adapter(self, adapter_id: int) -> bool:
        kind = self._kinds.get(adapter_id, "tokenformer")
        if kind in ("tokenformer", "hybrid"):
            self._tokenformer.activate_adapter(adapter_id)
        if kind in ("lora", "hybrid") and self._lora is not None:
            self._lora.activate_adapter(adapter_id)
        return True

    def deactivate_adapter(self, adapter_id: int) -> bool:
        kind = self._kinds.get(adapter_id, "tokenformer")
        if kind in ("tokenformer", "hybrid"):
            self._tokenformer.deactivate_adapter(adapter_id)
        if kind in ("lora", "hybrid") and self._lora is not None:
            self._lora.deactivate_adapter(adapter_id)
        return True

    # --- warmup plumbing ------------------------------------------------

    @contextmanager
    def dummy_lora_cache(self):
        with self._tokenformer.dummy_lora_cache():
            if self._lora is not None:
                with self._lora.dummy_lora_cache():
                    yield
            else:
                yield

    def add_dummy_lora(self, lora_request, rank: int = 8) -> bool:
        # Tokenformer's version is a no-op that exists purely to satisfy
        # the warmup path. Route dummies to Tokenformer today; once
        # LoRA is wired, dummies representing LoRA warmup slots will go
        # to the LoRA sub-manager (its add_dummy_lora actually
        # registers a rank-N zero adapter at the given slot).
        self._tokenformer.add_dummy_lora(lora_request, rank=rank)
        return True

    # --- misc -----------------------------------------------------------

    def supports_tower_connector_lora(self) -> bool:
        return False

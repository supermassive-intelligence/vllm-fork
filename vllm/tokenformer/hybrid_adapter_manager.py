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
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.tokenformer.adapter_format import (
    AdapterKind,
    load_adapter_from_pt,
)
from vllm.tokenformer.lora_from_pt import load_lora_model_from_pt
from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from vllm.config import VllmConfig
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.request import LoRARequest

logger = init_logger(__name__)


class PTWorkerLoRAManager(LRUCacheWorkerLoRAManager):
    """LoRA worker manager that loads adapters from ScalarLM's `.pt`
    format instead of HF PEFT's `adapter_config.json + safetensors`.

    Overrides `_load_adapter` to pull the LoRA half out of a `.pt`
    state dict via `load_adapter_from_pt` + `load_lora_model_from_pt`.
    Everything else — slot management, kernel setup, dummy-lora cache
    — is inherited unchanged.
    """

    def _load_adapter(self, lora_request: "LoRARequest") -> "LoRAModel":
        loaded = load_adapter_from_pt(lora_request.lora_path)
        if not loaded.lora_sd:
            raise ValueError(
                f"Adapter at {loaded.source_path} has no LoRA tensors "
                f"but was routed to the LoRA sub-manager. Check the "
                f"HybridAdapterManager classification step."
            )
        return load_lora_model_from_pt(
            loaded.lora_sd,
            lora_model_id=lora_request.adapter_id,
            device=self.device,
            dtype=(
                self.lora_config.lora_dtype
                if self.lora_config is not None
                else None
            ),
            model_vocab_size=self.vocab_size,
        )


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
        vllm_config: "VllmConfig | None" = None,
    ):
        """Instantiate both sub-managers.

        Load order is LoRA layer replacement first, then the Tokenformer
        surgeon. This yields the composition:

            base(x) + lora_delta(x)  (inside the MLP block)
            + tokenformer_delta(x)   (added by the surgeon wrapper)

        When `vllm_config` is None (or `vllm_config.lora_config` is
        None), the LoRA sub-manager is not instantiated and the
        hybrid manager behaves like a pure Tokenformer manager — this
        keeps the skeleton path alive for callers that haven't flipped
        to hybrid yet.
        """
        lora_enabled = (
            vllm_config is not None and vllm_config.lora_config is not None
        )

        if lora_enabled:
            # LoRA sub-manager replaces targeted linears with *WithLoRA
            # wrappers, returning the transformed model. We then feed
            # that into the Tokenformer surgeon.
            self._lora: Any = PTWorkerLoRAManager(
                vllm_config,
                device,
                model.embedding_modules,
            )
            model = self._lora.create_lora_manager(model, vllm_config)
        else:
            self._lora = None

        self._tokenformer = TokenformerModelManager(model=model, device=device)
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
        """Classify the adapter, then route each half to its sub-manager.

        A hybrid adapter's Tokenformer tensors go to the Tokenformer
        manager; its LoRA tensors go to the LoRA manager. Both halves
        share the same adapter id. `activate`/`set_active_adapters` use
        `self._kinds` to fan out correctly.

        The sub-managers each re-load the `.pt` file from disk. This is
        redundant (we already classified once) but keeps their existing
        interfaces intact. Phase 2 isn't perf-sensitive; we'll tighten
        this in a later step by passing pre-split dicts through.
        """
        loaded = load_adapter_from_pt(lora_request.lora_path)
        kind = loaded.kind
        self._kinds[lora_request.adapter_id] = kind

        # Tokenformer half.
        if kind in ("tokenformer", "hybrid"):
            self._tokenformer.add_adapter(lora_request)

        # LoRA half.
        if kind in ("lora", "hybrid"):
            if self._lora is None:
                raise RuntimeError(
                    f"Adapter {lora_request.adapter_id} at "
                    f"{loaded.source_path} contains LoRA tensors, but the "
                    f"HybridAdapterManager was constructed without a "
                    f"LoRA-enabled vllm_config. Pass "
                    f"--enable-lora alongside --enable-tokenformer."
                )
            self._lora.add_adapter(lora_request)

        return True

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
        """Fan out to both sub-managers.

        Both managers see the full request set so hybrid adapters
        (whose id is in both managers' registries) are activated in both
        places. Each sub-manager is expected to skip ids it doesn't own
        — Tokenformer uses the skip-unregistered guard we added in
        6529423ba, and the LRU LoRA manager already no-ops on unknown
        ids via `list_adapters` membership checks.
        """
        self._tokenformer.set_active_adapters(lora_requests, lora_mapping)
        if self._lora is not None:
            self._lora.set_active_adapters(lora_requests, lora_mapping)

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

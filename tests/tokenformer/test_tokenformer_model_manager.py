# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TokenformerModelManager adapter-lifecycle state.

Uses a bare nn.Linear as the model — it doesn't declare
SupportsTokenformer, so __init__ skips the surgeon and none of the
tests touch real weight-reload machinery.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from vllm.tokenformer.tokenformer_model_manager import (  # noqa: E402
    TokenformerModelManager,
)


@pytest.fixture
def manager():
    return TokenformerModelManager(
        model=nn.Linear(2, 2), device=torch.device("cpu")
    )


def test_get_dummy_lora_warmup_rank_returns_default(manager):
    # The warmup path in LoRAModelRunnerMixin.maybe_setup_dummy_loras
    # calls this on whatever manager is active; tokenformer must accept
    # it and leave the rank unchanged (its dummy loras are no-ops).
    assert manager.get_dummy_lora_warmup_rank(8) == 8
    assert manager.get_dummy_lora_warmup_rank(4) == 4


def test_remove_active_adapter_clears_active_marker(manager, monkeypatch):
    manager._registered_adapters[7] = SimpleNamespace(tokenformers={})
    manager._active_adapter = 7
    monkeypatch.setattr(manager, "deactivate_adapter", lambda _id: True)

    assert manager.remove_adapter(7) is True
    assert 7 not in manager._registered_adapters
    assert manager._active_adapter is None


def test_deactivate_all_after_remove_does_not_raise(manager, monkeypatch):
    """Regression: _remove_adapter used to leave _active_adapter set,
    so the next empty-batch deactivate_all_adapters() looked up the
    removed id in _registered_adapters and crashed with KeyError."""
    manager._registered_adapters[7] = SimpleNamespace(tokenformers={})
    manager._active_adapter = 7
    monkeypatch.setattr(manager, "deactivate_adapter", lambda _id: True)
    manager.remove_adapter(7)

    # Simulates set_active_adapters(set()) on the next lora-free batch.
    manager.deactivate_all_adapters()
    assert manager._active_adapter is None


def test_remove_unknown_adapter_returns_false(manager):
    assert manager.remove_adapter(99) is False


def test_list_adapters_returns_id_set(manager):
    # The worker-manager contract is set[int]; adapter objects (which
    # hold tensors) must not leak into the engine's list_loras RPC.
    manager._registered_adapters[3] = SimpleNamespace(tokenformers={})
    manager._registered_adapters[9] = SimpleNamespace(tokenformers={})
    assert manager.list_adapters() == {3, 9}


def test_pin_adapter_reports_unsupported(manager):
    assert manager.pin_adapter(1) is False


def test_adapter_slots_matches_capacity(manager):
    assert manager.adapter_slots == manager.capacity


def test_remove_adapter_syncs_lru_accounting(manager, monkeypatch):
    """External removal used to leave the id in _lru_adaptor_ids, so a
    later capacity eviction popped the stale id and evicted nothing."""
    manager._registered_adapters[7] = SimpleNamespace(tokenformers={})
    manager._lru_adaptor_ids.append(7)
    monkeypatch.setattr(manager, "deactivate_adapter", lambda _id: True)
    manager.remove_adapter(7)
    assert manager._lru_adaptor_ids == []


def test_remove_all_only_deactivates_active(manager, monkeypatch):
    """Only the active adapter has weights applied; remove_all used to
    run a full deactivation (model-sized weight reload) per registered
    adapter."""
    calls = []
    monkeypatch.setattr(
        manager, "deactivate_adapter", lambda _id: calls.append(_id)
    )
    manager._registered_adapters[1] = SimpleNamespace(tokenformers={})
    manager._registered_adapters[2] = SimpleNamespace(tokenformers={})
    manager._lru_adaptor_ids.extend([1, 2])
    manager._active_adapter = 2

    manager.remove_all_adapters()

    assert calls == [2]
    assert manager._registered_adapters == {}
    assert manager._lru_adaptor_ids == []
    assert manager._active_adapter is None

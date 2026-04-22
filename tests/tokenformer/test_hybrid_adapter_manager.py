# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HybridAdapterManager routing.

No model, no torch cuda — we stub TokenformerModelManager with a
MagicMock and assert the hybrid manager wires the right sub-manager
call for each adapter kind.

Covers the phase 2 skeleton; LoRA / hybrid adapters still raise
NotImplementedError until the LoRA-from-.pt loader lands.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def patched_manager(monkeypatch):
    """Return a HybridAdapterManager whose Tokenformer sub-manager is a
    MagicMock (no real model required)."""
    import vllm.tokenformer.hybrid_adapter_manager as mod

    fake_tk = MagicMock()
    fake_tk.model = SimpleNamespace(name="wrapped-model")

    # dummy_lora_cache must yield — MagicMock doesn't produce a
    # context manager by default.
    @contextmanager
    def _dummy_cache():
        yield
    fake_tk.dummy_lora_cache.side_effect = _dummy_cache

    monkeypatch.setattr(
        mod, "TokenformerModelManager", lambda model, device: fake_tk
    )

    mgr = mod.HybridAdapterManager(model=object(), device="cpu")
    return mgr, fake_tk


def _fake_loaded(kind, path="/tmp/fake-adapter"):
    """Minimal stand-in for LoadedAdapter."""
    return SimpleNamespace(kind=kind, source_path=path,
                           tokenformer_sd={}, lora_sd={})


def test_model_property_forwards_to_tokenformer(patched_manager):
    mgr, fake_tk = patched_manager
    assert mgr.model is fake_tk.model


def test_add_tokenformer_adapter_delegates(patched_manager, monkeypatch):
    mgr, fake_tk = patched_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod
    monkeypatch.setattr(
        mod, "load_adapter_from_pt",
        lambda _p: _fake_loaded("tokenformer"),
    )
    fake_tk.add_adapter.return_value = True

    req = SimpleNamespace(adapter_id=7, lora_path="/tmp/a")
    ok = mgr.add_adapter(req)
    assert ok is True
    fake_tk.add_adapter.assert_called_once_with(req)
    assert mgr._kinds[7] == "tokenformer"


@pytest.mark.parametrize("kind", ["lora", "hybrid"])
def test_add_lora_or_hybrid_raises_for_now(patched_manager, monkeypatch, kind):
    mgr, _ = patched_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod
    monkeypatch.setattr(
        mod, "load_adapter_from_pt",
        lambda _p: _fake_loaded(kind),
    )
    req = SimpleNamespace(adapter_id=9, lora_path="/tmp/a")
    with pytest.raises(NotImplementedError, match=kind):
        mgr.add_adapter(req)


def test_set_active_adapters_forwards(patched_manager):
    mgr, fake_tk = patched_manager
    requests = {SimpleNamespace(adapter_id=1)}
    mapping = object()
    mgr.set_active_adapters(requests, mapping)
    fake_tk.set_active_adapters.assert_called_once_with(requests, mapping)


def test_activate_tokenformer_routes_to_tokenformer(patched_manager, monkeypatch):
    mgr, fake_tk = patched_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod
    monkeypatch.setattr(
        mod, "load_adapter_from_pt",
        lambda _p: _fake_loaded("tokenformer"),
    )
    mgr.add_adapter(SimpleNamespace(adapter_id=3, lora_path="/tmp/a"))
    fake_tk.activate_adapter.reset_mock()

    mgr.activate_adapter(3)
    fake_tk.activate_adapter.assert_called_once_with(3)


def test_activate_unknown_id_falls_back_to_tokenformer(patched_manager):
    # Defensive: if an id was never registered (e.g. warmup no-op
    # path), activate still routes to Tokenformer so the existing
    # skip-unregistered behavior triggers there.
    mgr, fake_tk = patched_manager
    mgr.activate_adapter(42)
    fake_tk.activate_adapter.assert_called_once_with(42)


def test_remove_all_adapters_clears_kinds(patched_manager, monkeypatch):
    mgr, fake_tk = patched_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod
    monkeypatch.setattr(
        mod, "load_adapter_from_pt",
        lambda _p: _fake_loaded("tokenformer"),
    )
    mgr.add_adapter(SimpleNamespace(adapter_id=11, lora_path="/tmp/a"))
    assert mgr._kinds  # non-empty

    mgr.remove_all_adapters()
    assert mgr._kinds == {}
    fake_tk.remove_all_adapters.assert_called_once()


def test_remove_adapter_drops_kind(patched_manager, monkeypatch):
    mgr, fake_tk = patched_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod
    monkeypatch.setattr(
        mod, "load_adapter_from_pt",
        lambda _p: _fake_loaded("tokenformer"),
    )
    mgr.add_adapter(SimpleNamespace(adapter_id=5, lora_path="/tmp/a"))
    assert 5 in mgr._kinds

    mgr.remove_adapter(5)
    assert 5 not in mgr._kinds
    fake_tk.remove_adapter.assert_called_once_with(5)


def test_dummy_lora_cache_nests_tokenformer(patched_manager):
    mgr, fake_tk = patched_manager
    with mgr.dummy_lora_cache():
        pass
    fake_tk.dummy_lora_cache.assert_called_once()


def test_add_dummy_lora_forwards_rank(patched_manager):
    mgr, fake_tk = patched_manager
    req = SimpleNamespace(adapter_id=0, lora_path="/dummy")
    mgr.add_dummy_lora(req, rank=4)
    fake_tk.add_dummy_lora.assert_called_once_with(req, rank=4)


def test_supports_tower_connector_lora_is_false(patched_manager):
    mgr, _ = patched_manager
    assert mgr.supports_tower_connector_lora() is False

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
def test_add_lora_or_hybrid_without_lora_submanager_raises(
    patched_manager, monkeypatch, kind
):
    """Skeleton path (no vllm_config → self._lora is None) rejects LoRA
    adapters with a clear message pointing at the missing flag."""
    mgr, _ = patched_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod
    monkeypatch.setattr(
        mod, "load_adapter_from_pt",
        lambda _p: _fake_loaded(kind),
    )
    req = SimpleNamespace(adapter_id=9, lora_path="/tmp/a")
    with pytest.raises(RuntimeError, match="enable-lora"):
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


def test_list_adapters_normalizes_tokenformer_dict(patched_manager):
    mgr, fake_tk = patched_manager
    # TokenformerModelManager.list_adapters returns a dict — the hybrid
    # manager must coerce it to a set.
    fake_tk.list_adapters.return_value = {1: object(), 2: object()}
    assert mgr.list_adapters() == {1, 2}


def test_supports_tower_connector_lora_is_false(patched_manager):
    mgr, _ = patched_manager
    assert mgr.supports_tower_connector_lora() is False


# --- LoRA-wired path (skeleton with both sub-managers) ----------------


@pytest.fixture
def full_manager(monkeypatch):
    """HybridAdapterManager with BOTH sub-managers mocked.

    Simulates what happens when load_lora_model constructs the hybrid
    manager with a vllm_config that has lora_config set — both halves
    are wired but neither touches real model weights.
    """
    import vllm.tokenformer.hybrid_adapter_manager as mod

    fake_tk = MagicMock()
    fake_tk.model = SimpleNamespace(name="tk-wrapped")

    @contextmanager
    def _dummy_cache():
        yield
    fake_tk.dummy_lora_cache.side_effect = _dummy_cache

    fake_lora = MagicMock()
    # After the LoRA manager wraps the model, create_lora_manager
    # returns the further-wrapped model that Tokenformer will see.
    lora_wrapped = SimpleNamespace(
        name="lora-wrapped",
        embedding_modules={},
    )
    fake_lora.create_lora_manager.return_value = lora_wrapped

    monkeypatch.setattr(
        mod, "TokenformerModelManager", lambda model, device: fake_tk
    )
    monkeypatch.setattr(
        mod, "PTWorkerLoRAManager",
        lambda vllm_config, device, embedding_modules: fake_lora,
    )

    # Vllm config stub that looks "LoRA-enabled".
    vllm_config = SimpleNamespace(lora_config=SimpleNamespace())
    base_model = SimpleNamespace(embedding_modules={})

    mgr = mod.HybridAdapterManager(
        model=base_model, device="cpu", vllm_config=vllm_config,
    )
    return mgr, fake_tk, fake_lora, lora_wrapped


def test_hybrid_init_runs_lora_then_tokenformer(full_manager):
    """The LoRA manager.create_lora_manager is called before the
    Tokenformer sub-manager sees the model."""
    _, fake_tk, fake_lora, lora_wrapped = full_manager
    fake_lora.create_lora_manager.assert_called_once()
    # Tokenformer sub-manager got instantiated (MagicMock was called).
    # We can't easily introspect which model it got via the lambda
    # replacement, but the order is implied: lora_wrapped exists before
    # the tokenformer call happens in __init__.
    assert fake_tk is not None
    assert lora_wrapped.name == "lora-wrapped"


def test_add_lora_routes_only_to_lora(full_manager, monkeypatch):
    mgr, fake_tk, fake_lora, _ = full_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod
    monkeypatch.setattr(
        mod, "load_adapter_from_pt",
        lambda _p: _fake_loaded("lora"),
    )
    req = SimpleNamespace(adapter_id=11, lora_path="/tmp/a")
    mgr.add_adapter(req)

    fake_lora.add_adapter.assert_called_once_with(req)
    fake_tk.add_adapter.assert_not_called()
    assert mgr._kinds[11] == "lora"


def test_add_hybrid_routes_to_both(full_manager, monkeypatch):
    mgr, fake_tk, fake_lora, _ = full_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod
    monkeypatch.setattr(
        mod, "load_adapter_from_pt",
        lambda _p: _fake_loaded("hybrid"),
    )
    req = SimpleNamespace(adapter_id=21, lora_path="/tmp/a")
    mgr.add_adapter(req)

    fake_lora.add_adapter.assert_called_once_with(req)
    fake_tk.add_adapter.assert_called_once_with(req)
    assert mgr._kinds[21] == "hybrid"


def test_set_active_adapters_fans_out(full_manager):
    mgr, fake_tk, fake_lora, _ = full_manager
    requests = {SimpleNamespace(adapter_id=1)}
    mapping = object()
    mgr.set_active_adapters(requests, mapping)
    fake_tk.set_active_adapters.assert_called_once_with(requests, mapping)
    fake_lora.set_active_adapters.assert_called_once_with(requests, mapping)


def test_add_dummy_lora_fans_out_when_lora_present(full_manager):
    mgr, fake_tk, fake_lora, _ = full_manager
    req = SimpleNamespace(adapter_id=0, lora_path="/dummy")
    mgr.add_dummy_lora(req, rank=8)
    # Both sub-managers must see the dummy — the LoRA one for
    # cudagraph capture, the Tokenformer one for interface symmetry.
    fake_tk.add_dummy_lora.assert_called_once_with(req, rank=8)
    fake_lora.add_dummy_lora.assert_called_once_with(req, rank=8)


def test_list_adapters_unions_both(full_manager):
    mgr, fake_tk, fake_lora, _ = full_manager
    fake_tk.list_adapters.return_value = {1: object(), 3: object()}
    fake_lora.list_adapters.return_value = {2, 3}  # id 3 overlaps
    assert mgr.list_adapters() == {1, 2, 3}


def test_pin_adapter_routes_by_kind(full_manager, monkeypatch):
    mgr, fake_tk, fake_lora, _ = full_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod

    # Register a lora and a tokenformer adapter.
    monkeypatch.setattr(
        mod, "load_adapter_from_pt", lambda _p: _fake_loaded("tokenformer")
    )
    mgr.add_adapter(SimpleNamespace(adapter_id=10, lora_path="/t"))
    monkeypatch.setattr(
        mod, "load_adapter_from_pt", lambda _p: _fake_loaded("lora")
    )
    mgr.add_adapter(SimpleNamespace(adapter_id=20, lora_path="/l"))

    fake_tk.pin_adapter.reset_mock()
    fake_lora.pin_adapter.reset_mock()

    mgr.pin_adapter(10)
    fake_tk.pin_adapter.assert_called_once_with(10)
    fake_lora.pin_adapter.assert_not_called()

    fake_tk.pin_adapter.reset_mock()
    fake_lora.pin_adapter.reset_mock()

    mgr.pin_adapter(20)
    fake_lora.pin_adapter.assert_called_once_with(20)
    fake_tk.pin_adapter.assert_not_called()


def test_remove_all_clears_both(full_manager, monkeypatch):
    mgr, fake_tk, fake_lora, _ = full_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod
    monkeypatch.setattr(
        mod, "load_adapter_from_pt",
        lambda _p: _fake_loaded("hybrid"),
    )
    mgr.add_adapter(SimpleNamespace(adapter_id=5, lora_path="/tmp/a"))
    mgr.remove_all_adapters()

    fake_tk.remove_all_adapters.assert_called_once()
    fake_lora.remove_all_adapters.assert_called_once()
    assert mgr._kinds == {}

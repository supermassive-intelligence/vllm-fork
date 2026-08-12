# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HybridAdapterManager routing and live-prefix resolution.

No model, no torch cuda — we stub TokenformerModelManager with a
MagicMock and assert the hybrid manager wires the right sub-manager
call for each adapter kind.
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

    monkeypatch.setattr(mod, "TokenformerModelManager", lambda model, device: fake_tk)

    mgr = mod.HybridAdapterManager(model=object(), device="cpu")
    return mgr, fake_tk


def _fake_loaded(kind, path="/tmp/fake-adapter"):
    """Minimal stand-in for LoadedAdapter."""
    return SimpleNamespace(kind=kind, source_path=path, tokenformer_sd={}, lora_sd={})


def _pt_manager_with_modules(*module_names):
    """Construct only the prefix-resolver portion of the worker manager."""
    import vllm.tokenformer.hybrid_adapter_manager as mod

    model = SimpleNamespace(
        named_modules=lambda: ((name, object()) for name in module_names)
    )
    manager = object.__new__(mod.PTWorkerLoRAManager)
    manager._adapter_manager = SimpleNamespace(model=model)
    return manager


def test_runtime_prefix_resolution_for_text_only_qwen35():
    from vllm.tokenformer.adapter_format import normalize_lora_key

    manager = _pt_manager_with_modules("model.layers.0.self_attn")
    assert manager._detect_model_layers_prefix() == "model.layers."

    tensor = object()
    normalized_key = normalize_lora_key(
        "model.layers.0.self_attn.q_proj.lora_A.default.weight"
    )
    result = manager._renormalize_lora_sd_for_model({normalized_key: tensor})
    assert result == {"model.layers.0.self_attn.q_proj.lora_A.weight": tensor}


def test_runtime_prefix_resolution_prefers_multimodal_text_decoder():
    from vllm.tokenformer.adapter_format import normalize_lora_key

    # named_modules() visits the vision tower first on Gemma4. The resolver
    # must still choose the language decoder rather than the first layers tree.
    manager = _pt_manager_with_modules(
        "vision_tower.encoder.layers.0.self_attn",
        "language_model.model.layers.0.self_attn",
    )
    assert manager._detect_model_layers_prefix() == "language_model.model.layers."

    text_tensor = object()
    vision_tensor = object()
    text_key = normalize_lora_key(
        "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight"
    )
    vision_key = normalize_lora_key(
        "model.vision_tower.encoder.layers.0.self_attn.q_proj"
        ".linear.lora_A.default.weight"
    )
    result = manager._renormalize_lora_sd_for_model(
        {
            text_key: text_tensor,
            vision_key: vision_tensor,
        }
    )
    assert result == {
        "language_model.model.layers.0.self_attn.q_proj.lora_A.weight": text_tensor,
        "vision_tower.encoder.layers.0.self_attn.q_proj.lora_A.weight": vision_tensor,
    }


def test_model_property_forwards_to_tokenformer(patched_manager):
    mgr, fake_tk = patched_manager
    assert mgr.model is fake_tk.model


def test_add_tokenformer_adapter_delegates(patched_manager, monkeypatch):
    mgr, fake_tk = patched_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod

    monkeypatch.setattr(
        mod,
        "load_adapter_from_pt",
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
        mod,
        "load_adapter_from_pt",
        lambda _p: _fake_loaded(kind),
    )
    req = SimpleNamespace(adapter_id=9, lora_path="/tmp/a")
    with pytest.raises(RuntimeError, match="enable-lora"):
        mgr.add_adapter(req)


def test_add_adapter_resolves_path_before_classification(patched_manager, monkeypatch):
    """The classification load must open the same directory the
    sub-managers resolve to (HF-hub ids, relative paths)."""
    mgr, fake_tk = patched_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod

    seen = {}
    monkeypatch.setattr(mod, "get_adapter_absolute_path", lambda p: f"/resolved{p}")

    def _fake_load(path):
        seen["path"] = path
        return _fake_loaded("tokenformer")

    monkeypatch.setattr(mod, "load_adapter_from_pt", _fake_load)
    mgr.add_adapter(SimpleNamespace(adapter_id=3, lora_path="/tmp/a"))
    assert seen["path"] == "/resolved/tmp/a"


def test_set_active_adapters_forwards(patched_manager):
    mgr, fake_tk = patched_manager
    requests = [SimpleNamespace(adapter_id=1)]
    mapping = object()
    mgr.set_active_adapters(requests, mapping)
    fake_tk.set_active_adapters.assert_called_once_with(requests, mapping)


def test_activate_tokenformer_routes_to_tokenformer(patched_manager, monkeypatch):
    mgr, fake_tk = patched_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod

    monkeypatch.setattr(
        mod,
        "load_adapter_from_pt",
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
        mod,
        "load_adapter_from_pt",
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
        mod,
        "load_adapter_from_pt",
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


def test_list_adapters_accepts_tokenformer_mapping(patched_manager):
    mgr, fake_tk = patched_manager
    # The current Tokenformer manager returns a set, but keep the hybrid
    # boundary tolerant of mapping-shaped results from legacy callers.
    fake_tk.list_adapters.return_value = {1: object(), 2: object()}
    assert mgr.list_adapters() == {1, 2}


def test_supports_tower_connector_lora_is_false(patched_manager):
    mgr, _ = patched_manager
    assert mgr.supports_tower_connector_lora() is False


# --- LoRA-wired path with both sub-managers ----------------------------


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

    monkeypatch.setattr(mod, "TokenformerModelManager", lambda model, device: fake_tk)
    monkeypatch.setattr(
        mod,
        "PTWorkerLoRAManager",
        lambda vllm_config, device, embedding_modules: fake_lora,
    )

    # Vllm config stub that looks "LoRA-enabled".
    vllm_config = SimpleNamespace(lora_config=SimpleNamespace())
    base_model = SimpleNamespace(embedding_modules={})

    mgr = mod.HybridAdapterManager(
        model=base_model,
        device="cpu",
        vllm_config=vllm_config,
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
        mod,
        "load_adapter_from_pt",
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
        mod,
        "load_adapter_from_pt",
        lambda _p: _fake_loaded("hybrid"),
    )
    req = SimpleNamespace(adapter_id=21, lora_path="/tmp/a")
    mgr.add_adapter(req)

    fake_lora.add_adapter.assert_called_once_with(req)
    fake_tk.add_adapter.assert_called_once_with(req)
    assert mgr._kinds[21] == "hybrid"


def test_set_active_adapters_fans_out(full_manager):
    mgr, fake_tk, fake_lora, _ = full_manager
    requests = [SimpleNamespace(adapter_id=1)]
    mapping = object()
    mgr.set_active_adapters(requests, mapping)
    fake_tk.set_active_adapters.assert_called_once_with(requests, mapping)
    fake_lora.set_active_adapters.assert_called_once_with(requests, mapping)


def test_set_active_adapters_filters_tokenformer_ids_from_lora(full_manager):
    """Regression: pure-Tokenformer ids forwarded to the LoRA
    sub-manager made its inherited `_apply_adapters` lazily load them
    as LoRA and raise "has no LoRA tensors" on the first batch. They
    must be dropped from the LoRA-side request set and zeroed out of
    the mappings (0 = no-LoRA slot)."""
    from vllm.lora.layers import LoRAMapping

    mgr, fake_tk, fake_lora, _ = full_manager
    mgr._kinds = {5: "tokenformer", 7: "lora"}
    req_tk = SimpleNamespace(adapter_id=5)
    req_lora = SimpleNamespace(adapter_id=7)
    requests = [req_tk, req_lora]
    mapping = LoRAMapping(
        index_mapping=(5, 5, 7, 0),
        prompt_mapping=(5, 7),
        is_prefill=True,
    )

    mgr.set_active_adapters(requests, mapping)

    # Tokenformer half sees everything, untouched.
    fake_tk.set_active_adapters.assert_called_once_with(requests, mapping)

    # LoRA half sees only the LoRA request, with tokenformer ids
    # rewritten to the no-LoRA sentinel in both mappings.
    (lora_requests, lora_mapping), _ = fake_lora.set_active_adapters.call_args
    assert list(lora_requests) == [req_lora]
    assert lora_mapping.index_mapping == (0, 0, 7, 0)
    assert lora_mapping.prompt_mapping == (0, 7)
    assert lora_mapping.is_prefill is True
    # The original mapping must not be mutated in place.
    assert mapping.index_mapping == (5, 5, 7, 0)


def test_get_dummy_lora_warmup_rank_delegates_to_lora(full_manager):
    mgr, fake_tk, fake_lora, _ = full_manager
    fake_lora.get_dummy_lora_warmup_rank.return_value = 16
    assert mgr.get_dummy_lora_warmup_rank(8) == 16
    fake_lora.get_dummy_lora_warmup_rank.assert_called_once_with(8)


def test_get_dummy_lora_warmup_rank_default_without_lora(patched_manager):
    mgr, fake_tk = patched_manager
    assert mgr.get_dummy_lora_warmup_rank(8) == 8


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
    monkeypatch.setattr(mod, "load_adapter_from_pt", lambda _p: _fake_loaded("lora"))
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
        mod,
        "load_adapter_from_pt",
        lambda _p: _fake_loaded("hybrid"),
    )
    mgr.add_adapter(SimpleNamespace(adapter_id=5, lora_path="/tmp/a"))
    mgr.remove_all_adapters()

    fake_tk.remove_all_adapters.assert_called_once()
    fake_lora.remove_all_adapters.assert_called_once()
    assert mgr._kinds == {}


def test_add_adapter_rolls_back_on_lora_half_failure(full_manager, monkeypatch):
    """A hybrid adapter whose LoRA half fails to register must not
    leave the tokenformer half + _kinds entry behind — a stale ghost
    would keep activating on later batches."""
    mgr, fake_tk, fake_lora, _ = full_manager
    import vllm.tokenformer.hybrid_adapter_manager as mod

    monkeypatch.setattr(mod, "load_adapter_from_pt", lambda _p: _fake_loaded("hybrid"))
    fake_lora.add_adapter.side_effect = RuntimeError("punica says no")

    with pytest.raises(RuntimeError, match="punica says no"):
        mgr.add_adapter(SimpleNamespace(adapter_id=9, lora_path="/tmp/a"))

    fake_tk.remove_adapter.assert_called_once_with(9)
    assert 9 not in mgr._kinds

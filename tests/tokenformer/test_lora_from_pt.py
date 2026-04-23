# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the LoRA-from-.pt loader.

`infer_lora_rank` and `build_peft_helper_from_pt` are pure logic and
can run with fake shape-only tensors (no torch needed). The full
`load_lora_model_from_pt` path needs real torch tensors because
`LoRAModel.from_lora_tensors` moves tensors between devices; those
tests skip cleanly when torch is unavailable.
"""

from types import SimpleNamespace

import pytest


def _fake_tensor(shape):
    """Shape-only stand-in for tests of the rank/alpha inference logic."""
    return SimpleNamespace(shape=shape)


# --- infer_lora_rank ----------------------------------------------------


def test_infer_rank_single_module():
    from vllm.tokenformer.lora_from_pt import infer_lora_rank

    sd = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": _fake_tensor((16, 4096)),
        "model.layers.0.self_attn.q_proj.lora_B.weight": _fake_tensor((4096, 16)),
    }
    assert infer_lora_rank(sd) == 16


def test_infer_rank_multiple_modules_consistent():
    from vllm.tokenformer.lora_from_pt import infer_lora_rank

    sd = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": _fake_tensor((32, 4096)),
        "model.layers.0.self_attn.k_proj.lora_A.weight": _fake_tensor((32, 4096)),
        "model.layers.0.self_attn.v_proj.lora_A.weight": _fake_tensor((32, 4096)),
        # lora_B tensors are ignored for rank inference.
        "model.layers.0.self_attn.q_proj.lora_B.weight": _fake_tensor((4096, 32)),
    }
    assert infer_lora_rank(sd) == 32


def test_infer_rank_inconsistent_raises():
    from vllm.tokenformer.lora_from_pt import infer_lora_rank

    sd = {
        "q_proj.lora_A.weight": _fake_tensor((16, 4096)),
        "k_proj.lora_A.weight": _fake_tensor((32, 4096)),
    }
    with pytest.raises(ValueError, match="Inconsistent"):
        infer_lora_rank(sd)


def test_infer_rank_no_lora_a_raises():
    from vllm.tokenformer.lora_from_pt import infer_lora_rank

    # Only lora_B tensors (malformed adapter).
    sd = {"q_proj.lora_B.weight": _fake_tensor((4096, 16))}
    with pytest.raises(ValueError, match="No lora_A tensors"):
        infer_lora_rank(sd)


def test_infer_rank_zero_raises():
    from vllm.tokenformer.lora_from_pt import infer_lora_rank

    sd = {"q_proj.lora_A.weight": _fake_tensor((0, 4096))}
    with pytest.raises(ValueError, match="positive integer"):
        infer_lora_rank(sd)


def test_infer_rank_ignores_non_lora_keys():
    from vllm.tokenformer.lora_from_pt import infer_lora_rank

    sd = {
        "some.embed_tokens.weight": _fake_tensor((32000, 4096)),
        "q_proj.lora_A.weight": _fake_tensor((8, 4096)),
    }
    assert infer_lora_rank(sd) == 8


# --- build_peft_helper_from_pt -----------------------------------------


def test_default_alpha_is_2x_rank():
    from vllm.tokenformer.lora_from_pt import build_peft_helper_from_pt

    sd = {"q_proj.lora_A.weight": _fake_tensor((16, 4096))}
    helper = build_peft_helper_from_pt(sd)
    assert helper.r == 16
    assert helper.lora_alpha == 32
    assert helper.vllm_lora_scaling_factor == pytest.approx(32 / 16)


def test_explicit_alpha_is_respected():
    from vllm.tokenformer.lora_from_pt import build_peft_helper_from_pt

    sd = {"q_proj.lora_A.weight": _fake_tensor((16, 4096))}
    helper = build_peft_helper_from_pt(sd, lora_alpha=64)
    assert helper.r == 16
    assert helper.lora_alpha == 64


def test_alpha_multiplier_controls_default():
    from vllm.tokenformer.lora_from_pt import build_peft_helper_from_pt

    sd = {"q_proj.lora_A.weight": _fake_tensor((8, 4096))}
    helper = build_peft_helper_from_pt(sd, lora_alpha_multiplier=4.0)
    assert helper.lora_alpha == 32


def test_metadata_alpha_overrides_default():
    from vllm.tokenformer.lora_from_pt import build_peft_helper_from_pt

    sd = {"q_proj.lora_A.weight": _fake_tensor((16, 4096))}
    helper = build_peft_helper_from_pt(sd, metadata={"lora_alpha": 48})
    assert helper.r == 16
    assert helper.lora_alpha == 48


def test_metadata_use_rslora_is_respected():
    from vllm.tokenformer.lora_from_pt import build_peft_helper_from_pt

    sd = {"q_proj.lora_A.weight": _fake_tensor((16, 4096))}
    helper = build_peft_helper_from_pt(sd, metadata={"use_rslora": True})
    import math
    assert helper.vllm_lora_scaling_factor == pytest.approx(
        helper.lora_alpha / math.sqrt(helper.r)
    )


def test_explicit_alpha_beats_metadata():
    from vllm.tokenformer.lora_from_pt import build_peft_helper_from_pt

    sd = {"q_proj.lora_A.weight": _fake_tensor((16, 4096))}
    # Explicit kwarg takes precedence — tests stay deterministic even
    # when a file happens to ship metadata.
    helper = build_peft_helper_from_pt(
        sd, lora_alpha=100, metadata={"lora_alpha": 48},
    )
    assert helper.lora_alpha == 100


def test_warning_when_no_metadata_alpha(caplog):
    import logging
    from vllm.tokenformer.lora_from_pt import build_peft_helper_from_pt

    sd = {"q_proj.lora_A.weight": _fake_tensor((16, 4096))}
    with caplog.at_level(logging.WARNING, logger="vllm.tokenformer.lora_from_pt"):
        build_peft_helper_from_pt(sd)
    # Warning must name the default we picked so operators can verify
    # it matches their training config.
    assert any("lora_alpha" in r.message for r in caplog.records)


def test_no_warning_when_metadata_alpha_present(caplog):
    import logging
    from vllm.tokenformer.lora_from_pt import build_peft_helper_from_pt

    sd = {"q_proj.lora_A.weight": _fake_tensor((16, 4096))}
    with caplog.at_level(logging.WARNING, logger="vllm.tokenformer.lora_from_pt"):
        build_peft_helper_from_pt(sd, metadata={"lora_alpha": 32})
    assert not any("lora_alpha" in r.message for r in caplog.records)


def test_rslora_scaling_uses_sqrt_of_rank():
    from vllm.tokenformer.lora_from_pt import build_peft_helper_from_pt

    sd = {"q_proj.lora_A.weight": _fake_tensor((16, 4096))}
    helper = build_peft_helper_from_pt(sd, use_rslora=True)
    # vllm_lora_scaling_factor = lora_alpha / sqrt(r) under rsLoRA
    import math
    assert helper.vllm_lora_scaling_factor == pytest.approx(
        helper.lora_alpha / math.sqrt(helper.r)
    )


# --- load_lora_model_from_pt -------------------------------------------


def test_load_lora_model_rejects_rank_over_max(monkeypatch):
    """The trainer's rank must be ≤ the server's max_lora_rank.
    Otherwise vLLM pre-allocated slots are too small and set_lora
    crashes with a cryptic shape-mismatch. Catch it early with a
    clear message."""
    from vllm.tokenformer.lora_from_pt import load_lora_model_from_pt

    sd = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": _fake_tensor((32, 4096)),
        "model.layers.0.self_attn.q_proj.lora_B.weight": _fake_tensor((4096, 32)),
    }
    with pytest.raises(ValueError, match=r"max_lora_rank=16"):
        load_lora_model_from_pt(sd, lora_model_id=1, max_lora_rank=16)


def test_load_lora_model_from_pt_empty_sd_raises():
    from vllm.tokenformer.lora_from_pt import load_lora_model_from_pt

    with pytest.raises(ValueError, match="empty"):
        load_lora_model_from_pt({}, lora_model_id=1)


def test_load_lora_model_from_pt_dispatches(monkeypatch):
    """Without torch, stub LoRAModel.from_lora_tensors and verify the
    loader passes the right arguments through."""
    import vllm.tokenformer.lora_from_pt as mod

    captured = {}

    class _FakeLoRAModel:
        @classmethod
        def from_lora_tensors(cls, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(id=kwargs["lora_model_id"])

    monkeypatch.setattr(mod, "LoRAModel", _FakeLoRAModel)

    sd = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": _fake_tensor((16, 4096)),
        "model.layers.0.self_attn.q_proj.lora_B.weight": _fake_tensor((4096, 16)),
    }

    result = mod.load_lora_model_from_pt(sd, lora_model_id=42)

    assert result.id == 42
    assert captured["lora_model_id"] == 42
    assert captured["tensors"] is sd
    helper = captured["peft_helper"]
    assert helper.r == 16
    assert helper.lora_alpha == 32


def test_explicit_peft_helper_is_used(monkeypatch):
    import vllm.tokenformer.lora_from_pt as mod
    from vllm.lora.peft_helper import PEFTHelper

    captured = {}

    class _FakeLoRAModel:
        @classmethod
        def from_lora_tensors(cls, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

    monkeypatch.setattr(mod, "LoRAModel", _FakeLoRAModel)

    helper = PEFTHelper(r=8, lora_alpha=16, target_modules=[])
    sd = {
        # Shape doesn't match helper.r intentionally; loader should not
        # re-infer when an explicit helper is passed.
        "q_proj.lora_A.weight": _fake_tensor((64, 4096)),
        "q_proj.lora_B.weight": _fake_tensor((4096, 64)),
    }
    mod.load_lora_model_from_pt(sd, lora_model_id=1, peft_helper=helper)

    assert captured["peft_helper"] is helper


# --- real torch round-trip (skipped if torch not installed) ------------


def test_load_lora_model_from_pt_real_torch(monkeypatch):
    torch = pytest.importorskip("torch")

    # We still stub LoRAModel.from_lora_tensors so this test doesn't
    # require a full model; the point is that real tensor shapes
    # thread through inference + helper construction.
    import vllm.tokenformer.lora_from_pt as mod

    captured = {}

    class _FakeLoRAModel:
        @classmethod
        def from_lora_tensors(cls, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

    monkeypatch.setattr(mod, "LoRAModel", _FakeLoRAModel)

    sd = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(16, 4096),
        "model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4096, 16),
        "model.layers.0.self_attn.k_proj.lora_A.weight": torch.zeros(16, 4096),
        "model.layers.0.self_attn.k_proj.lora_B.weight": torch.zeros(4096, 16),
    }
    mod.load_lora_model_from_pt(sd, lora_model_id=7, device="cpu")

    helper = captured["peft_helper"]
    assert helper.r == 16
    assert captured["device"] == "cpu"
    assert captured["tensors"] is sd

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for unpack_packed_modules_state_dict — the ScalarLM
state-dict rewrite shared by the gemma3/qwen2/qwen3/qwen3_moe
ForCausalLM.state_dict overrides."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models.utils import unpack_packed_modules_state_dict

MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
# 4 q heads, 2 kv heads, head_dim 8 -> q rows 32, k/v rows 16 each
# (fused qkv has 64 rows; a naive 3-way chunk would slice at 22/43).
CFG = SimpleNamespace(
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    hidden_size=32,
)


def _decoder_sd(prefix=""):
    qkv = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32)
    return {
        f"{prefix}model.layers.0.self_attn.qkv_proj.weight": qkv,
        f"{prefix}model.layers.0.self_attn.qkv_proj.bias": torch.arange(
            64, dtype=torch.float32
        ),
        f"{prefix}model.layers.0.mlp.gate_up_proj.weight": torch.zeros(20, 32),
        f"{prefix}model.layers.0.self_attn.attn._k_scale": torch.tensor(1.0),
        f"{prefix}model.layers.1.mlp.experts.w13_weight": torch.zeros(2, 4),
        f"{prefix}model.norm.weight": torch.ones(32),
    }


def test_top_level_unpacks_with_gqa_sizes():
    sd = _decoder_sd()
    qkv = sd["model.layers.0.self_attn.qkv_proj.weight"]
    out = unpack_packed_modules_state_dict(
        sd, prefix="", packed_modules_mapping=MAPPING, config=CFG
    )
    assert out is sd  # must hand back the same (destination) dict

    assert "model.layers.0.self_attn.qkv_proj.weight" not in out
    assert torch.equal(out["model.layers.0.self_attn.q_proj.weight"], qkv[:32])
    assert torch.equal(out["model.layers.0.self_attn.k_proj.weight"], qkv[32:48])
    assert torch.equal(out["model.layers.0.self_attn.v_proj.weight"], qkv[48:])
    assert out["model.layers.0.self_attn.q_proj.bias"].shape == (32,)
    assert out["model.layers.0.self_attn.k_proj.bias"].shape == (16,)

    assert out["model.layers.0.mlp.gate_proj.weight"].shape == (10, 32)
    assert out["model.layers.0.mlp.up_proj.weight"].shape == (10, 32)

    # Non-loadable keys are dropped; everything else survives.
    assert "model.layers.0.self_attn.attn._k_scale" not in out
    assert "model.layers.1.mlp.experts.w13_weight" not in out
    assert "model.norm.weight" in out


def test_head_dim_none_falls_back_to_hidden_over_heads():
    cfg = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=None,  # HF configs may carry an explicit None
        hidden_size=32,
    )
    out = unpack_packed_modules_state_dict(
        _decoder_sd(), prefix="", packed_modules_mapping=MAPPING, config=cfg
    )
    assert out["model.layers.0.self_attn.q_proj.weight"].shape == (32, 32)
    assert out["model.layers.0.self_attn.k_proj.weight"].shape == (16, 32)


def test_prefix_scopes_rewrite_to_own_subtree():
    """Regression for the multimodal-wrapper bug: keys of sibling
    modules already in the shared destination (a vision tower with an
    incompatibly-shaped qkv_proj) must not be touched — splitting them
    with this config's sizes raised RuntimeError in torch.split."""
    sd = _decoder_sd(prefix="language_model.")
    vision_qkv = torch.zeros(9, 5)  # 32/16/16 split cannot apply
    sd["vision_tower.blocks.0.attn.qkv_proj.weight"] = vision_qkv
    sd["vision_tower.blocks.0.attn.attn._k_scale"] = torch.tensor(1.0)

    out = unpack_packed_modules_state_dict(
        sd, prefix="language_model.", packed_modules_mapping=MAPPING, config=CFG
    )

    # Sibling keys untouched (same objects, not unpacked, not dropped).
    assert out["vision_tower.blocks.0.attn.qkv_proj.weight"] is vision_qkv
    assert "vision_tower.blocks.0.attn.attn._k_scale" in out
    # Own subtree unpacked as usual.
    assert "language_model.model.layers.0.self_attn.qkv_proj.weight" not in out
    assert out["language_model.model.layers.0.self_attn.q_proj.weight"].shape == (
        32,
        32,
    )
    assert "language_model.model.layers.1.mlp.experts.w13_weight" not in out


class _FakeTextModel(nn.Module):
    """Stand-in for a ForCausalLM with the shared state_dict override
    (same gate + rewrite pattern as gemma3/qwen2/qwen3/qwen3_moe)."""

    packed_modules_mapping = MAPPING
    config = CFG

    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(32, 64, bias=False)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        from vllm.model_executor.models.utils import (
            scalarlm_state_dict_export_enabled,
        )

        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if not scalarlm_state_dict_export_enabled():
            return state_dict
        return unpack_packed_modules_state_dict(
            state_dict,
            prefix=prefix,
            packed_modules_mapping=self.packed_modules_mapping,
            config=self.config,
        )


class _FakeVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        # Same attribute name as the text model's projection, but a
        # shape the text config's split sizes cannot apply to.
        self.qkv_proj = nn.Linear(5, 9, bias=False)


class _FakeMMWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # Registration order mirrors Gemma3ForConditionalGeneration:
        # the vision tower's keys land in the shared destination before
        # the language model's overridden state_dict runs.
        self.vision_tower = _FakeVisionTower()
        self.language_model = _FakeTextModel()


def test_wrapper_recursion_leaves_siblings_untouched(monkeypatch):
    """torch's Module.state_dict calls each child's (overridden)
    state_dict with the shared destination and the child's prefix; the
    override must only rewrite its own subtree."""
    import vllm.model_executor.models.utils as mutils

    monkeypatch.setattr(mutils, "_scalarlm_state_dict_export", True)
    sd = _FakeMMWrapper().state_dict()

    assert sd["vision_tower.qkv_proj.weight"].shape == (9, 5)
    assert "language_model.qkv_proj.weight" not in sd
    assert sd["language_model.q_proj.weight"].shape == (32, 32)
    assert sd["language_model.k_proj.weight"].shape == (16, 32)
    assert sd["language_model.v_proj.weight"].shape == (16, 32)


def test_override_is_canonical_until_export_enabled(monkeypatch):
    """Regression (codex ultra review): the rewrite used to apply
    unconditionally, so vanilla flows consuming the canonical state
    dict — sharded saves, dummy init — silently lost expert tensors
    and got unpacked projections even with tokenformer disabled. The
    override must be a no-op until a tokenformer manager enables the
    export layout."""
    import vllm.model_executor.models.utils as mutils

    monkeypatch.setattr(mutils, "_scalarlm_state_dict_export", False)
    sd = _FakeTextModel().state_dict()
    assert "qkv_proj.weight" in sd  # canonical, still packed
    assert "q_proj.weight" not in sd

    monkeypatch.setattr(mutils, "_scalarlm_state_dict_export", True)
    sd = _FakeTextModel().state_dict()
    assert "qkv_proj.weight" not in sd
    assert sd["q_proj.weight"].shape == (32, 32)


def test_tokenformer_manager_enables_export(monkeypatch):
    import vllm.model_executor.models.utils as mutils
    from vllm.tokenformer.tokenformer_model_manager import (
        TokenformerModelManager,
    )

    monkeypatch.setattr(mutils, "_scalarlm_state_dict_export", False)
    TokenformerModelManager(model=nn.Linear(2, 2), device=torch.device("cpu"))
    assert mutils.scalarlm_state_dict_export_enabled()

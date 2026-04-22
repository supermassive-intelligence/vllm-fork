# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for adapter `.pt` classification and splitting.

Covers §4.3 of docs/design/hybrid_lora_tokenformer.md.
"""

import pytest

from vllm.tokenformer.adapter_format import (
    AdapterClassification,
    classify_adapter,
    has_lora_keys,
    has_tokenformer_keys,
    split_adapter_state_dict,
)


# --- classification -----------------------------------------------------


def _fake_sd(*keys):
    """Build a state-dict-like dict with sentinel values."""
    return {k: i for i, k in enumerate(keys)}


def test_pure_tokenformer_is_classified_tokenformer():
    sd = _fake_sd(
        "model.layers.0.mlp.tokenformer_k",
        "model.layers.0.mlp.tokenformer_v",
        "model.layers.0.mlp.tokenformer_p",
    )
    c = classify_adapter(sd)
    assert c == AdapterClassification(has_tokenformer=True, has_lora=False)
    assert c.kind == "tokenformer"


def test_pure_lora_is_classified_lora():
    sd = _fake_sd(
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    )
    c = classify_adapter(sd)
    assert c == AdapterClassification(has_tokenformer=False, has_lora=True)
    assert c.kind == "lora"


def test_hybrid_is_classified_hybrid():
    sd = _fake_sd(
        "model.layers.0.mlp.tokenformer_k",
        "model.layers.0.self_attn.q_proj.lora_A.weight",
    )
    c = classify_adapter(sd)
    assert c.has_tokenformer and c.has_lora
    assert c.kind == "hybrid"


def test_tokenformer_with_base_overrides_is_still_tokenformer():
    # Base-weight overrides (norms, embeddings, lm_head) count as
    # Tokenformer for classification purposes.
    sd = _fake_sd(
        "model.layers.0.mlp.tokenformer_k",
        "model.embed_tokens.weight",
        "lm_head.weight",
    )
    assert classify_adapter(sd).kind == "tokenformer"


def test_empty_state_dict_is_rejected():
    c = classify_adapter({})
    assert c.has_tokenformer is False and c.has_lora is False
    with pytest.raises(ValueError, match="neither tokenformer"):
        _ = c.kind


def test_unrelated_keys_only_is_rejected():
    # Just base weights with no tokenformer or lora markers — malformed.
    sd = _fake_sd("model.embed_tokens.weight", "lm_head.weight")
    c = classify_adapter(sd)
    assert c.has_tokenformer is False and c.has_lora is False
    with pytest.raises(ValueError):
        _ = c.kind


# --- key predicates -----------------------------------------------------


def test_has_tokenformer_keys_matches_leaf_only():
    # The match is on the leaf segment, not substring, so a key that
    # contains "tokenformer_k" mid-path should NOT match.
    assert has_tokenformer_keys(["mlp.tokenformer_k"]) is True
    assert (
        has_tokenformer_keys(["mlp.tokenformer_k.extra"])
        is False
    )


def test_has_lora_keys_requires_delimited_segment():
    # Match requires `.lora_A.` / `.lora_B.` as a path segment —
    # a variable named e.g. `lora_A_config` must not match.
    assert has_lora_keys(["q_proj.lora_A.weight"]) is True
    assert has_lora_keys(["q_proj.lora_B.weight"]) is True
    assert has_lora_keys(["some.lora_A_config"]) is False
    assert has_lora_keys(["prefix_lora_A.weight"]) is False


# --- splitting ----------------------------------------------------------


def test_split_routes_tokenformer_and_base_to_tokenformer_sd():
    sd = {
        "model.layers.0.mlp.tokenformer_k": "tk",
        "model.embed_tokens.weight": "emb",
        "lm_head.weight": "head",
    }
    tk_sd, lora_sd = split_adapter_state_dict(sd)
    assert set(tk_sd) == {
        "model.layers.0.mlp.tokenformer_k",
        "model.embed_tokens.weight",
        "lm_head.weight",
    }
    assert lora_sd == {}


def test_split_routes_lora_to_lora_sd():
    sd = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": "A",
        "model.layers.0.self_attn.q_proj.lora_B.weight": "B",
    }
    tk_sd, lora_sd = split_adapter_state_dict(sd)
    assert tk_sd == {}
    assert set(lora_sd) == {
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    }


def test_split_hybrid():
    sd = {
        "model.layers.0.mlp.tokenformer_k": "tk",
        "model.layers.0.self_attn.q_proj.lora_A.weight": "A",
        "model.layers.0.self_attn.q_proj.lora_B.weight": "B",
        "model.embed_tokens.weight": "emb",
    }
    tk_sd, lora_sd = split_adapter_state_dict(sd)
    assert set(tk_sd) == {
        "model.layers.0.mlp.tokenformer_k",
        "model.embed_tokens.weight",
    }
    assert set(lora_sd) == {
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    }


def test_split_is_not_destructive():
    sd = {
        "model.layers.0.mlp.tokenformer_k": "tk",
        "model.layers.0.self_attn.q_proj.lora_A.weight": "A",
    }
    sd_copy = dict(sd)
    _ = split_adapter_state_dict(sd)
    assert sd == sd_copy  # input not mutated

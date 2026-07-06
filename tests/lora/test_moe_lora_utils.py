# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the separate-expert MoE LoRA stacking math.

`moe_lora_utils` is torch-only by design (no vLLM imports) so this reshape logic
is testable without a built vLLM — we load it directly by file path to skip the
heavy `vllm` package __init__.

See docs/superpowers/plans/2026-07-06-separate-expert-lora-converter.md.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "vllm" / "lora" / "moe_lora_utils.py"
)
_spec = importlib.util.spec_from_file_location("moe_lora_utils", _MODULE_PATH)
moe_lora_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moe_lora_utils)

stack_separate_expert_lora = moe_lora_utils.stack_separate_expert_lora
SEPARATE_EXPERT_LEAF_TRIPLES = moe_lora_utils.SEPARATE_EXPERT_LEAF_TRIPLES


def _per_expert(num_experts, rank, in_features, out_features):
    """Distinct per-expert 2-D LoRA tensors (A:(rank,in), B:(out,rank)) so the
    stacking is checked by *identity of values*, not just shape."""
    a = [torch.randn(rank, in_features) for _ in range(num_experts)]
    b = [torch.randn(out_features, rank) for _ in range(num_experts)]
    return a, b


def test_stack_shapes_match_set_lora_contract():
    # set_lora wants lora_a[p]=(E,rank,in), lora_b[p]=(E,out,rank) for p in w1/w2/w3.
    E, rank, in_f, out_f = 4, 8, 16, 16
    a_by_proj, b_by_proj = [], []
    for _ in range(3):  # gate, down, up
        a, b = _per_expert(E, rank, in_f, out_f)
        a_by_proj.append(a)
        b_by_proj.append(b)

    lora_a, lora_b = stack_separate_expert_lora(a_by_proj, b_by_proj)

    assert len(lora_a) == len(lora_b) == 3
    for p in range(3):
        assert lora_a[p].shape == (E, rank, in_f)
        assert lora_b[p].shape == (E, out_f, rank)
        assert lora_a[p].is_contiguous() and lora_b[p].is_contiguous()


def test_stack_preserves_per_expert_values_and_order():
    # The stacked leading axis must index experts in the given order, unmodified.
    E, rank, in_f, out_f = 3, 4, 8, 8
    a_by_proj, b_by_proj = [], []
    for _ in range(3):
        a, b = _per_expert(E, rank, in_f, out_f)
        a_by_proj.append(a)
        b_by_proj.append(b)

    lora_a, lora_b = stack_separate_expert_lora(a_by_proj, b_by_proj)

    for p in range(3):
        for e in range(E):
            assert torch.equal(lora_a[p][e], a_by_proj[p][e])
            assert torch.equal(lora_b[p][e], b_by_proj[p][e])


def test_num_experts_leads_every_projection():
    # A single expert count must lead all three projections (set_lora asserts
    # num_experts == w1_a.shape[0] == w2_a.shape[0] == w3_a.shape[0]).
    E = 5
    a_by_proj = [[torch.randn(2, 4) for _ in range(E)] for _ in range(3)]
    b_by_proj = [[torch.randn(4, 2) for _ in range(E)] for _ in range(3)]
    lora_a, _ = stack_separate_expert_lora(a_by_proj, b_by_proj)
    assert lora_a[0].shape[0] == lora_a[1].shape[0] == lora_a[2].shape[0] == E


def test_rejects_wrong_projection_count():
    a_by_proj = [[torch.randn(2, 4)], [torch.randn(2, 4)]]  # only 2 projections
    b_by_proj = [[torch.randn(4, 2)], [torch.randn(4, 2)]]
    with pytest.raises(ValueError, match="3 projections"):
        stack_separate_expert_lora(a_by_proj, b_by_proj)


def test_rejects_mismatched_or_empty_expert_lists():
    a_by_proj = [[torch.randn(2, 4)], [torch.randn(2, 4)], []]  # 3rd proj empty
    b_by_proj = [[torch.randn(4, 2)], [torch.randn(4, 2)], []]
    with pytest.raises(ValueError, match="non-empty"):
        stack_separate_expert_lora(a_by_proj, b_by_proj)


def test_leaf_triples_are_gate_down_up_order():
    # The list order must be (gate, down, up) to map onto set_lora's [w1, w2, w3].
    assert ("w1", "w2", "w3") in SEPARATE_EXPERT_LEAF_TRIPLES
    assert ("gate_proj", "down_proj", "up_proj") in SEPARATE_EXPERT_LEAF_TRIPLES

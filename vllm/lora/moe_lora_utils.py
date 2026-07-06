# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-tensor helpers for converting *separate*-expert MoE LoRA (`.pt`) weights
into the per-projection stacked lists that `FusedMoEWithLoRA.set_lora` consumes.

Kept dependency-light (torch only, no vLLM imports) so the reshape math is unit
testable without a built vLLM. The stateful glue that reads a `LoRAModel` and
routes to `set_lora` lives in `model_manager.LoRAModelManager`.

Two expert layouts exist in the wild:

- **grouped** (Qwen3MoE `Qwen3MoeExperts`): PEFT adapts one fused module and
  exports two stacked tensors (`experts.base_layer` = gate_up, `experts` = down).
  Handled by `model_manager._stack_moe_lora_weights_gated` — NOT here.
- **separate** (Mixtral / PhiMoE / Qwen2MoE): experts are a `ModuleList` of
  per-expert `nn.Linear`s, so the `.pt` carries one ordinary 2-D LoRA per expert
  per projection (`experts.{i}.w1`, `.w2`, `.w3`). Those are what this module
  stacks.

See `docs/superpowers/plans/2026-07-06-separate-expert-lora-converter.md`.
"""

from __future__ import annotations

import torch

# Per-expert projection leaf names, in the (gate, down, up) order that maps onto
# `set_lora`'s `[w1, w2, w3]` contract (w1=gate → w13 slot 0, w2=down → w2 slot,
# w3=up → w13 slot 1). Probed in order; the first triple whose expert-0 weights
# are all present wins.
#   - Mixtral / PhiMoE name them w1 (gate), w2 (down), w3 (up).
#   - Qwen2MoE-style separate experts use gate_proj / down_proj / up_proj.
SEPARATE_EXPERT_LEAF_TRIPLES: tuple[tuple[str, str, str], ...] = (
    ("w1", "w2", "w3"),
    ("gate_proj", "down_proj", "up_proj"),
)


def stack_separate_expert_lora(
    a_by_proj: list[list[torch.Tensor]],
    b_by_proj: list[list[torch.Tensor]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Stack per-expert 2-D LoRA tensors into `set_lora`'s per-projection lists.

    Inputs are two 3-element lists — one entry per projection in (gate, down, up)
    order — each holding `num_experts` ordinary 2-D LoRA tensors:

        a_by_proj[p][e]: (rank, in_features)     lora_A of expert `e`, projection `p`
        b_by_proj[p][e]: (out_features, rank)    lora_B of expert `e`, projection `p`

    Returns `(lora_a, lora_b)`, each a 3-element `[w1, w2, w3]` list of stacked
    tensors with the num_experts-leading shapes `FusedMoEWithLoRA.set_lora`
    asserts:

        lora_a[p]: (num_experts, rank, in_features)
        lora_b[p]: (num_experts, out_features, rank)

    Unlike the grouped path, no reshape/permute is needed — the per-expert tensors
    are already ordinary 2-D LoRAs, so a plain stack along a new leading expert
    axis yields exactly the required layout.
    """
    if len(a_by_proj) != 3 or len(b_by_proj) != 3:
        raise ValueError(
            f"expected 3 projections (gate, down, up); got "
            f"{len(a_by_proj)} A-lists and {len(b_by_proj)} B-lists"
        )
    lora_a: list[torch.Tensor] = []
    lora_b: list[torch.Tensor] = []
    for p in range(3):
        a_experts, b_experts = a_by_proj[p], b_by_proj[p]
        if not a_experts or len(a_experts) != len(b_experts):
            raise ValueError(
                f"projection {p}: need matching non-empty A/B expert lists, got "
                f"{len(a_experts)} A and {len(b_experts)} B"
            )
        lora_a.append(torch.stack(a_experts, dim=0).contiguous())
        lora_b.append(torch.stack(b_experts, dim=0).contiguous())
    return lora_a, lora_b

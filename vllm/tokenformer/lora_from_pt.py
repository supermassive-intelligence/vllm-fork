# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a vLLM `LoRAModel` from a ScalarLM `.pt` state-dict slice.

Upstream vLLM expects HF PEFT format (adapter_config.json +
adapter_model.safetensors). ScalarLM trains and serves both Tokenformer
and LoRA adapters in a single `.pt` file whose inner
`model_state_dict` contains `lora_A` / `lora_B` tensors. This module
bridges the two: given the LoRA half of a split state dict (from
`adapter_format.split_adapter_state_dict`), it infers rank / alpha from
tensor shapes and hands a `LoRAModel` back.

See `docs/design/hybrid_lora_tokenformer.md` — this is the option C
pure-logic loader, not yet wired into the hybrid manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)


def infer_lora_rank(lora_sd: dict[str, Any]) -> int:
    """Infer LoRA rank from the shapes of lora_A tensors.

    `lora_A.weight` has shape `(rank, in_features)` by construction
    (see vllm/lora/lora_weights.py). We look across every lora_A tensor
    in the state dict and require a single consistent rank — mixing
    ranks within one adapter is not supported.

    Raises `ValueError` if the state dict has no lora_A tensors, or if
    multiple distinct ranks are found.
    """
    ranks_by_key: dict[str, int] = {}
    for key, tensor in lora_sd.items():
        if ".lora_A." in key:
            try:
                rank = int(tensor.shape[0])
            except (AttributeError, TypeError, IndexError) as exc:
                raise ValueError(
                    f"Cannot read shape[0] of tensor at {key!r} to infer rank"
                ) from exc
            ranks_by_key[key] = rank

    if not ranks_by_key:
        raise ValueError(
            "No lora_A tensors found — cannot infer rank from state dict."
        )

    unique_ranks = set(ranks_by_key.values())
    if len(unique_ranks) != 1:
        raise ValueError(
            f"Inconsistent LoRA ranks across lora_A tensors: "
            f"{sorted(unique_ranks)}. Per-layer rank variation is not "
            f"supported. Offending keys: {sorted(ranks_by_key.keys())[:5]}…"
        )
    rank = unique_ranks.pop()
    if rank <= 0:
        raise ValueError(
            f"Inferred LoRA rank is {rank}; expected a positive integer. "
            f"Check the training-side export."
        )
    return rank


def build_peft_helper_from_pt(
    lora_sd: dict[str, Any],
    *,
    lora_alpha: int | None = None,
    lora_alpha_multiplier: float = 2.0,
    use_rslora: bool = False,
) -> PEFTHelper:
    """Construct a minimal `PEFTHelper` from a LoRA-only state dict slice.

    `.pt` adapters don't carry the sidecar `adapter_config.json` that
    PEFT ships with, so we infer what we can from tensor shapes and
    fall back to sensible defaults for everything else.

    - `r` is inferred via `infer_lora_rank`.
    - `lora_alpha` defaults to `rank * lora_alpha_multiplier` (most
      PEFT recipes use 2x). Callers that know the training-time alpha
      (e.g. read from a sidecar) should pass it explicitly.
    - `target_modules` is left empty — vLLM's loader doesn't need it
      at tensor-construction time; packing is driven by the model's
      `packed_modules_mapping` at a later stage.
    """
    r = infer_lora_rank(lora_sd)
    if lora_alpha is None:
        lora_alpha = int(r * lora_alpha_multiplier)
    return PEFTHelper(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[],
        use_rslora=use_rslora,
    )


def load_lora_model_from_pt(
    lora_sd: dict[str, Any],
    *,
    lora_model_id: int,
    device: "str | torch.device" = "cuda",
    dtype: "torch.dtype | None" = None,
    model_vocab_size: int | None = None,
    peft_helper: PEFTHelper | None = None,
) -> LoRAModel:
    """Build a `LoRAModel` from a LoRA-only `.pt` state-dict slice.

    Callers are expected to have already partitioned a hybrid `.pt`
    into its tokenformer and lora halves via
    `adapter_format.split_adapter_state_dict` before calling this.

    If `peft_helper` is None, one is inferred from tensor shapes via
    `build_peft_helper_from_pt`.
    """
    if not lora_sd:
        raise ValueError("lora_sd is empty — nothing to load.")
    if peft_helper is None:
        peft_helper = build_peft_helper_from_pt(lora_sd)
    logger.info(
        "Loading LoRA adapter %s from .pt state-dict slice: "
        "rank=%d, alpha=%d, %d tensors.",
        lora_model_id, peft_helper.r, peft_helper.lora_alpha, len(lora_sd),
    )
    return LoRAModel.from_lora_tensors(
        lora_model_id=lora_model_id,
        tensors=lora_sd,
        peft_helper=peft_helper,
        device=str(device),
        dtype=dtype,
        model_vocab_size=model_vocab_size,
    )

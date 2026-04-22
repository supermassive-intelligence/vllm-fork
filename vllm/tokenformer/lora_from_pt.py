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
    metadata: dict[str, Any] | None = None,
) -> PEFTHelper:
    """Construct a minimal `PEFTHelper` from a LoRA-only state dict slice.

    `.pt` adapters don't carry the sidecar `adapter_config.json` that
    PEFT ships with, so we infer what we can from tensor shapes and
    accept the rest via either an explicit kwarg or a `metadata` dict
    embedded in the `.pt` file.

    Resolution order for `lora_alpha` / `use_rslora`:
      1. Explicit kwarg (takes precedence so tests stay deterministic).
      2. Value in `metadata` (what the trainer wrote into the `.pt`).
      3. Default (`rank * lora_alpha_multiplier`, `use_rslora=False`).

    A warning is logged when the default is used — silently guessing
    alpha would produce a LoRA delta that's the wrong strength.
    """
    r = infer_lora_rank(lora_sd)
    metadata = metadata or {}

    if lora_alpha is None:
        meta_alpha = metadata.get("lora_alpha")
        if meta_alpha is not None:
            lora_alpha = int(meta_alpha)
        else:
            lora_alpha = int(r * lora_alpha_multiplier)
            logger.warning(
                "LoRA adapter has no `lora_alpha` in its metadata; "
                "defaulting to rank * %g = %d. If the adapter was trained "
                "with a different alpha the delta will be mis-scaled. "
                "Bake `lora_alpha` into the .pt metadata dict to silence "
                "this warning.",
                lora_alpha_multiplier, lora_alpha,
            )

    # use_rslora explicit arg wins; otherwise honor metadata; otherwise default.
    if not use_rslora:
        use_rslora = bool(metadata.get("use_rslora", False))

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
    metadata: dict[str, Any] | None = None,
) -> LoRAModel:
    """Build a `LoRAModel` from a LoRA-only `.pt` state-dict slice.

    Callers are expected to have already partitioned a hybrid `.pt`
    into its tokenformer and lora halves via
    `adapter_format.split_adapter_state_dict` before calling this.

    If `peft_helper` is None, one is built from tensor shapes +
    metadata via `build_peft_helper_from_pt`.
    """
    if not lora_sd:
        raise ValueError("lora_sd is empty — nothing to load.")
    if peft_helper is None:
        peft_helper = build_peft_helper_from_pt(lora_sd, metadata=metadata)
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

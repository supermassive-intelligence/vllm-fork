# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter `.pt` file-format helpers.

Both Tokenformer and LoRA adapters ship in the same `.pt` format: a torch
checkpoint with a top-level `model_state_dict` key whose value is a
`dict[str, Tensor]`. The *kind* of an adapter is determined by which keys
are present:

- Tokenformer: leaf keys ending in `tokenformer_k`, `tokenformer_v`, or
  `tokenformer_p`, plus optional base weight overrides (e.g. embeddings,
  norms, lm_head).
- LoRA: keys whose path contains a `.lora_A.` or `.lora_B.` segment.

A single file may contain both kinds; such a file is "hybrid".

This module is pure dict/string logic so it can be unit-tested without
torch, CUDA, or any model weights loaded.

See `docs/design/hybrid_lora_tokenformer.md` §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

AdapterKind: TypeAlias = Literal["tokenformer", "lora", "hybrid"]

_TOKENFORMER_LEAF_SUFFIXES = (
    "tokenformer_k",
    "tokenformer_v",
    "tokenformer_p",
)

_LORA_PATH_SEGMENTS = (
    ".lora_A.",
    ".lora_B.",
)


def _leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def has_tokenformer_keys(state_dict_keys) -> bool:
    """True iff any key looks like a tokenformer parameter."""
    return any(_leaf(k) in _TOKENFORMER_LEAF_SUFFIXES for k in state_dict_keys)


def has_lora_keys(state_dict_keys) -> bool:
    """True iff any key looks like a LoRA parameter."""
    return any(any(seg in k for seg in _LORA_PATH_SEGMENTS)
               for k in state_dict_keys)


@dataclass(frozen=True)
class AdapterClassification:
    has_tokenformer: bool
    has_lora: bool

    @property
    def kind(self) -> AdapterKind:
        if self.has_tokenformer and self.has_lora:
            return "hybrid"
        if self.has_tokenformer:
            return "tokenformer"
        if self.has_lora:
            return "lora"
        raise ValueError(
            "Adapter state dict contains neither tokenformer_{k,v,p} nor "
            ".lora_A./.lora_B. keys — cannot classify. Check that the "
            "adapter file was produced by a supported trainer."
        )


def classify_adapter(state_dict) -> AdapterClassification:
    """Classify an adapter state dict by the keys it contains.

    `state_dict` must be the inner mapping (the value at
    `checkpoint['model_state_dict']`), not the raw checkpoint object.

    Raises `ValueError` if the state dict contains neither kind of key.
    """
    keys = list(state_dict.keys()) if hasattr(state_dict, "keys") \
        else list(state_dict)
    return AdapterClassification(
        has_tokenformer=has_tokenformer_keys(keys),
        has_lora=has_lora_keys(keys),
    )


def split_adapter_state_dict(state_dict):
    """Partition an adapter state dict into (tokenformer_sd, lora_sd).

    - Keys whose leaf is `tokenformer_{k,v,p}` go to `tokenformer_sd`.
    - Keys whose path contains `.lora_A.` or `.lora_B.` go to `lora_sd`.
    - Other keys (base weight overrides like `embed_tokens.weight`,
      `lm_head.weight`, `input_layernorm.weight`, ...) are treated as
      Tokenformer base-weight overrides and go to `tokenformer_sd`. This
      matches today's TokenformerModelManager.activate_adapter behavior
      where *any* non-LoRA key is copied into the base state dict.

    Returns two new dicts; does not mutate the input.
    """
    tokenformer_sd = {}
    lora_sd = {}
    for k, v in state_dict.items():
        if any(seg in k for seg in _LORA_PATH_SEGMENTS):
            lora_sd[k] = v
        else:
            tokenformer_sd[k] = v
    return tokenformer_sd, lora_sd

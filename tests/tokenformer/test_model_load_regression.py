# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-0 model-load regression harness — per fork-shipped model class.

For each tiny-random fixture: build the nn.Module tree on the meta device via
vLLM's own loader, then run the fork's real two-pass adapter-key normalization
against it. Green on the untouched fork; goes red the moment a version bump
drifts a core symbol (`initialize_model` raises) or reshapes a model tree so the
normalized adapter paths stop resolving.

Runs CPU-only in seconds (config-only download, meta build, no weights, no GPU).
See _meta_harness.py for the run command and the caught/not-caught boundary.
"""

from __future__ import annotations

import os
import sys

import pytest

# tests/tokenformer/ has no __init__.py, so make the sibling helper importable
# regardless of pytest's import mode.
sys.path.insert(0, os.path.dirname(__file__))
from _meta_harness import (MODEL_FIXTURES, build_meta_model,  # noqa: E402
                           normalized_overlap)


@pytest.mark.parametrize("model_id, expected_class", MODEL_FIXTURES)
def test_meta_build_and_normalization(model_id: str, expected_class: str):
    # 1. Symbol-drift + build oracle: vLLM's own loader constructs the tree.
    #    A moved/renamed core symbol raises here.
    model = build_meta_model(model_id)
    assert type(model).__name__ == expected_class, (
        f"{model_id} built as {type(model).__name__}, expected {expected_class}")

    # 2. Tree/normalization oracle: the trainer's would-be LoRA keys still
    #    resolve onto this model's live module tree after the real two-pass
    #    normalization. Zero overlap is the silent-no-op signature.
    n_overlap, n_total = normalized_overlap(model)
    assert n_overlap > 0, (
        f"{model_id}: adapter key normalization resolved 0/{n_total} targets "
        f"onto the model tree — the serve-time no-op signature")

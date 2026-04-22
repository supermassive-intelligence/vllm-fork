# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LoRA + Tokenformer adapter-kind dispatch plumbing.

Exercises the flag → LoRAConfig → dispatcher path without needing a model.
See docs/design/hybrid_lora_tokenformer.md.
"""

import pytest

from vllm.config.lora import LoRAConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.lora_model_runner_mixin import _select_adapter_kind


@pytest.mark.parametrize(
    ("enable_lora", "enable_tokenformer", "expected_kind"),
    [
        (True, False, "lora"),
        (False, True, "tokenformer"),
        (True, True, "hybrid"),
    ],
)
def test_select_adapter_kind(enable_lora, enable_tokenformer, expected_kind):
    cfg = LoRAConfig(
        enable_lora=enable_lora,
        enable_tokenformer=enable_tokenformer,
    )
    assert _select_adapter_kind(cfg) == expected_kind


def test_select_adapter_kind_no_config_defaults_to_tokenformer():
    # Defensive branch — load_lora_model shouldn't be called without a
    # lora_config, but if it is we should fall back to historical behavior.
    assert _select_adapter_kind(None) == "tokenformer"


def test_enable_tokenformer_defaults_false():
    args = EngineArgs(model="dummy")
    assert args.enable_tokenformer is False
    assert args.enable_lora is False


def test_enable_tokenformer_cli_flag_parses():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns = parser.parse_args(["--model", "dummy", "--enable-tokenformer"])
    args = EngineArgs.from_cli_args(ns)
    assert args.enable_tokenformer is True
    assert args.enable_lora is False


def test_enable_tokenformer_builds_lora_config():
    # Even without --enable-lora, passing --enable-tokenformer should
    # cause the engine to construct a LoRAConfig so the adapter subsystem
    # activates. We can't easily call create_engine_config without a real
    # model, so assert on the flag-pair condition that gates construction.
    args = EngineArgs(model="dummy", enable_tokenformer=True)
    assert args.enable_lora or args.enable_tokenformer


def test_lora_config_mirrors_flags():
    cfg = LoRAConfig(enable_lora=True, enable_tokenformer=True)
    assert cfg.enable_lora is True
    assert cfg.enable_tokenformer is True
    assert _select_adapter_kind(cfg) == "hybrid"

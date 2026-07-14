# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-0 meta-device model-load regression harness (shared helpers).

The vLLM-fork upgrade plan (docs/reports/2026-07-13-vllm-fork-migration-plan.md,
Phase 0) needs a CPU-only, seconds-fast oracle that goes red the moment a version
bump drifts a core symbol or reshapes a model tree the adapter layer targets.

This reuses the sweep preflight's mechanism
(test/finetune_sweep/preflight.py): build ONLY the nn.Module tree on the meta
device via vLLM's own `initialize_model` (no weights, no GPU, no engine), then run
the REAL two-pass adapter-key normalization against that live tree. Unlike the
preflight — which runs an embedded string script per model via `docker compose
run` — this imports the fork symbols directly, because the harness itself runs
inside the cray image.

What it catches: (1) symbol-drift — if a bump moves/renames a core-vLLM symbol the
loader path imports, `initialize_model` raises here; (2) tree/normalization drift —
if a model class reshapes its module tree, the normalized adapter paths stop
resolving. What it does NOT catch: MoE-LoRA tensor numerics or real weight loading
— those need hardware (Phase 5). Keep that boundary honest.

Run it (from the ScalarLM repo root), bind-mounting the current fork python over
the built image so no recompile is needed:

    docker run --rm --entrypoint python3 \
      -v "$PWD/vllm/vllm:/app/cray/vllm/vllm:ro" scalarlm-cray:latest \
      -m pytest /app/cray/vllm/tests/tokenformer/test_model_load_regression.py -q
"""

from __future__ import annotations

import socket

import pytest


def _has_custom_op(name: str) -> bool:
    """True if vLLM's compiled `_C` extension exposes op `name`.

    A source/`.so` mismatch (the stale-image bind-mount: current model python
    over an older built extension) can leave current model source referencing a
    compiled op the built `.so` lacks. Feature-detect so those fixtures skip
    cleanly instead of erroring at construction."""
    try:
        import torch
        import vllm._custom_ops  # noqa: F401  (import registers the _C ops)
        getattr(torch.ops._C, name)
        return True
    except Exception:
        return False


# The gemma activation (`GeluAndMul`, tanh-approx) binds this compiled op at
# construction; the stale bind-mount image's `.so` predates it. See gemma3-text.
_GEMMA_ACT_OP_PRESENT = _has_custom_op("gelu_tanh_and_mul")

# Model fixtures = the ARCH FAMILIES the sweep's LoRA path actually serves, one
# tiny-random fixture each (config-only download, meta build — seconds). This is
# an ADAPTER-LAYER regression oracle, so the coverage target is what the sweep
# exercises, NOT the whole vLLM registry — upstream-only registry classes the
# fork never serves via a LoRA .pt (e.g. ExaoneMoe, Tarsier) are out of scope.
# `marks` records real, gated gaps so a hole is visible rather than silent.
#
# IDs / expected classes verified building 2026-07-14 on the 2026-06-11 CPU cray
# image: dense + qwen3-moe with the migration-branch python bind-mounted; the
# gemma family probed both bind-mount and as-built (NO_BIND).
MODEL_FIXTURES: list = [
    pytest.param("hf-internal-testing/tiny-random-LlamaForCausalLM",
                 "LlamaForCausalLM", id="llama"),
    pytest.param("yujiepan/qwen3-tiny-random", "Qwen3ForCausalLM", id="qwen3"),
    pytest.param("yujiepan/qwen2.5-tiny-random", "Qwen2ForCausalLM", id="qwen2"),
    pytest.param("yujiepan/qwen3-moe-tiny-random", "Qwen3MoeForCausalLM",
                 id="qwen3-moe"),
    # Gemma dense text (Gemma3ForCausalLM) — gemma-3-4b-it serves in the sweep, so
    # the fork's causal-LM key normalization must resolve onto gemma's tree. vLLM
    # fuses qkv/gate_up, so the exact-name overlap is the SAME 2/13 baseline as
    # llama (only o_proj + down_proj match unfused). Gemma binds the compiled
    # `gelu_tanh_and_mul` op at construction, which the stale-`.so` bind-mount
    # lacks — gate on the op so this RUNS on a consistent/rebuilt image (or the
    # as-built NO_BIND run) and SKIPS cleanly on the stale bind-mount.
    pytest.param("hf-internal-testing/tiny-random-Gemma3ForCausalLM",
                 "Gemma3ForCausalLM", id="gemma3-text",
                 marks=pytest.mark.skipif(
                     not _GEMMA_ACT_OP_PRESENT,
                     reason="built _C lacks gelu_tanh_and_mul (stale-.so "
                            "bind-mount); rebuild the image or run NO_BIND")),
    # Gemma multimodal (Gemma3/Gemma4 ...ForConditionalGeneration) — the arch of
    # the carry-forward vision-prefix fix (_detect_model_layers_prefix must pick
    # the language_model decoder, not the vision tower). Blocked on THIS image:
    # the HF Gemma3Processor/Gemma4Processor need transformers>=5.13, which the
    # image caps (<5.13) → ModelConfig ValidationError before build. IDs +
    # expected classes are grounded and ready; they unblock when Phase 2 lifts
    # the transformers cap.
    pytest.param("yujiepan/gemma-3-tiny-random",
                 "Gemma3ForConditionalGeneration", id="gemma3-mm",
                 marks=pytest.mark.skip(
                     reason="Gemma3Processor needs transformers>=5.13 (image "
                            "caps <5.13); Phase-2 gated")),
    pytest.param("tiny-random/gemma-4-dense",
                 "Gemma4ForConditionalGeneration", id="gemma4",
                 marks=pytest.mark.skip(
                     reason="Gemma4Processor needs transformers>=5.13 (image "
                            "caps <5.13); Phase-2 gated")),
]

# The trainer's would-be LoRA target leaves, by parent block. Mirrors
# preflight.DEFAULT_LORA_TARGETS (unfused Llama/Qwen + fused ChatGLM + granite
# shared_mlp) so a fused arch is not false-flagged. Extra leaves only ADD overlap.
LORA_TARGETS: dict = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h"),
    "self_attention": ("query_key_value", "dense"),
    "shared_mlp": ("input_linear", "output_linear"),
}

_GROUP_READY = False


def ensure_process_group() -> None:
    """Idempotently stand up a single-process gloo group. Model construction
    reads the PP group / TP world size even at TP=1, so the group must exist;
    we never run a collective. gloo works on the CPU and GPU images alike."""
    global _GROUP_READY
    if _GROUP_READY:
        return
    from vllm.distributed import init_distributed_environment
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:%d" % port, backend="gloo")
    _GROUP_READY = True


def build_meta_model(model_id: str):
    """Build only the nn.Module tree on the meta device via vLLM's own loader.
    Raises on symbol-drift (a moved core symbol) — that raise IS the signal."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.distributed import initialize_model_parallel

    from vllm.distributed import (destroy_model_parallel,
                                  model_parallel_is_initialized)

    vllm_config = EngineArgs(
        model=model_id, load_format="dummy", enforce_eager=True,
        trust_remote_code=True,
    ).create_engine_config()
    ensure_process_group()
    with set_current_vllm_config(vllm_config):
        # Re-init the TP/PP/EP groups per model INSIDE this model's config
        # context: initialize_model_parallel reads get_current_vllm_config() to
        # decide EP-group creation, so a dense model's groups are wrong for a
        # subsequent MoE model (its expert-parallel group would be missing).
        # This is why the sweep preflight runs one model per process; we
        # tear down and rebuild the groups instead.
        if model_parallel_is_initialized():
            destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=1,
                                  pipeline_model_parallel_size=1)
        with torch.device("meta"):
            return initialize_model(vllm_config)


def normalized_overlap(model) -> tuple[int, int]:
    """Run the REAL two-pass adapter-key normalization against the live meta
    tree and return (n_overlap, n_total). Pass 1 = `normalize_lora_key`
    (static, what load_adapter_from_pt does); pass 2 = the live-tree
    `_renormalize_lora_sd_for_model`. Overlap > 0 means the trainer's keys
    still resolve onto this model's modules."""
    from vllm.tokenformer.adapter_format import normalize_lora_key
    from vllm.tokenformer.hybrid_adapter_manager import PTWorkerLoRAManager

    synth = [f"model.layers.0.{block}.{leaf}.lora_A.default.weight"
             for block, leaves in LORA_TARGETS.items() for leaf in leaves]
    pass1 = {normalize_lora_key(k): None for k in synth}

    class _AM:
        def __init__(self, m):
            self.model = m

    class _Stand(PTWorkerLoRAManager):
        def __init__(self, m):
            self._adapter_manager = _AM(m)

    pass2 = _Stand(model)._renormalize_lora_sd_for_model(pass1)

    def module_path(k: str) -> str:
        for tail in (".lora_A.weight", ".lora_B.weight"):
            if k.endswith(tail):
                return k[:-len(tail)]
        return k

    lora_paths = {module_path(k) for k in pass2}
    base = {n for n, _ in model.named_modules()}
    return len(lora_paths & base), len(lora_paths)

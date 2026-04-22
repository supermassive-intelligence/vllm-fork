# Hybrid LoRA + Tokenformer Adapters

Design doc for enabling LoRA and Tokenformer adapters to coexist on a single
vLLM server.

**Status:** draft
**Scope:** ScalarLM fork (`scalarlm-on-v0.19.0`)

## 1. Problem

Today, "LoRA" and "Tokenformer" are mutually exclusive — the runner's adapter
plumbing is hardwired to create a `TokenformerModelManager`, and the
`enable_lora` flag really means "enable the Tokenformer-flavored manager". A
Tokenformer-only workflow is fine; a LoRA-only workflow accidentally gets a
Tokenformer manager that no-ops most LoRA calls; a mixed workflow is not
expressible.

Users want both:

- **Tokenformer adapter** — MLP-level residual module inserted at load time
  via `TokenformerSurgeon`. One adapter is "active" at a time; base weights
  of selected layers (e.g. embeddings, norms) may be overwritten from the
  adapter's state dict, and the extra `tokenformer_k/v/p` parameters
  contribute a delta on top of the base MLP output. When no adapter is
  loaded, the tokenformer parameters are zeros and the wrapped model is
  behavior-equivalent to the base model.
- **LoRA adapter** — per-request low-rank deltas applied inside linear
  layers (q/k/v/o, gate/up/down, etc.) via Punica kernels. Many adapters can
  coexist, and different requests in the same batch can select different
  adapter slots.

The two mechanisms target different places in the model and have different
per-request semantics. They should compose, not race.

**Unified file format.** Both adapter kinds ship as a single `.pt` file
with a `model_state_dict`, matching the existing Tokenformer loader
(`tokenformer_model_manager.py:27-46`). The *kind* of an adapter (or
whether it's mixed) is determined by the keys present:

- `tokenformer_k`, `tokenformer_v`, `tokenformer_p` (possibly plus base
  weight overrides like `embed_tokens.weight`, `lm_head.weight`) →
  Tokenformer component.
- `lora_A.weight`, `lora_B.weight` (scoped by target module name) →
  LoRA component.
- A single file may contain both. In that case the adapter is "hybrid":
  its Tokenformer portion activates globally and its LoRA portion
  occupies a LoRA slot.

We do not accept HF PEFT's `adapter_model.safetensors` + `adapter_config.json`
layout in v1. If a user wants to run a PEFT-trained LoRA, they convert it
to the `.pt` shape offline.

## 2. Current state

This section is prescriptive — every claim is pinned to a file and line on
this branch so the design stays accurate as code changes.

### 2.1 Manager selection

`vllm/v1/worker/lora_model_runner_mixin.py:32-48`

```python
def load_lora_model(self, model, vllm_config, device):
    if not supports_lora(model):
        raise ValueError(...)
    self.lora_manager = TokenformerModelManager(model=model, device=device)
    return self.lora_manager.model
```

The runner never looks at anything that would distinguish LoRA from
Tokenformer. There is no `enable_tokenformer` flag anywhere in
`vllm/engine/arg_utils.py` or `vllm/config/`. `enable_lora=True`
(arg_utils.py:516) is the only switch, and it currently produces a
Tokenformer manager.

### 2.2 Tokenformer surgeon

`vllm/tokenformer/tokenformer_surgeon.py:166-170`

```python
def _is_adapter_layer(self, layer_name):
    parts = layer_name.split(".")
    if any(part in _NON_LANGUAGE_PATH_COMPONENTS for part in parts):
        return False
    return "mlp" in parts[-1]
```

The surgeon wraps a module as a `TokenformerAdapter` iff its leaf name is
`mlp` (case-sensitive component match) and no ancestor is a vision/audio
tower or multimodal projector. In practice this means *blocks*, not the
individual `q_proj`/`k_proj`/`gate_proj`/`up_proj`/`down_proj` linears that
LoRA targets. The surgeon wraps the block; LoRA replaces the linears
*inside* the block. They operate at different granularity and do not
overlap structurally.

### 2.3 Tokenformer activation

`vllm/tokenformer/tokenformer_model_manager.py:69-107` (`activate_adapter`):

- Snapshots any base weight it is about to overwrite into
  `self.original_tensors`.
- Restores originals, then copies the adapter's tensors (skipping any key
  whose name contains `'lora'`) into the state dict.
- Calls `self.model.load_weights(state_dict.items())` and then
  `process_weights_after_loading(...)`.

The `'lora'` skip on line 97 is defensive — it was added with LoRA
coexistence in mind but never exercised because both paths can't run yet.

`process_weights_after_loading` is the risky bit: it re-runs
quantization prep, fused-tensor setup, etc. If LoRA has already replaced
linears with `*WithLoRA` wrappers and registered them with the Punica
kernel infrastructure, we need to confirm that re-running
`process_weights_after_loading` doesn't invalidate that state.

### 2.4 Warmup path

`vllm/tokenformer/tokenformer_model_manager.py:214-223` (`add_dummy_lora`
is a no-op), and `set_active_adapters` (line 154) now skips unregistered
ids (fix landed in `6529423ba`). LoRA's real `add_dummy_lora` actually
creates a zero-rank dummy and registers it. The two have divergent
warmup semantics that a hybrid manager has to pick between.

### 2.5 Interfaces

Models declare `SupportsLoRA` and `SupportsTokenformer` independently
(`vllm/model_executor/models/interfaces.py`). Gemma3, Gemma4, Llama,
Qwen3 declare both. Neither protocol enumerates target modules — the
target set is decided by the *manager*, not the model.

## 3. Goals & non-goals

Goals:

1. A single server can load and serve both LoRA and Tokenformer adapters.
2. Per-request LoRA selection continues to work unchanged.
3. A Tokenformer adapter is a server-global switch (same semantics as today).
4. When only one type is enabled, behavior and latency are unchanged from today.
5. Tokenformer adapters that don't overwrite base weights (the common case)
   have zero interaction cost with LoRA.

Non-goals (for v1):

- Per-request Tokenformer selection. Tokenformer stays global.
- Stacking multiple Tokenformer adapters. One at a time.
- Automatic adapter-type inference from the request ID. The request has to
  say whether it's LoRA or Tokenformer.
- Changing the Tokenformer adapter file format.

## 4. Design

### 4.1 Shape

Introduce a composite manager that owns a LoRA manager and a Tokenformer
manager and routes calls by adapter kind:

```
HybridAdapterManager
├── LoRAModelManager         (per-request, many slots)
└── TokenformerModelManager  (global, one active)
```

The runner keeps its existing `self.lora_manager` attribute (for backward
compat), but the object it holds is the hybrid when both are enabled.
When only LoRA is enabled, it's a plain `LoRAModelManager`. When only
Tokenformer is enabled, it's today's `TokenformerModelManager`.

### 4.2 Load order

When both are enabled, apply **LoRA first, then Tokenformer**:

1. `LoRAModelManager._create_lora_modules()` replaces targeted linears
   with `*WithLoRA` wrappers. The original linear becomes `base_layer`
   inside the wrapper.
2. `TokenformerSurgeon.insert_adapter_modules()` walks the resulting tree
   and wraps each eligible `mlp` block with a `TokenformerAdapter`. The
   wrapper holds a reference to the (already-LoRA-replaced) MLP block;
   LoRA wrappers live *inside* the Tokenformer wrapper unchanged.

At forward time: request arrives → LoRA layers select per-request A/B →
LoRA produces a delta → the enclosing MLP runs normally → Tokenformer
adds its own delta on top. Order is `base + lora_delta + tokenformer_delta`,
which is what we want: the tokenformer adapter models "what does this
MLP block look like *after* LoRA has already been applied". That matches
the user's intuition ("tokenformer adds no-op augmentation; LoRA adds
per-request deltas on top of what tokenformer produces").

### 4.3 Adapter-kind detection and request routing

Because both kinds share one file format (§1), the kind is a property
of the file's contents, not of the request. Detection happens once at
`add_adapter` time by inspecting `model_state_dict` keys:

```python
def classify_adapter(state_dict: dict[str, Tensor]) -> tuple[bool, bool]:
    """Returns (has_tokenformer, has_lora)."""
    has_tk = any(
        k.endswith("tokenformer_k") or k.endswith("tokenformer_v")
        or k.endswith("tokenformer_p")
        for k in state_dict
    )
    has_lora = any(".lora_A." in k or ".lora_B." in k for k in state_dict)
    return has_tk, has_lora
```

Four possible shapes:

| has_tokenformer | has_lora | Classification |
|---|---|---|
| true | false | pure Tokenformer |
| false | true | pure LoRA |
| true | true | hybrid |
| false | false | reject at load — nothing to activate |

The state dict is **split at load time** into two sub-dicts. The
Tokenformer portion is registered with `TokenformerModelManager`; the
LoRA portion is registered with `LoRAModelManager` (after reshaping
into the internal LoRA weight layout). Both are keyed by the same
`adapter_id`, so a single `set_active_adapter(id)` on the hybrid
manager fans out to both sub-managers.

Request routing is then a property of the *loaded* adapter, not the
request — the client just passes an adapter id. The hybrid manager
looks the id up in its registry and activates whichever components
exist. This is strictly simpler than carrying an `adapter_kind` field
on `LoRARequest` and it's the route we take.

Request side-effect for hybrid adapters: activating a hybrid adapter
because one request references it imposes the Tokenformer portion on
*all* requests in the batch (that's just the nature of global-state
Tokenformer). The adapter author is responsible for making the
Tokenformer portion near-identity for requests that weren't specifically
targeted at it. Document this as a sharp edge in §4.7.

### 4.4 Weight overwrite conflicts

The Tokenformer adapter may overwrite base weights (embeddings, lm_head,
norms). LoRA never overwrites base weights — it stores A/B externally.
So there is no direct state-dict collision. However:

- `process_weights_after_loading` runs across the whole model.
  `*WithLoRA` layers implement the hook (they have to, to set up Punica
  metadata). We need to verify that calling it *after* LoRA adapters are
  already loaded is either a no-op or idempotent. If not, we gate the
  call: only run it over the subset of modules whose state actually
  changed. See §6 (open questions) — needs a targeted test before we can
  choose.

- The `'lora'`-substring skip in `activate_adapter` (line 97) becomes
  load-bearing once hybrid adapters exist: a single `.pt` can now
  legitimately contain both Tokenformer and LoRA tensors, and the
  Tokenformer activation path must *only* replay the Tokenformer
  portion into the base state dict. Tighten the match to
  `.lora_A.` / `.lora_B.` path segments (not substring), and in the
  hybrid manager do the split at `add_adapter` time so `activate_adapter`
  only ever sees a pre-filtered Tokenformer sub-dict — the in-activate
  skip stays as a defense-in-depth assert.

### 4.5 Warmup path

When both are enabled:

- LoRA's real `add_dummy_lora` registers a dummy at slot 0 so cudagraph
  profiling sees the LoRA kernel shapes.
- Tokenformer's `add_dummy_lora` remains a no-op (the surgeon-injected
  wrappers always run, with zero-initialized tokenformer params when no
  adapter is loaded, so warmup already exercises the tokenformer code
  path).

The hybrid's `add_dummy_lora` forwards only to the LoRA manager. The
skip-unregistered-id guard we added in `6529423ba` keeps this safe if a
dummy request ever leaks into the tokenformer path.

### 4.6 Config & CLI

Introduce one new flag:

```
--enable-tokenformer           # default: false
```

And redefine `--enable-lora`:

- `--enable-lora` → LoRA manager only (today's *intended* semantics).
- `--enable-tokenformer` → Tokenformer manager only (today's *actual*
  semantics, renamed).
- Both flags → hybrid manager.
- Neither flag → no adapter manager (existing behavior).

This is a breaking change for anyone currently relying on
`--enable-lora` producing a Tokenformer manager. The v1 rollout ships
with a compatibility shim: if `--enable-lora` is set *and* any adapter
path the server sees looks like a Tokenformer adapter (by the sniffing
rule in §4.3), log a deprecation warning and implicitly set
`--enable-tokenformer` as well. Remove the shim after one release.

### 4.7 Hybrid-adapter semantics (sharp edge)

A hybrid `.pt` binds a Tokenformer portion to a LoRA portion under one
id. When a request picks this id:

- LoRA portion lands in a per-request slot — only that request sees it.
- Tokenformer portion is activated globally — every other request in
  flight (and every subsequent request until the next Tokenformer
  change) runs with that Tokenformer delta too.

Two consequences:

1. Mixing a hybrid adapter with LoRA-only adapters in the same batch is
   fine only if the hybrid's Tokenformer portion is trained to be
   close-to-identity outside of its specific task (the same constraint
   that applies to any shared Tokenformer today).
2. Two different hybrid adapters cannot both be "active" at once. The
   scheduler serializes them the same way it serializes plain
   Tokenformer swaps: requests tagged with hybrid-B wait until the
   current batch using hybrid-A drains, then the Tokenformer portion
   swaps, then the batch runs. See §6 on the swap-latency concern.

Document the constraint in the user-facing guide next to the
`--enable-tokenformer` flag.

## 5. Phased implementation

The work is large; ship it in reviewable chunks.

**Phase 0 — baseline tests (no behavior change).** Land tests that pin
down today's behavior:

- LoRA-only server with a LoRA adapter: confirm `activate_adapter` works.
  (This will *fail* today, which motivates phase 1.)
- Tokenformer-only server with a Tokenformer adapter: confirm activation.
- Warmup run with both `enable_lora=True` and a registered LoRA adapter
  vs. a registered Tokenformer adapter.

**Phase 1 — split the managers.** Stop always instantiating Tokenformer
from `load_lora_model`. Introduce `--enable-tokenformer`, wire the
existing Tokenformer manager behind it, and fix `--enable-lora` to
instantiate `LoRAModelManager` as originally intended. Compatibility
shim described in §4.6. No hybrid yet — the two flags are still
mutually exclusive at this point.

**Phase 2 — hybrid manager.** Add `HybridAdapterManager` and allow both
flags to be set together. Load order per §4.2. Adapter-kind detection
and state-dict splitting per §4.3. Warmup per §4.5. Hybrid-adapter
semantics per §4.7.

**Phase 3 — `process_weights_after_loading` tightening.** Resolve the
open question in §6. Either confirm idempotency or narrow the call
scope.

**Phase 4 — docs, examples, benchmarks.** Recipe update showing a
server with a Tokenformer base adapter *and* several LoRA adapters
handling different request slices.

## 6. Open questions

- **`process_weights_after_loading` idempotency after LoRA.** Does
  re-running it clobber Punica state? Needs a targeted test: load a LoRA
  adapter, run inference, call `process_weights_after_loading` again,
  verify identical outputs. If not idempotent, phase 3 needs to track
  per-module dirty state and scope the call.

- **Quantization interaction.** Tokenformer adapters currently load
  un-quantized float tensors and rely on `process_weights_after_loading`
  to re-quantize. If LoRA-wrapped linears are already present *and*
  quantized, the second-pass quantization needs to skip them. Concretely:
  does `ColumnParallelLinearWithLoRA` expose the raw base weight in a
  way that `process_weights_after_loading` can requantize without going
  through the LoRA path?

- **LoRA key naming convention inside `.pt`.** The sniffing rule in
  §4.3 assumes `.lora_A.` / `.lora_B.` path segments scoped by target
  module (e.g. `model.layers.0.self_attn.q_proj.lora_A.weight`). The
  training side has to emit keys in that shape or the hybrid loader
  can't route them. We should pin the expected layout in a
  `training/adapter-format.md` doc and add a schema check on load so
  malformed `.pt`s fail early with a useful message instead of silently
  skipping.

- **Profiling.** Tokenformer adds ~hundreds of milliseconds when
  swapping adapters (weight reload + process_weights). LoRA swap is
  free. Mixed workloads where Tokenformer selection changes frequently
  will see tail-latency spikes. We should document this and consider an
  LRU pin for the active Tokenformer to reduce thrash.

## 7. Testing

Must-have tests before merging phase 2:

1. **Hybrid activation smoke.** Load 1 Tokenformer + 1 LoRA adapter,
   send a request that uses each, verify both take effect (differentiable
   output vs. base model).
2. **LoRA unaffected by Tokenformer swap.** Activate Tokenformer A,
   send LoRA-tagged requests, activate Tokenformer B, send the *same*
   LoRA-tagged requests, verify only the Tokenformer-attributable part
   of the output changed.
3. **Tokenformer unaffected by LoRA swap.** Inverse of #2.
4. **Warmup stability.** Server boots with both flags, no registered
   adapters; cudagraph profiling completes without the
   "Adapter 1 not found" assert that motivated `6529423ba`.
5. **Malformed adapter.** A `.pt` with neither tokenformer_* nor
   lora_A/lora_B keys is rejected at load with a clear error.
6. **Hybrid adapter.** A `.pt` containing both tokenformer_* and
   lora_A/lora_B keys loads; activating it applies both portions;
   deactivating it restores the base weights *and* frees the LoRA slot.
7. **Split correctness.** After `add_adapter` on a hybrid `.pt`, the
   TokenformerModel's state dict contains only tokenformer tensors and
   base overrides, and the LoRAModel's weights contain only lora_A/B
   tensors — no cross-contamination.

## 8. Alternatives considered

- **Make tokenformer a special case of LoRA.** Rewrite the Tokenformer
  adapter as a LoRA variant and use the LoRA manager for both. Rejected:
  the tokenformer delta is a learned attention block, not a low-rank
  matrix product, and fitting it into LoRA kernels would require
  invasive changes to the Punica layer API. Also loses the "overwrite
  base weights" capability.

- **Always run the Tokenformer surgeon, even for LoRA-only servers.**
  Tokenformer parameters would just sit at zero. Rejected: wraps every
  `mlp` block with an `nn.Module` that does extra work (the zero-init
  params still participate in the computation on line 69–88 of the
  surgeon), slowing LoRA-only servers for no benefit.

- **Two separate manager slots on the runner
  (`self.lora_manager` + `self.tokenformer_manager`).** Mechanically
  simpler than a hybrid manager. Rejected for v1 because every existing
  call site assumes a single `self.lora_manager`; touching all of them
  is a larger change than wrapping them behind one dispatching object.
  Worth revisiting if the hybrid proves awkward.

# ScalarLM adapter `.pt` format

This doc is the contract between trainers and the ScalarLM adapter
loader. It covers both Tokenformer and LoRA adapters, which ship in
the same file format and are distinguished by their key shapes.
Hybrid adapters (both kinds together) are supported.

See `docs/design/hybrid_lora_tokenformer.md` for the end-to-end design
this format feeds.

## File layout

A single `.pt` file with a dict of:

```python
torch.save({
    "model_state_dict": {...},   # tensor state dict (see below)
    "metadata":         {...},   # optional — defaults to {}
}, "adapter.pt")
```

The loader finds the first `*.pt` file in the directory it's pointed
at (alphabetically). Only `model_state_dict` is required; `metadata`
is optional and its absence is not an error.

## Key shapes

Keys in `model_state_dict` are classified by the loader:

- **Tokenformer keys** — any key whose **leaf** is one of
  `tokenformer_k`, `tokenformer_v`, `tokenformer_p`. Example:
  `model.layers.0.mlp.tokenformer_k`. The match is on the leaf name
  only; a key like `some_tokenformer_k_config` does *not* match.

- **LoRA keys** — any key containing `.lora_A.` or `.lora_B.` as a
  path segment. Example:
  `model.layers.0.self_attn.q_proj.lora_A.weight`. The match is on
  the segment (with dots on both sides); a variable named
  `lora_A_config` does *not* match.

- **Base weight overrides** — any other key. These are copied into
  the base model state dict at Tokenformer activation time (e.g. to
  fine-tune `embed_tokens.weight`, `lm_head.weight`, layernorm
  weights, etc.). They're treated as part of the Tokenformer half of
  the adapter.

A file with neither Tokenformer nor LoRA keys is rejected at load.

### LoRA tensor shapes

- `lora_A.weight`: `(rank, in_features)`
- `lora_B.weight`: `(out_features, rank)`

Rank must be consistent across every `lora_A` tensor in the file —
the loader rejects files with mixed ranks.

### LoRA module naming

Keys should follow standard `model.layers.<i>.<module>.lora_A.weight`
naming. Do **not** ship fused `qkv_proj` LoRAs; emit individual
`q_proj`, `k_proj`, `v_proj` LoRAs and the loader will auto-fuse
based on the model's `packed_modules_mapping`.

vLLM's name parser accepts PEFT's `base_model.model.` prefix if
present, but it isn't required.

## Metadata fields

All optional. The loader applies defaults when a field is missing and
logs at WARN level when it does so, so trainers can spot unexpected
defaulting.

| key | type | default | notes |
|---|---|---|---|
| `lora_alpha` | int | `2 * rank` | LoRA scaling numerator. If this is wrong, the delta is silently mis-scaled, so we recommend always setting it. |
| `use_rslora` | bool | `false` | Rank-stabilized LoRA. Changes scaling to `lora_alpha / sqrt(rank)`. |

No `target_modules` field — vLLM figures out packing from the model's
`packed_modules_mapping` at load time.

## Example — hybrid adapter

```python
sd = {
    # Tokenformer half
    "model.layers.0.mlp.tokenformer_k": torch.randn(hidden, kv_size),
    "model.layers.0.mlp.tokenformer_v": torch.randn(hidden, kv_size),
    "model.layers.0.mlp.tokenformer_p": torch.randn(hidden, kv_size),
    # LoRA half
    "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(16, hidden),
    "model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(hidden, 16),
    "model.layers.0.self_attn.k_proj.lora_A.weight": torch.randn(16, hidden),
    "model.layers.0.self_attn.k_proj.lora_B.weight": torch.randn(kv_dim, 16),
    # ... etc
    # Base weight override
    "lm_head.weight": torch.randn(vocab, hidden),
}

torch.save(
    {
        "model_state_dict": sd,
        "metadata": {"lora_alpha": 32, "use_rslora": False},
    },
    "/path/to/adapter_dir/adapter.pt",
)
```

Serve this with both flags set:

```bash
vllm serve <base_model> \
    --enable-lora \
    --enable-tokenformer \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template examples/tool_chat_template_gemma4.jinja
```

Load the adapter at request time with the standard `lora_path` /
`lora_int_id` pair. The server classifies the `.pt` on load and
routes the two halves to the Tokenformer and LoRA sub-managers
respectively. No client-side change is needed.

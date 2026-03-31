# mode_engine

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MESAL--1.2-blue" alt="License: MESAL-1.2">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/Use-Non--Commercial-red" alt="Non-Commercial Use Only">
  </a>
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/CLA-Required-orange" alt="CLA Required">
  </a>
  <a href="https://github.com/sunvir/mode_engine">
    <img src="https://img.shields.io/badge/Source%20Available-Yes-success" alt="Source Available">
  </a>
  <a href="https://github.com/sunvir/mode_engine/stargazers">
    <img src="https://img.shields.io/github/stars/sunvir/mode_engine?style=social" alt="GitHub Stars">
  </a>
</p>


**Mixture of Directed Experts (MoDE) engine** for extending causal language models with externally-gated expert blocks.

`mode_engine` is a standalone PyTorch library that handles:

- Loading HuggingFace causal LMs through a **plugin architecture** — new model families require only a single new file
- Wrapping decoder layers with **vectorized MoE expansion blocks** whose expert routing is controlled externally (per-user, per-session gate indices)
- Growing or removing expansion blocks without disturbing base model weights
- **Batched inference** with per-sequence, per-layer expert routing, supporting both streaming and non-streaming generation

It has no dependency on any web framework. It is used as the ML core of [Polymodels](https://polymodels.net), but can be imported into any Python project.

---

## Installation

Note that this package is not yet available on PyPI. You can install it directly from the repository:

```bash
pip install mode_engine
```

Or install the latest from source:

```bash
pip install git+https://github.com/sunvir/mode_engine.git
```

For local development (editable install):

```bash
pip install -e /path/to/mode_engine
```

---

## Quick start

```python
import torch
from mode_engine import MoDEEngine, ExpansionState, MoDEInferencer, GenerationConfig

engine = MoDEEngine()
device = torch.device("cpu")

# Load a model — plugin is resolved automatically
model, tokenizer, plugin = engine.load_model(
    model_id="Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    quantization_config=None,
    device=device,
)

# Expand layers 0 and 1 with 4 experts each
state = ExpansionState()
state = engine.expand_model(
    model=model,
    plugin=plugin,
    layers_to_expand=[0, 1],
    num_experts=4,
    expansion_state=state,
)
print(state.to_dict())
# {'expanded_layers': [0, 1], 'num_experts_per_layer': {0: 0, 1: 0},
#  'max_experts_per_layer': {0: 4, 1: 4}, 'num_experts': 4}

# Run batched inference
num_layers = len(plugin.get_layers(model))
inferencer = MoDEInferencer(model, tokenizer, device)

# Same expert for all layers: seq-0 → expert 2, seq-1 → expert 0, seq-2 → passthrough
per_seq = torch.tensor([2, 0, -1])
gate_indices = per_seq.unsqueeze(0).expand(num_layers, -1)  # (num_layers, batch_size)

result = inferencer.generate(
    prompts=["Tell me about Paris", "What is quantum computing?", "Hello"],
    gate_indices=gate_indices,
    config=GenerationConfig(max_new_tokens=128, do_sample=True, temperature=0.8),
)
for text in result.texts:
    print(text)

# Remove expansion from layer 0
original_layers = list(plugin.get_layers(model))  # snapshot at load time
removed, state = engine.remove_expansion(
    model=model,
    plugin=plugin,
    layers_to_remove=[0],
    original_layers=original_layers,
    expansion_state=state,
)
```

---

## Architecture

```
mode_engine/
├── engine.py          # MoDEEngine — load_model, expand_model, remove_expansion
├── inference.py       # MoDEInferencer — batched generation, streaming, GenerationConfig
├── layers/
│   ├── expert.py      # MyMLP, VectorizedExternallyGatedMoELayer
│   └── expanded.py    # ExpandedDecoderLayer
└── models/
    ├── plugin.py      # ModelPlugin ABC
    ├── registry.py    # PluginRegistry
    ├── auto.py        # AutoPlugin  (generic fallback, priority 0)
    ├── llama.py       # LlamaPlugin (meta-llama/*, priority 10)
    ├── qwen3.py       # Qwen3Plugin (Qwen/Qwen3-*, priority 10)
    └── gemma3.py      # Gemma3Plugin (google/gemma-*, priority 10)
```

### How gating works

Each `ExpandedDecoderLayer` wraps an original decoder layer and adds a
`VectorizedExternallyGatedMoELayer` residual block. At inference time the
caller passes `gate_indices` — a `(num_layers, batch_size)` integer tensor —
through the model's `forward()` kwargs. Layer *i* reads `gate_indices[i]` to
obtain a `(batch_size,)` vector routing each sequence to its expert.
An index of `-1` is a passthrough (zero residual, base model only).

This shape allows **different experts to be selected per layer**, enabling
fine-grained routing control across the model depth. To use the same expert
for all layers, expand a 1-D tensor: `per_seq.unsqueeze(0).expand(num_layers, -1)`.

The MoDE wrapper classes (`_MoDE_*ForCausalLM`) accept and forward
`gate_indices` so callers do not need to modify HuggingFace internals.

### Expert initialisation

Expert output projections are zero-initialised so the expansion block starts
as an identity. This means loading a fresh expanded model produces the same
outputs as the original — training starts from a stable baseline.

---

## Adding a new model family

Create a file in `mode_engine/models/` and decorate it with
`@PluginRegistry.register`:

```python
# mode_engine/models/mistral.py
from transformers import MistralForCausalLM, AutoTokenizer
from mode_engine.models.plugin import ModelPlugin
from mode_engine.models.registry import PluginRegistry

class _MoDE_MistralForCausalLM(MistralForCausalLM):
    def forward(self, *args, activate_block=0, gate_indices=None, **kwargs):
        return super().forward(*args, activate_block=activate_block,
                               gate_indices=gate_indices, **kwargs)

@PluginRegistry.register
class MistralPlugin(ModelPlugin):
    priority = 10

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "mistral" in model_id.lower()

    @classmethod
    def load(cls, model_id, dtype, quantization_config, device, hf_token=None):
        model = _MoDE_MistralForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config,
            torch_dtype=dtype, token=hf_token,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    @classmethod
    def get_layers(cls, model):
        return model.model.layers

    @classmethod
    def set_layer(cls, model, idx, layer):
        model.model.layers[idx] = layer
```

Then import it in `mode_engine/models/__init__.py` alongside the others.
No other files need to change.

---

## API reference

### `MoDEEngine`

#### `load_model(model_id, dtype, quantization_config, device, hf_token=None)`
Resolves the correct plugin, calls `plugin.load()`, runs `plugin.post_load()`, and returns `(model, tokenizer, plugin_class)`. The returned plugin should be stored so it can be passed to `expand_model` / `remove_expansion` without re-resolving.

#### `expand_model(model, plugin, layers_to_expand, num_experts, expansion_state)`
Wraps each layer in `layers_to_expand` with an `ExpandedDecoderLayer`. If a layer is already expanded and `num_experts` exceeds the current max, the expert list is grown and the block is rebuilt (existing trained weights in lower-index slots are preserved). Modifies `model` in-place; returns the updated `ExpansionState`.

#### `remove_expansion(model, plugin, layers_to_remove, original_layers, expansion_state)`
Restores original layers from the `original_layers` snapshot. Pass an empty list to remove all expanded layers. Returns `(removed_indices, updated_state)`.

### `MoDEInferencer`

Runs the token-generation loop directly (no HuggingFace `.generate()`) so that `gate_indices` is correctly forwarded at every decode step.

#### `__init__(model, tokenizer, device)`
Holds references only — no model-level state is introduced.

#### `generate(prompts, gate_indices, config=None, stream=False)`

| Argument | Type | Description |
|---|---|---|
| `prompts` | `list[str]` | One prompt per sequence |
| `gate_indices` | `Tensor (num_layers, batch_size)` | Per-layer, per-sequence expert routing. Use `-1` for passthrough. |
| `config` | `GenerationConfig` | Sampling settings. Defaults to greedy, 256 tokens. |
| `stream` | `bool` | If `True`, return a generator instead of blocking. |

Returns `InferenceResult` when `stream=False`, or a `Generator[list[str | None], None, None]` when `stream=True`. Each generator step yields one decoded token per sequence; `None` means that sequence has finished.

**Streaming example:**

```python
for token_batch in inferencer.generate(
    prompts=["Tell me about Paris", "What is quantum computing?"],
    gate_indices=torch.tensor([[2, 0]] * num_layers),
    config=GenerationConfig(max_new_tokens=200, do_sample=True, temperature=0.9),
    stream=True,
):
    # token_batch[i] is the new token string for seq i, or None if finished
    for i, tok in enumerate(token_batch):
        if tok is not None:
            print(f"[seq {i}] {tok}", end="", flush=True)
```

### `GenerationConfig`

| Field | Default | Description |
|---|---|---|
| `max_new_tokens` | `256` | Maximum tokens to generate |
| `temperature` | `1.0` | Softmax temperature (1.0 = no change) |
| `top_p` | `1.0` | Nucleus sampling probability mass (1.0 = disabled) |
| `top_k` | `0` | Keep top-k logits (0 = disabled) |
| `do_sample` | `False` | Sample from distribution; `False` = greedy argmax |
| `eos_token_id` | `None` | Stop token; falls back to `tokenizer.eos_token_id` |
| `pad_token_id` | `None` | Pad token for finished sequences; falls back to `tokenizer.pad_token_id` |
| `repetition_penalty` | `1.0` | Penalise repeated tokens (1.0 = disabled) |

### `InferenceResult`

| Field | Type | Description |
|---|---|---|
| `texts` | `list[str]` | Decoded output per sequence |
| `token_ids` | `list[list[int]]` | Raw generated token ids per sequence |
| `prompt_token_counts` | `list[int]` | Prompt token count per sequence |
| `generated_token_counts` | `list[int]` | Generated token count per sequence |

### `ExpansionState`

Lightweight dataclass tracking which layers are expanded:

| Field | Type | Description |
|---|---|---|
| `expanded_layers` | `list[int]` | Layer indices currently wrapped |
| `num_experts_per_layer` | `dict[int, int]` | Loaded experts per layer |
| `max_experts_per_layer` | `dict[int, int]` | Capacity (slots) per layer |
| `num_experts` | `int` | Global expert count set at last expand call |

Call `.to_dict()` to serialise.

### `ModelPlugin` (ABC)

| Method | Required | Description |
|---|---|---|
| `matches(model_id)` | ✅ | Return `True` if this plugin handles the model |
| `load(model_id, dtype, quant_config, device, hf_token)` | ✅ | Load and return `(model, tokenizer)` |
| `get_layers(model)` | ✅ | Return the list of decoder layers |
| `set_layer(model, idx, layer)` | ✅ | Replace layer at index |
| `post_load(model, model_id)` | ➖ | Optional post-load surgery (default: no-op) |

### `PluginRegistry`

| Method | Description |
|---|---|
| `register(plugin_cls)` | Decorator that adds a plugin to the registry |
| `resolve(model_id)` | Return the highest-priority matching plugin |
| `list_plugins()` | Return names of all registered plugins |

---

## Requirements

- Python ≥ 3.11
- torch ≥ 2.9
- transformers ≥ 4.57 (HuggingFace main branch recommended)
- accelerate ≥ 1.12
- bitsandbytes ≥ 0.49 (optional — only needed for 4-bit quantization)

---

## License

Copyright © 2026 Sunvir S. Gujral. All rights reserved.

Free for non-commercial use with attribution. Commercial use requires a
written license from the author. See [LICENSE](./LICENSE) for full terms.



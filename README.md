# mode_engine

**Mixture-of-Decoder-Experts (MoDE) engine** for extending causal language models with externally-gated expert blocks.

`mode_engine` is a standalone PyTorch library that handles:

- Loading HuggingFace causal LMs through a **plugin architecture** — new model families require only a single new file
- Wrapping decoder layers with **vectorized MoE expansion blocks** whose expert routing is controlled externally (per-user, per-session gate indices)
- Growing or removing expansion blocks without disturbing base model weights

It has no dependency on any web framework. It is used as the ML core of [modelstudio](../modelstudio), but can be imported into any Python project.

---

## Installation

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
from mode_engine import MoDEEngine, ExpansionState

engine = MoDEEngine()

# Load a model — plugin is resolved automatically
model, tokenizer, plugin = engine.load_model(
    model_id="Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    quantization_config=None,
    device=torch.device("cpu"),
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
caller passes `gate_indices` — a `(batch_size,)` integer tensor — through
the model's `forward()` kwargs. Each batch item is routed to exactly one
expert; an index of `-1` is a passthrough (zero residual, base model only).

The MoDE wrapper classes (`_MoDE_*ForCausalLM`) accept and forward
`gate_indices` so HuggingFace's generation loop does not need modification.

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

# MoDE Engine  
# Copyright (c) 2026 Sunvir S. Gujral  
# All rights reserved.

# This project is licensed under the MoDE Engine Source‑Available License (MESAL) v1.2.  
# Non‑commercial use only. Commercial use requires a separate license.

# Attribution is required for all uses, modifications, and redistributions.

# For commercial licensing inquiries, contact the Author.

"""MoDEEngine — high-level entry point for model loading and expansion."""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from mode_engine.layers.expert import MyMLP
from mode_engine.layers.expanded import ExpandedDecoderLayer
from mode_engine.models.plugin import ModelPlugin
from mode_engine.models.registry import PluginRegistry

logger = logging.getLogger(__name__)


class ExpansionState:
    """Tracks which layers have been expanded and how many experts each holds."""

    def __init__(self):
        self.expanded_layers: list[int] = []
        self.num_experts_per_layer: dict[int, int] = {}
        self.max_experts_per_layer: dict[int, int] = {}
        self.num_experts: int = 0

    def to_dict(self) -> dict:
        return {
            "expanded_layers": list(self.expanded_layers),
            "num_experts_per_layer": dict(self.num_experts_per_layer),
            "max_experts_per_layer": dict(self.max_experts_per_layer),
            "num_experts": self.num_experts,
        }


class MoDEEngine:
    """Orchestrates model loading, MoE expansion, and layer restoration.

    This class is pure ML — no FastAPI, no Firestore, no server state.
    Callers (e.g. ``ModelService`` in modelstudio) are responsible for
    caching, shared state, and async execution.
    """

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_id: str,
        dtype: torch.dtype,
        quantization_config: Any,
        device: torch.device,
        hf_token: str | None = None,
    ) -> tuple[nn.Module, Any, type[ModelPlugin]]:
        """Load *model_id* from HuggingFace and return ``(model, tokenizer, plugin)``.

        The returned *plugin* should be stored alongside the model so that
        subsequent ``expand_model`` / ``remove_expansion`` calls can use the
        correct layer accessors without re-resolving.
        """
        plugin = PluginRegistry.resolve(model_id)
        logger.info(f"Loading '{model_id}' with {plugin.__name__}")

        model, tokenizer = plugin.load(
            model_id=model_id,
            dtype=dtype,
            quantization_config=quantization_config,
            device=device,
            hf_token=hf_token,
        )
        model = plugin.post_load(model, model_id)
        model.eval()
        logger.info(f"Model '{model_id}' loaded successfully")
        return model, tokenizer, plugin

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand_model(
        self,
        model: nn.Module,
        plugin: type[ModelPlugin],
        layers_to_expand: list[int],
        num_experts: int,
        expansion_state: ExpansionState,
    ) -> ExpansionState:
        """Add or grow MoE expansion blocks on *layers_to_expand*.

        Modifies *model* in-place.  Returns an updated ``ExpansionState``
        (the caller should persist this back to shared model state).

        If a layer is already expanded:
          - If ``num_experts`` ≤ current max: no-op for that layer.
          - If ``num_experts`` > current max: existing experts are preserved,
            new expert slots are appended, and the expansion block is rebuilt.
        """
        layers = plugin.get_layers(model)

        for i, layer in enumerate(layers):
            if i not in layers_to_expand:
                continue

            already_expanded = i in expansion_state.expanded_layers

            if already_expanded:
                current_max = expansion_state.max_experts_per_layer[i]
                if num_experts <= current_max:
                    logger.info(f"Layer {i} already expanded to {current_max} experts — skipping")
                    continue

                # Grow: preserve existing experts, append new ones
                logger.info(
                    f"Layer {i}: increasing max experts {current_max} → {num_experts}"
                )
                in_features = layer.expansion_block.experts[0][0].in_features
                out_features = layer.expansion_block.experts[0][-1].out_features
                new_experts = nn.ModuleList([
                    MyMLP(in_features, out_features, in_features * 2)
                    for _ in range(num_experts - current_max)
                ])
                layer.expansion_block.experts.extend(new_experts)

                # Rebuild ExpandedDecoderLayer with updated expert count
                rebuilt = ExpandedDecoderLayer(
                    layer.original_layer,
                    num_experts=num_experts,
                    layer_number=i,
                )
                rebuilt.expansion_block.to(model.device).to(model.dtype)
                plugin.set_layer(model, i, rebuilt)

                expansion_state.num_experts_per_layer[i] = 0   # existing experts must be reloaded
                expansion_state.max_experts_per_layer[i] = num_experts

            else:
                # Fresh expansion
                logger.info(f"Layer {i}: wrapping with {num_experts}-expert expansion block")
                expanded = ExpandedDecoderLayer(layer, num_experts=num_experts, layer_number=i)
                expanded.expansion_block.to(model.device).to(model.dtype)
                plugin.set_layer(model, i, expanded)

                expansion_state.expanded_layers.append(i)
                expansion_state.num_experts_per_layer[i] = 0
                expansion_state.max_experts_per_layer[i] = num_experts

        expansion_state.num_experts = num_experts
        return expansion_state

    # ------------------------------------------------------------------
    # Expansion removal
    # ------------------------------------------------------------------

    def remove_expansion(
        self,
        model: nn.Module,
        plugin: type[ModelPlugin],
        layers_to_remove: list[int],
        original_layers: list[nn.Module],
        expansion_state: ExpansionState,
    ) -> tuple[list[int], ExpansionState]:
        """Restore original layers, removing their expansion blocks.

        Args:
            model:            The currently loaded model.
            plugin:           Plugin used to set layers.
            layers_to_remove: Indices to restore.  Empty list means remove all.
            original_layers:  Snapshot of layers taken at load time (index = layer idx).
            expansion_state:  Current expansion bookkeeping.

        Returns:
            ``(removed_indices, updated_state)``
        """
        if not expansion_state.expanded_layers:
            return [], expansion_state

        target = (
            [int(i) for i in layers_to_remove if int(i) in expansion_state.expanded_layers]
            if layers_to_remove
            else list(expansion_state.expanded_layers)
        )

        removed: list[int] = []
        num_layers = len(original_layers)

        for idx in sorted(set(target)):
            if idx < 0 or idx >= num_layers:
                continue
            try:
                plugin.set_layer(model, idx, original_layers[idx])
                removed.append(idx)
                logger.info(f"Layer {idx}: expansion removed, original layer restored")
            except Exception as exc:
                logger.warning(f"Layer {idx}: failed to restore original layer — {exc}")

        removed_set = set(removed)
        expansion_state.expanded_layers = [i for i in expansion_state.expanded_layers if i not in removed_set]
        expansion_state.num_experts_per_layer = {
            k: v for k, v in expansion_state.num_experts_per_layer.items() if int(k) not in removed_set
        }
        expansion_state.max_experts_per_layer = {
            k: v for k, v in expansion_state.max_experts_per_layer.items() if int(k) not in removed_set
        }
        if not expansion_state.expanded_layers:
            expansion_state.num_experts = 0

        return removed, expansion_state

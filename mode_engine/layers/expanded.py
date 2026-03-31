# MoDE Engine  
# Copyright (c) 2026 Sunvir S. Gujral  
# All rights reserved.

# This project is licensed under the MoDE Engine Source‑Available License (MESAL) v1.2.  
# Non‑commercial use only. Commercial use requires a separate license.

# Attribution is required for all uses, modifications, and redistributions.

# For commercial licensing inquiries, contact the Author.
"""ExpandedDecoderLayer: wraps an original decoder layer with a MoE expansion block."""
import torch
import torch.nn as nn
import logging

from mode_engine.layers.expert import VectorizedExternallyGatedMoELayer

logger = logging.getLogger(__name__)


class ExpandedDecoderLayer(nn.Module):
    """Wraps a base-model decoder layer with an external MoE expansion block.

    During the forward pass the original layer runs first, then the
    expansion block adds a residual correction selected by ``gate_indices``.
    Output type (tensor vs tuple) is preserved to stay compatible with the
    parent model's layer-iteration logic.
    """

    def __init__(self, original_layer: nn.Module, num_experts: int = 3, layer_number: int = 0):
        super().__init__()
        self.original_layer = original_layer
        self.layer_number = layer_number
        self.num_experts = num_experts

        input_size: int = original_layer.self_attn.q_proj.in_features
        logger.info(f"ExpandedDecoderLayer — layer {layer_number}: input_size={input_size}, num_experts={num_experts}")

        self.expansion_block = VectorizedExternallyGatedMoELayer(
            input_dim=input_size,
            output_dim=input_size,
            num_experts=num_experts,
        )

        # Propagate attention_type if present (required by some models, e.g. Qwen3)
        if hasattr(original_layer, "attention_type"):
            self.attention_type = original_layer.attention_type

    def forward(self, *args, **kwargs):
        outputs = self.original_layer(*args, **kwargs)

        is_tensor = isinstance(outputs, torch.Tensor)
        if is_tensor:
            hidden_states = outputs
            other_outputs = ()
        else:
            hidden_states = outputs[0]
            other_outputs = outputs[1:]

        gate_indices = kwargs.get("gate_indices")
        expanded = self.expansion_block(hidden_states, gate_indices[self.layer_number])

        if is_tensor:
            return expanded
        return (expanded,) + other_outputs

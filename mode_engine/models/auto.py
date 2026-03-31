# MoDE Engine  
# Copyright (c) 2026 Sunvir S. Gujral  
# All rights reserved.

# This project is licensed under the MoDE Engine Source‑Available License (MESAL) v1.2.  
# Non‑commercial use only. Commercial use requires a separate license.

# Attribution is required for all uses, modifications, and redistributions.

# For commercial licensing inquiries, contact the Author.
"""AutoPlugin — generic fallback for any model not matched by a specific plugin."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from mode_engine.models.plugin import ModelPlugin
from mode_engine.models.registry import PluginRegistry


class _MoDE_AutoModelForCausalLM(AutoModelForCausalLM):
    """AutoModelForCausalLM extended to accept MoDE gate kwargs."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        activate_block: Optional[int] = 0,
        gate_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            activate_block=activate_block,
            gate_indices=gate_indices,
            **kwargs,
        )


@PluginRegistry.register
class AutoPlugin(ModelPlugin):
    """Generic fallback plugin — works with any HuggingFace causal LM."""

    priority = 0  # lowest priority; only used when nothing else matches

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return True  # catches everything

    @classmethod
    def load(
        cls,
        model_id: str,
        dtype: torch.dtype,
        quantization_config: Any,
        device: torch.device,
        hf_token: str | None = None,
    ) -> tuple[nn.Module, Any]:
        model = _MoDE_AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            token=hf_token,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    @classmethod
    def get_layers(cls, model: nn.Module) -> list[nn.Module]:
        return model.model.layers

    @classmethod
    def set_layer(cls, model: nn.Module, idx: int, layer: nn.Module) -> None:
        model.model.layers[idx] = layer

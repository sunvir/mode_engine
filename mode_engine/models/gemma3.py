# MoDE Engine  
# Copyright (c) 2026 Sunvir S. Gujral  
# All rights reserved.

# This project is licensed under the MoDE Engine Source‑Available License (MESAL) v1.2.  
# Non‑commercial use only. Commercial use requires a separate license.

# Attribution is required for all uses, modifications, and redistributions.

# For commercial licensing inquiries, contact the Author.
"""Gemma3Plugin — handles google/gemma-3-* model families."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Gemma3ForCausalLM, AutoTokenizer

from mode_engine.models.plugin import ModelPlugin
from mode_engine.models.registry import PluginRegistry


class _MoDE_Gemma3ForCausalLM(Gemma3ForCausalLM):
    """Gemma3ForCausalLM extended to accept MoDE gate kwargs."""

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
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            activate_block=activate_block,
            gate_indices=gate_indices,
            **kwargs,
        )


@PluginRegistry.register
class Gemma3Plugin(ModelPlugin):
    """Plugin for Gemma3 model families (google/gemma-3-*).

    Gemma3 requires a post-load class swap: the model is first loaded
    via ``AutoModelForCausalLM`` (which handles the config correctly)
    and then its internals are transplanted into ``_MoDE_Gemma3ForCausalLM``
    so that the MoDE gate kwargs are accepted.
    """

    priority = 10

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "gemma" in model_id.lower()

    @classmethod
    def load(
        cls,
        model_id: str,
        dtype: torch.dtype,
        quantization_config: Any,
        device: torch.device,
        hf_token: str | None = None,
    ) -> tuple[nn.Module, Any]:
        # Load via AutoModel — Gemma3 config is handled correctly this way
        model = AutoModelForCausalLM.from_pretrained(
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
    def post_load(cls, model: nn.Module, model_id: str) -> nn.Module:
        """Swap the loaded AutoModel into a MoDE_Gemma3ForCausalLM shell."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Gemma3Plugin.post_load: swapping to _MoDE_Gemma3ForCausalLM")
        torch.set_default_dtype(model.config.torch_dtype or torch.bfloat16)
        new_model = _MoDE_Gemma3ForCausalLM(model.config)
        new_model.lm_head = model.lm_head
        new_model.model = model.model
        new_model.to(model.device)
        return new_model

    @classmethod
    def get_layers(cls, model: nn.Module) -> list[nn.Module]:
        return model.model.layers

    @classmethod
    def set_layer(cls, model: nn.Module, idx: int, layer: nn.Module) -> None:
        model.model.layers[idx] = layer

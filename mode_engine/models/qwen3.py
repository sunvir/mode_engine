"""Qwen3Plugin — handles Qwen/Qwen3-* model families."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM, AutoTokenizer

from mode_engine.models.plugin import ModelPlugin
from mode_engine.models.registry import PluginRegistry


class _MoDE_Qwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3ForCausalLM extended to accept MoDE gate kwargs."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep: int = 0,
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
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            activate_block=activate_block,
            gate_indices=gate_indices,
            **kwargs,
        )


@PluginRegistry.register
class Qwen3Plugin(ModelPlugin):
    """Plugin for Qwen3 model families (Qwen/Qwen3-*)."""

    priority = 10

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return "qwen3" in model_id.lower()

    @classmethod
    def load(
        cls,
        model_id: str,
        dtype: torch.dtype,
        quantization_config: Any,
        device: torch.device,
        hf_token: str | None = None,
    ) -> tuple[nn.Module, Any]:
        model = _MoDE_Qwen3ForCausalLM.from_pretrained(
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

# MoDE Engine  
# Copyright (c) 2026 Sunvir S. Gujral  
# All rights reserved.

# This project is licensed under the MoDE Engine Source‑Available License (MESAL) v1.2.  
# Non‑commercial use only. Commercial use requires a separate license.

# Attribution is required for all uses, modifications, and redistributions.

# For commercial licensing inquiries, contact the Author.
"""ModelPlugin — abstract base class every model plugin must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class ModelPlugin(ABC):
    """Abstract base class for per-model-family loading and layer access.

    Subclasses handle model-specific details (which ``from_pretrained``
    class to use, where the decoder layers live, any post-load surgery)
    so that the rest of the engine stays architecture-agnostic.

    Registration
    ------------
    Decorate a concrete subclass with ``@PluginRegistry.register`` to
    make it discoverable::

        @PluginRegistry.register
        class MyModelPlugin(ModelPlugin):
            priority = 10
            ...

    Resolution
    ----------
    ``PluginRegistry.resolve(model_id)`` iterates registered plugins,
    calls ``matches()``, and returns the highest-priority match.
    Falls back to ``AutoPlugin`` when nothing matches.
    """

    #: Higher value wins when multiple plugins match the same model_id.
    priority: int = 0

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def matches(cls, model_id: str) -> bool:
        """Return True if this plugin should handle *model_id*."""

    @classmethod
    @abstractmethod
    def load(
        cls,
        model_id: str,
        dtype: torch.dtype,
        quantization_config: Any,
        device: torch.device,
        hf_token: str | None = None,
    ) -> tuple[nn.Module, Any]:
        """Download/load the model and tokenizer.

        Returns
        -------
        (model, tokenizer)
        """

    @classmethod
    @abstractmethod
    def get_layers(cls, model: nn.Module) -> list[nn.Module]:
        """Return the list of decoder layers for *model*."""

    @classmethod
    @abstractmethod
    def set_layer(cls, model: nn.Module, idx: int, layer: nn.Module) -> None:
        """Replace the decoder layer at *idx* with *layer*."""

    # ------------------------------------------------------------------
    # Optional hook
    # ------------------------------------------------------------------

    @classmethod
    def post_load(cls, model: nn.Module, model_id: str) -> nn.Module:
        """Optional post-load surgery (e.g. class swap for Gemma3).

        The default implementation is a no-op.
        """
        return model

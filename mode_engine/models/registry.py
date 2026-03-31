# MoDE Engine  
# Copyright (c) 2026 Sunvir S. Gujral  
# All rights reserved.

# This project is licensed under the MoDE Engine Source‑Available License (MESAL) v1.2.  
# Non‑commercial use only. Commercial use requires a separate license.

# Attribution is required for all uses, modifications, and redistributions.

# For commercial licensing inquiries, contact the Author.
"""PluginRegistry — discovers and resolves ModelPlugin subclasses."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mode_engine.models.plugin import ModelPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry that maps model IDs to the appropriate ModelPlugin.

    Usage
    -----
    Register a plugin (typically at module import time)::

        @PluginRegistry.register
        class Qwen3Plugin(ModelPlugin):
            ...

    Resolve the best plugin for a given model ID::

        plugin = PluginRegistry.resolve("Qwen/Qwen3-0.6B")
    """

    _plugins: list[type[ModelPlugin]] = []

    @classmethod
    def register(cls, plugin_cls: type[ModelPlugin]) -> type[ModelPlugin]:
        """Register *plugin_cls* and return it unchanged (works as a decorator)."""
        cls._plugins.append(plugin_cls)
        logger.debug(f"Registered model plugin: {plugin_cls.__name__} (priority={plugin_cls.priority})")
        return plugin_cls

    @classmethod
    def resolve(cls, model_id: str) -> type[ModelPlugin]:
        """Return the highest-priority plugin that matches *model_id*.

        Falls back to ``AutoPlugin`` when no registered plugin matches.
        """
        candidates = [p for p in cls._plugins if p.matches(model_id)]
        if not candidates:
            from mode_engine.models.auto import AutoPlugin
            logger.info(f"No specific plugin found for '{model_id}', falling back to AutoPlugin")
            return AutoPlugin
        best = max(candidates, key=lambda p: p.priority)
        logger.info(f"Resolved plugin '{best.__name__}' for model '{model_id}'")
        return best

    @classmethod
    def list_plugins(cls) -> list[str]:
        """Return the names of all registered plugins (for diagnostics)."""
        return [p.__name__ for p in cls._plugins]

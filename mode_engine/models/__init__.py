# MoDE Engine  
# Copyright (c) 2026 Sunvir S. Gujral  
# All rights reserved.

# This project is licensed under the MoDE Engine Source‑Available License (MESAL) v1.2.  
# Non‑commercial use only. Commercial use requires a separate license.

# Attribution is required for all uses, modifications, and redistributions.

# For commercial licensing inquiries, contact the Author.
"""Model plugins — import all plugins here so they self-register."""
from mode_engine.models.plugin import ModelPlugin
from mode_engine.models.registry import PluginRegistry

# Import plugins to trigger @PluginRegistry.register decorators.
# Order matters only for equal-priority ties; higher-specificity plugins
# should be imported before the AutoPlugin fallback.
from mode_engine.models.llama import LlamaPlugin
from mode_engine.models.qwen3 import Qwen3Plugin
from mode_engine.models.gemma3 import Gemma3Plugin
from mode_engine.models.auto import AutoPlugin

__all__ = [
    "ModelPlugin",
    "PluginRegistry",
    "LlamaPlugin",
    "Qwen3Plugin",
    "Gemma3Plugin",
    "AutoPlugin",
]

# MoDE Engine  
# Copyright (c) 2026 Sunvir S. Gujral  
# All rights reserved.

# This project is licensed under the MoDE Engine Source‑Available License (MESAL) v1.2.  
# Non‑commercial use only. Commercial use requires a separate license.

# Attribution is required for all uses, modifications, and redistributions.

# For commercial licensing inquiries, contact the Author.
"""mode_engine — Mixture of Directed Experts (MoDE) engine.

Public API
----------
    from mode_engine import MoDEEngine, ExpansionState
    from mode_engine import MoDEInferencer, GenerationConfig, InferenceResult
    from mode_engine.models import PluginRegistry, ModelPlugin
    from mode_engine.layers import ExpandedDecoderLayer, MyMLP
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mode_engine")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from mode_engine.engine import MoDEEngine, ExpansionState
from mode_engine.inference import MoDEInferencer, GenerationConfig, InferenceResult
# Importing models triggers plugin self-registration
from mode_engine import models  # noqa: F401

__all__ = [
    "__version__",
    "MoDEEngine",
    "ExpansionState",
    "MoDEInferencer",
    "GenerationConfig",
    "InferenceResult",
]

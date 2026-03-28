"""mode_engine — Mixture of Directed Experts (MoDE) engine.

Public API
----------
    from mode_engine import MoDEEngine, ExpansionState
    from mode_engine.models import PluginRegistry, ModelPlugin
    from mode_engine.layers import ExpandedDecoderLayer, MyMLP
"""
from mode_engine.engine import MoDEEngine, ExpansionState
# Importing models triggers plugin self-registration
from mode_engine import models  # noqa: F401

__all__ = ["MoDEEngine", "ExpansionState"]

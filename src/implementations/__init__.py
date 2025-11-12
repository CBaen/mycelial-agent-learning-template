"""
Reference Implementations Module

This module provides simple, production-ready reference implementations of the
core MAE engines: FRL, VDN, and HAVEN.

Use these implementations as:
1. Starting points for your own custom implementations
2. Learning references to understand the interfaces
3. Functional engines for prototyping and testing

For production systems with advanced requirements, extend these implementations
or create your own from scratch using the abstract base classes in src/core/.
"""

from .simple_frl import SimpleFRL, create_frl_engine
from .simple_vdn import SimpleVDN, create_vdn_engine
from .simple_haven import SimpleHAVEN, create_haven_coordinator

__all__ = [
    # FRL
    "SimpleFRL",
    "create_frl_engine",

    # VDN
    "SimpleVDN",
    "create_vdn_engine",

    # HAVEN
    "SimpleHAVEN",
    "create_haven_coordinator",
]

__version__ = "1.0.0"

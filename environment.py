"""
Disaster Relief Coordination — OpenEnv Environment (root shim)
================================================================
Thin wrapper that imports from src/env/environment.py.
This file exists at the repo root for OpenEnv submission compatibility.
"""

from src.env.environment import DisasterReliefEnv

__all__ = ["DisasterReliefEnv"]

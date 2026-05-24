"""
DACPO (Directed Point Cloud Orientation) Inference Module

This module implements the DACPO algorithm for unifying normal orientations
in oriented point clouds using a trained PTv3 MaskDecoder model.
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies and missing modules during development
def __getattr__(name):
    if name == "FlipOptimizer":
        from .optimization import FlipOptimizer
        return FlipOptimizer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "FlipOptimizer",
    
]

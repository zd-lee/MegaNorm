__version__ = "0.1.0"


def __getattr__(name):
    if name == "FlipOptimizer":
        from .optimization import FlipOptimizer

        return FlipOptimizer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "FlipOptimizer",
]

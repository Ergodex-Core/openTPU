"""OpenTPU instruction-driven performance model. Predictions are not RTL measurements."""
from .config import Architecture, MemoryConfig, Unsupported
from .engine import Simulator

__all__ = ['Architecture', 'MemoryConfig', 'Unsupported', 'Simulator']

# EmbedSim — Embedded Block-Diagram Simulation Framework
# Lightweight, float32-first simulation for 32-bit embedded platforms.
#
# Usage:
#   from embedsim import EmbedSim, VectorSignal, VectorGain, ...
#
# Precision:
#   All blocks default to float32. Override per-block with dtype=np.float64.
#   Change embedsim.core_blocks.DEFAULT_DTYPE = np.float64 to switch globally.

from .core_blocks import *
from .source_blocks import *
from .processing_blocks import *
from .dynamic_blocks import *
from .simulation_engine import *
try:
    from .fmu_blocks import *
except ImportError:
    pass  # fmpy not installed - FMU blocks unavailable

__all__ = []
__version__ = '1.0.0'
__author__ = 'EmbedSim Framework'

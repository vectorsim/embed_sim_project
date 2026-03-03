# EmbedSim — Embedded Block-Diagram Simulation Framework
# Lightweight, float32-first simulation for 32-bit embedded platforms.
#
# Usage:
#   from embedsim import EmbedSim, VectorSignal, VectorGain, ...
#
# Precision:
#   All blocks default to float32. Override per-block with dtype=np.float64.
#   Change embedsim.core_blocks.DEFAULT_DTYPE = np.float64 to switch globally.


# Version info
__version__ = '1.0.0'
__author__ = 'EmbedSim Framework'

# Define __all__ to control what's exported with "from embedsim import *"
__all__ = [
    # Core
    'ThreePhaseProcessorSimBlock'

]

# Note: There are two VectorDelay classes (in processing_blocks and simulation_engine)
# The one from processing_blocks is exported by default as it's more commonly used
# If you need the LoopBreaker version, import it directly:
# from embedsim.simulation_engine import VectorDelay as LoopBreakingDelay
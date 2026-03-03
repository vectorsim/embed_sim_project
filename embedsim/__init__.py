# EmbedSim — Embedded Block-Diagram Simulation Framework
# Lightweight, float32-first simulation for 32-bit embedded platforms.
#
# Usage:
#   from embedsim import EmbedSim, VectorSignal, VectorGain, ...
#
# Precision:
#   All blocks default to float32. Override per-block with dtype=np.float64.
#   Change embedsim.core_blocks.DEFAULT_DTYPE = np.float64 to switch globally.

# Core blocks
from .core_blocks import (
    DEFAULT_DTYPE,
    VectorSignal,
    VectorBlock,
    validate_vector_dimension,
    validate_inputs_exist
)

# Source blocks
from .source_blocks import (
    VectorConstant,
    VectorStep,
    ThreePhaseGenerator,
    SinusoidalGenerator,
    VectorRamp
)

# Processing blocks
from .processing_blocks import (
    VectorGain,
    VectorSum,
    VectorDelay,
    VectorProduct,
    VectorAbs,
    VectorSaturation
)

# Dynamic blocks
from .dynamic_blocks import (
    VectorIntegrator,
    StateSpaceBlock,
    TransferFunctionBlock,
    VectorEnd
)

# Simulation engine
from .simulation_engine import (
    LoopBreaker,
    VectorDelay as SimVectorDelay,  # Note: VectorDelay is also in processing_blocks
    VectorScope,
    EmbedSim,
    traverse_blocks_from_sinks_with_loops,
    ODESolver
)

# Code generator
from .code_generator import (
    SimBlockBase,
    CodeGenStart,
    CodeGenEnd,
    MCUTarget
)

# Script blocks (optional)
try:
    from .script_blocks import ScriptBlock

    _has_script_blocks = True
except ImportError:
    _has_script_blocks = False

# FMU blocks (optional)
try:
    from .fmu_blocks import FMUBlock

    _has_fmu_blocks = True
except ImportError:
    _has_fmu_blocks = False

# Version info
__version__ = '1.0.0'
__author__ = 'EmbedSim Framework'

# Define __all__ to control what's exported with "from embedsim import *"
__all__ = [
    # Core
    'DEFAULT_DTYPE',
    'VectorSignal',
    'VectorBlock',
    'validate_vector_dimension',
    'validate_inputs_exist',

    # Sources
    'VectorConstant',
    'VectorStep',
    'ThreePhaseGenerator',
    'SinusoidalGenerator',
    'VectorRamp',

    # Processing
    'VectorGain',
    'VectorSum',
    'VectorDelay',  # From processing_blocks
    'VectorProduct',
    'VectorAbs',
    'VectorSaturation',

    # Dynamic
    'VectorIntegrator',
    'StateSpaceBlock',
    'TransferFunctionBlock',
    'VectorEnd',

    # Simulation Engine
    'LoopBreaker',
    'VectorScope',
    'EmbedSim',
    'traverse_blocks_from_sinks_with_loops',
    'ODESolver',

    # Code Generation
    'SimBlockBase',
    'CodeGenStart',
    'CodeGenEnd',
    'MCUTarget',

    # Plot Helper
    'PlotHelper',
    'create_plotter'
]

# Add optional modules to __all__ if available
if _has_script_blocks:
    __all__.append('ScriptBlock')

if _has_fmu_blocks:
    __all__.append('FMUBlock')

# Note: There are two VectorDelay classes (in processing_blocks and simulation_engine)
# The one from processing_blocks is exported by default as it's more commonly used
# If you need the LoopBreaker version, import it directly:
# from embedsim.simulation_engine import VectorDelay as LoopBreakingDelay
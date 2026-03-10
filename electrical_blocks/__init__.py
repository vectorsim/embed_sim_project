"""
electrical_blocks
=================

EmbedSim electrical domain blocks for PMSM FOC simulation and C code generation.

Includes
--------
    SMCBlock                  — Sliding Mode Controller (d-q axis inner loop)
    SpeedPIBlock              — Speed PI controller (outer loop, RK4-compatible)
    SVPWMBlock                — Space Vector PWM modulator
    CoordinateTransformBlocks — Clarke / Park / inverse transforms
    SMOBlock                  — Sliding Mode Observer (sensorless estimation)
    PMSM_Motor_WithSensorsBlock — FMU-backed PMSM motor with sensor outputs
    PYXInspector              — Introspects .pyx files for CodeGen attribute population

Each block derives from SimBlockBase (VectorBlock) and supports:
    - Pure Python backend  (always available)
    - Compiled Cython/C backend  (use_c_backend=True, requires build step)
    - CodeGen attributes (PYX_FILE, step_func, state_struct, C_SOURCES, C_HEADERS)
      consumed by embedsim.code_generator.CodeGenerator to emit embedsim_loop.c

Quick start
-----------
    from electrical_blocks.smc_block        import SMCBlock
    from electrical_blocks.speed_pi_block   import SpeedPIBlock
    from electrical_blocks.svpwm_block      import SVPWMBlock
    from electrical_blocks.smo_block        import SMOBlock
    from electrical_blocks.coordinate_transform_blocks import (
        ClarkeTransformBlock, ParkTransformBlock,
        InverseParkTransformBlock, InverseClarkeTransformBlock,
    )
    from electrical_blocks.PMSM_Motor_WithSensorsBlock import PMSM_Motor_WithSensorsBlock
    from electrical_blocks.pyx_inspector    import PYXInspector

Build C backends (Windows)
--------------------------
    cd electrical_blocks/c_src
    python setup_smc.py            build_ext --inplace
    python setup_smo.py            build_ext --inplace
    python setup_speed_pi.py       build_ext --inplace
    python setup_svpwm.py          build_ext --inplace
    python setup_coordinate_transform.py build_ext --inplace

Or from project root:
    build_all.bat

Author  : EmbedSim Framework
Version : 2.0.0
"""

# ── Lazy imports — only pull in what is available ────────────────────────────
# This prevents hard import failures when optional C backends are not compiled.

from importlib import import_module as _imp
import warnings as _warnings

def _safe_import(module: str, symbol: str):
    try:
        return getattr(_imp(module), symbol)
    except Exception as e:
        _warnings.warn(f"electrical_blocks: could not import {symbol} from {module}: {e}")
        return None


# Core simulation blocks
SMCBlock        = _safe_import("electrical_blocks.smc_block",       "SMCBlock")
SpeedPIBlock    = _safe_import("electrical_blocks.speed_pi_block",  "SpeedPIBlock")
SVPWMBlock      = _safe_import("electrical_blocks.svpwm_block",     "SVPWMBlock")
SMOBlock        = _safe_import("electrical_blocks.smo_block",       "SMOBlock")

# Coordinate transforms
ClarkeTransformBlock        = _safe_import("electrical_blocks.coordinate_transform_blocks", "ClarkeTransformBlock")
ParkTransformBlock          = _safe_import("electrical_blocks.coordinate_transform_blocks", "ParkTransformBlock")
InverseParkTransformBlock   = _safe_import("electrical_blocks.coordinate_transform_blocks", "InverseParkTransformBlock")
InverseClarkeTransformBlock = _safe_import("electrical_blocks.coordinate_transform_blocks", "InverseClarkeTransformBlock")

# FMU-backed motor block
PMSM_Motor_WithSensorsBlock = _safe_import("electrical_blocks.PMSM_Motor_WithSensorsBlock", "PMSM_Motor_WithSensorsBlock")

# CodeGen inspector
PYXInspector = _safe_import("electrical_blocks.pyx_inspector", "PYXInspector")


__all__ = [
    "SMCBlock",
    "SpeedPIBlock",
    "SVPWMBlock",
    "SMOBlock",
    "ClarkeTransformBlock",
    "ParkTransformBlock",
    "InverseParkTransformBlock",
    "InverseClarkeTransformBlock",
    "PMSM_Motor_WithSensorsBlock",
    "PYXInspector",
]

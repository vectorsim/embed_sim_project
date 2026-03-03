"""
pmsm_blocks — PMSM Field-Oriented Control Block Library
========================================================

A self-contained library of EmbedSim SimBlockBase blocks for Permanent
Magnet Synchronous Motor (PMSM) simulation and Field-Oriented Control (FOC).

Every block supports the dual Python / C backend:
    block = SomeBlock("name", use_c_backend=False)   # pure Python (default)
    block = SomeBlock("name", use_c_backend=True)    # compiled C via Cython

Blocks
------
    Motor model:
        PMSMMotorBlock          — 4-state dq-frame motor (id, iq, ω, θ)

    Transformations:
        ClarkeTransformBlock    — abc → αβ (power-invariant)
        InvClarkeTransformBlock — αβ → abc
        ParkTransformBlock      — αβ → dq
        InvParkTransformBlock   — dq → αβ

    Controllers:
        PIControllerBlock       — scalar PI with anti-windup

    Utilities:
        VectorCombineBlock      — merge N scalar signals into one vector
        RecordingSinkBlock      — extended sink with statistics and RMS

Usage
-----
    from pmsm_blocks import (
        PMSMMotorBlock,
        ClarkeTransformBlock,
        ParkTransformBlock,
        InvParkTransformBlock,
        PIControllerBlock,
        VectorCombineBlock,
        RecordingSinkBlock,
    )
"""

from .motor_block       import PMSMMotorBlock
from .transform_blocks  import (
    ClarkeTransformBlock,
    InvClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)
from .pi_controller     import PIControllerBlock
from .utility_blocks    import VectorCombineBlock, RecordingSinkBlock, SignalExtractBlock, ExtractDelay

__all__ = [
    "PMSMMotorBlock",
    "ClarkeTransformBlock",
    "InvClarkeTransformBlock",
    "ParkTransformBlock",
    "InvParkTransformBlock",
    "PIControllerBlock",
    "VectorCombineBlock",
    "RecordingSinkBlock",
    "SignalExtractBlock",
    "ExtractDelay",
]

__version__ = "1.0.0"
__author__  = "EmbedSim / ControlForge"

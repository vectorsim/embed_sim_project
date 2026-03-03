"""
transform_blocks.py
===================

Clarke / Park transformation blocks for PMSM FOC.

All blocks derive from SimBlockBase (→ VectorBlock) and support the
dual Python / C backend.

Blocks
------
    ClarkeTransformBlock    — abc → αβ  (power-invariant, 3→2)
    InvClarkeTransformBlock — αβ → abc  (2→3)
    ParkTransformBlock      — αβ → dq   (angle from port 1)
    InvParkTransformBlock   — dq → αβ   (angle from port 1)

Signal conventions
------------------
    ClarkeTransformBlock
        port 0: [ia, ib, ic]
        output: [α, β]

    InvClarkeTransformBlock
        port 0: [α, β]
        output: [ia, ib, ic]

    ParkTransformBlock
        port 0: [α, β]
        port 1: [θ]          (rotor angle in radians)
        output: [d, q]

    InvParkTransformBlock
        port 0: [d, q]
        port 1: [θ]
        output: [α, β]

C backend stubs
---------------
    clarke_transform_compute  (in, out)
    inv_clarke_transform_compute
    park_transform_compute
    inv_park_transform_compute

    All accept the same struct pattern:
        InputSignals  — relevant signal + theta where needed
        OutputSignals — result vector
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional

from ._path_utils import setup_embedsim_path
setup_embedsim_path()

from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks import VectorSignal

_TWO_THIRDS = 2.0 / 3.0
_ONE_THIRD  = 1.0 / 3.0
_SQRT3      = np.sqrt(3.0)
_PI2_3      = 2.0 * np.pi / 3.0


# ==============================================================================
# Clarke   abc → αβ
# ==============================================================================

class ClarkeTransformBlock(SimBlockBase):
    """
    Power-invariant Clarke transform.

        α = (2ia − ib − ic) / 3
        β = (ib − ic) / √3

    Input port 0 : [ia, ib, ic]
    Output       : [α, β]
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.is_dynamic  = False
        self.vector_size = 2
        if use_c_backend:
            self._load_wrapper()

    # -- Python ----------------------------------------------------------------

    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        if not input_values or len(input_values[0].value) < 3:
            self.output = VectorSignal([0.0, 0.0], self.name, dtype=self.dtype)
            return self.output

        ia, ib, ic = (float(v) for v in input_values[0].value[:3])
        alpha = _TWO_THIRDS * ia - _ONE_THIRD * ib - _ONE_THIRD * ic
        beta  = (ib - ic) / _SQRT3

        self.output = VectorSignal(
            np.array([alpha, beta], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    # -- C backend -------------------------------------------------------------

    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        u = np.zeros(3, dtype=np.float32)
        if input_values and len(input_values[0].value) >= 3:
            u[:3] = input_values[0].value[:3]
        self._wrapper.set_inputs(u)
        self._wrapper.compute()
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def _load_wrapper(self) -> None:
        try:
            from transforms_wrapper import ClarkeTransformWrapper
            self._wrapper = ClarkeTransformWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'transforms_wrapper' not found.\n"
                "Generate with CodeGenEnd.generate_pyx_stub() and compile."
            )

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return f"ClarkeTransformBlock('{self.name}', backend={be})"


# ==============================================================================
# Inverse Clarke   αβ → abc
# ==============================================================================

class InvClarkeTransformBlock(SimBlockBase):
    """
    Inverse power-invariant Clarke transform.

        ia =  α
        ib = −α/2 + β·√3/2
        ic = −α/2 − β·√3/2

    Input port 0 : [α, β]
    Output       : [ia, ib, ic]
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.is_dynamic  = False
        self.vector_size = 3
        if use_c_backend:
            self._load_wrapper()

    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        if not input_values or len(input_values[0].value) < 2:
            self.output = VectorSignal([0.0, 0.0, 0.0], self.name, dtype=self.dtype)
            return self.output

        alpha, beta = float(input_values[0].value[0]), float(input_values[0].value[1])
        ia =  alpha
        ib = -0.5 * alpha + 0.5 * _SQRT3 * beta
        ic = -0.5 * alpha - 0.5 * _SQRT3 * beta

        self.output = VectorSignal(
            np.array([ia, ib, ic], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        u = np.zeros(2, dtype=np.float32)
        if input_values and len(input_values[0].value) >= 2:
            u[:2] = input_values[0].value[:2]
        self._wrapper.set_inputs(u)
        self._wrapper.compute()
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def _load_wrapper(self) -> None:
        try:
            from transforms_wrapper import InvClarkeTransformWrapper
            self._wrapper = InvClarkeTransformWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'transforms_wrapper' not found.\n"
                "Generate with CodeGenEnd.generate_pyx_stub() and compile."
            )

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return f"InvClarkeTransformBlock('{self.name}', backend={be})"


# ==============================================================================
# Park   αβ → dq
# ==============================================================================

class ParkTransformBlock(SimBlockBase):
    """
    Park transform (αβ → dq).

        d =  α·cos(θ) + β·sin(θ)
        q = −α·sin(θ) + β·cos(θ)

    Input port 0 : [α, β]
    Input port 1 : [θ]          (rotor electrical angle, radians)
    Output       : [d, q]
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.is_dynamic  = False
        self.vector_size = 2
        if use_c_backend:
            self._load_wrapper()

    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        if not input_values or len(input_values) < 2:
            self.output = VectorSignal([0.0, 0.0], self.name, dtype=self.dtype)
            return self.output

        alpha, beta = float(input_values[0].value[0]), float(input_values[0].value[1])
        theta       = float(input_values[1].value[0])

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        d     =  alpha * cos_t + beta * sin_t
        q     = -alpha * sin_t + beta * cos_t

        self.output = VectorSignal(
            np.array([d, q], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        # flat buffer: [α, β, θ]
        u = np.zeros(3, dtype=np.float32)
        if input_values and len(input_values) >= 1:
            u[0] = input_values[0].value[0]
            u[1] = input_values[0].value[1]
        if input_values and len(input_values) >= 2:
            u[2] = input_values[1].value[0]
        self._wrapper.set_inputs(u)
        self._wrapper.compute()
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def _load_wrapper(self) -> None:
        try:
            from transforms_wrapper import ParkTransformWrapper
            self._wrapper = ParkTransformWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'transforms_wrapper' not found.\n"
                "Generate with CodeGenEnd.generate_pyx_stub() and compile."
            )

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return f"ParkTransformBlock('{self.name}', backend={be})"


# ==============================================================================
# Inverse Park   dq → αβ
# ==============================================================================

class InvParkTransformBlock(SimBlockBase):
    """
    Inverse Park transform (dq → αβ).

        α = d·cos(θ) − q·sin(θ)
        β = d·sin(θ) + q·cos(θ)

    Input port 0 : [d, q]
    Input port 1 : [θ]          (rotor electrical angle, radians)
    Output       : [α, β]
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.is_dynamic  = False
        self.vector_size = 2
        if use_c_backend:
            self._load_wrapper()

    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        if not input_values or len(input_values) < 2:
            self.output = VectorSignal([0.0, 0.0], self.name, dtype=self.dtype)
            return self.output

        d     = float(input_values[0].value[0])
        q     = float(input_values[0].value[1])
        theta = float(input_values[1].value[0])

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        alpha = d * cos_t - q * sin_t
        beta  = d * sin_t + q * cos_t

        self.output = VectorSignal(
            np.array([alpha, beta], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        # flat buffer: [d, q, θ]
        u = np.zeros(3, dtype=np.float32)
        if input_values and len(input_values) >= 1:
            u[0] = input_values[0].value[0]
            u[1] = input_values[0].value[1]
        if input_values and len(input_values) >= 2:
            u[2] = input_values[1].value[0]
        self._wrapper.set_inputs(u)
        self._wrapper.compute()
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def _load_wrapper(self) -> None:
        try:
            from transforms_wrapper import InvParkTransformWrapper
            self._wrapper = InvParkTransformWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'transforms_wrapper' not found.\n"
                "Generate with CodeGenEnd.generate_pyx_stub() and compile."
            )

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return f"InvParkTransformBlock('{self.name}', backend={be})"

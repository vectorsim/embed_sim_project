"""
coordinate_transform_blocks.py
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

from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks import VectorSignal

# Constants
_TWO_THIRDS = 2.0 / 3.0
_ONE_THIRD  = 1.0 / 3.0
_SQRT3      = np.sqrt(3.0)
_INV_SQRT3  = 1.0 / np.sqrt(3.0)
_HALF_SQRT3 = np.sqrt(3.0) / 2.0


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
        beta  = (ib - ic) * _INV_SQRT3

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
            from coordinate_transform_wrapper import ClarkeTransformWrapper
            self._wrapper = ClarkeTransformWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'coordinate_transform_wrapper' not found.\n"
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
    Inverse Clarke transform.

        a = α
        b = -α/2 + (√3/2)β
        c = -α/2 - (√3/2)β

    Input port 0 : [α, β]
    Output       : [a, b, c]
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

        alpha, beta = (float(v) for v in input_values[0].value[:2])

        a = alpha
        b = -0.5 * alpha + _HALF_SQRT3 * beta
        c = -0.5 * alpha - _HALF_SQRT3 * beta

        self.output = VectorSignal(
            np.array([a, b, c], dtype=np.float32), self.name, dtype=self.dtype
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
            from coordinate_transform_wrapper import InvClarkeTransformWrapper
            self._wrapper = InvClarkeTransformWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'coordinate_transform_wrapper' not found.\n"
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
    Park transform (rotating reference frame).

        d =  α·cosθ + β·sinθ
        q = -α·sinθ + β·cosθ

    Input port 0 : [α, β]
    Input port 1 : [θ]        (rotor angle in radians)
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

        # Get α,β from port 0
        if len(input_values[0].value) < 2:
            self.output = VectorSignal([0.0, 0.0], self.name, dtype=self.dtype)
            return self.output

        alpha, beta = (float(v) for v in input_values[0].value[:2])

        # Get θ from port 1
        theta = float(input_values[1].value[0]) if input_values[1].value.size > 0 else 0.0

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        d =  alpha * cos_theta + beta * sin_theta
        q = -alpha * sin_theta + beta * cos_theta

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
        u = np.zeros(2, dtype=np.float32)
        theta = 0.0

        if input_values and len(input_values) >= 2:
            if len(input_values[0].value) >= 2:
                u[:2] = input_values[0].value[:2]
            if input_values[1].value.size > 0:
                theta = float(input_values[1].value[0])

        self._wrapper.set_inputs(u)
        self._wrapper.set_theta(theta)
        self._wrapper.compute()
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def _load_wrapper(self) -> None:
        try:
            from coordinate_transform_wrapper import ParkTransformWrapper
            self._wrapper = ParkTransformWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'coordinate_transform_wrapper' not found.\n"
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
    Inverse Park transform.

        α = d·cosθ - q·sinθ
        β = d·sinθ + q·cosθ

    Input port 0 : [d, q]
    Input port 1 : [θ]        (rotor angle in radians)
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

        # Get d,q from port 0
        if len(input_values[0].value) < 2:
            self.output = VectorSignal([0.0, 0.0], self.name, dtype=self.dtype)
            return self.output

        d, q = (float(v) for v in input_values[0].value[:2])

        # Get θ from port 1
        theta = float(input_values[1].value[0]) if input_values[1].value.size > 0 else 0.0

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        alpha = d * cos_theta - q * sin_theta
        beta  = d * sin_theta + q * cos_theta

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
        u = np.zeros(2, dtype=np.float32)
        theta = 0.0

        if input_values and len(input_values) >= 2:
            if len(input_values[0].value) >= 2:
                u[:2] = input_values[0].value[:2]
            if input_values[1].value.size > 0:
                theta = float(input_values[1].value[0])

        self._wrapper.set_inputs(u)
        self._wrapper.set_theta(theta)
        self._wrapper.compute()
        y = self._wrapper.get_outputs()
        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def _load_wrapper(self) -> None:
        try:
            from coordinate_transform_wrapper import InvParkTransformWrapper
            self._wrapper = InvParkTransformWrapper()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'coordinate_transform_wrapper' not found.\n"
                "Generate with CodeGenEnd.generate_pyx_stub() and compile."
            )

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return f"InvParkTransformBlock('{self.name}', backend={be})"


# ==============================================================================
# Factory function for easy access
# ==============================================================================

def create_transform_blocks(use_c_backend: bool = False, dtype=None) -> dict:
    """
    Create all four transform blocks with consistent naming.

    Args:
        use_c_backend: Whether to use C backend
        dtype: Data type for signals

    Returns:
        Dictionary with block instances
    """
    return {
        'clarke': ClarkeTransformBlock('clarke', use_c_backend, dtype),
        'inv_clarke': InvClarkeTransformBlock('inv_clarke', use_c_backend, dtype),
        'park': ParkTransformBlock('park', use_c_backend, dtype),
        'inv_park': InvParkTransformBlock('inv_park', use_c_backend, dtype),
    }


# ==============================================================================
# Test / example usage
# ==============================================================================

if __name__ == "__main__":
    # Quick test of Python backend
    print("Testing coordinate transform blocks (Python backend)...")

    # Test Clarke transform
    clarke = ClarkeTransformBlock('test_clarke', use_c_backend=False)
    test_input = VectorSignal(np.array([1.0, 0.5, -0.5]), 'input')
    result = clarke.compute_py(0, 0, [test_input])
    print(f"Clarke [1, 0.5, -0.5] -> [{result.value[0]:.3f}, {result.value[1]:.3f}]")

    # Test inverse Clarke
    inv_clarke = InvClarkeTransformBlock('test_inv_clarke', use_c_backend=False)
    test_input = VectorSignal(np.array([1.0, 0.0]), 'input')
    result = inv_clarke.compute_py(0, 0, [test_input])
    print(f"Inverse Clarke [1, 0] -> [{result.value[0]:.3f}, {result.value[1]:.3f}, {result.value[2]:.3f}]")

    # Test Park
    park = ParkTransformBlock('test_park', use_c_backend=False)
    test_ab = VectorSignal(np.array([1.0, 0.0]), 'ab')
    test_theta = VectorSignal(np.array([np.pi/2]), 'theta')
    result = park.compute_py(0, 0, [test_ab, test_theta])
    print(f"Park [1, 0] @ 90° -> [{result.value[0]:.3f}, {result.value[1]:.3f}]")

    # Test inverse Park
    inv_park = InvParkTransformBlock('test_inv_park', use_c_backend=False)
    test_dq = VectorSignal(np.array([1.0, 0.0]), 'dq')
    result = inv_park.compute_py(0, 0, [test_dq, test_theta])
    print(f"Inverse Park [1, 0] @ 90° -> [{result.value[0]:.3f}, {result.value[1]:.3f}]")



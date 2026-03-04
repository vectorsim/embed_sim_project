"""
coordinate_transform_blocks.py
===================

Clarke / Park transformation blocks for PMSM FOC.
MATRIX OPTIMIZED VERSION - Uses matrix-based C backend.

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

C backend (Matrix-based)
------------------------
    All C backend wrappers now use matrix multiplication internally:
    - ClarkeTransformWrapper    — Uses pre-computed 3x2 Clarke matrix
    - InvClarkeTransformWrapper — Uses pre-computed 2x3 inverse matrix
    - ParkTransformWrapper      — Updates 2x2 rotation matrix each cycle
    - InvParkTransformWrapper   — Updates 2x2 inverse rotation matrix
    - ClarkeParkTransformWrapper — Combined transform for optimization
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks import VectorSignal

# Constants (keep these for Python backend)
_TWO_THIRDS = np.float32(2.0 / 3.0)
_ONE_THIRD  = np.float32(1.0 / 3.0)
_SQRT3      = np.float32(np.sqrt(3.0))
_INV_SQRT3  = np.float32(1.0 / np.sqrt(3.0))
_HALF_SQRT3 = np.float32(np.sqrt(3.0) / 2.0)


# ==============================================================================
# Base class for all transform blocks with common functionality
# ==============================================================================

class TransformBlockBase(SimBlockBase):
    """Base class for all coordinate transform blocks."""

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None):
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self._c_wrapper = None

    def compute(self, t, dt, input_values=None):
        """Dispatch to C or Python backend based on use_c_backend flag."""
        if self.use_c_backend:
            return self.compute_c(t, dt, input_values)
        return self.compute_py(t, dt, input_values)

    def _load_wrapper(self, wrapper_name: str) -> None:
        """Load Cython wrapper by name."""
        try:
            import coordinate_transform_wrapper as ctw
            self._c_wrapper = getattr(ctw, wrapper_name)()
        except ImportError:
            raise ImportError(
                "Cython wrapper 'coordinate_transform_wrapper' not found.\n"
                "Generate with CodeGenEnd.generate_pyx_stub() and compile.\n"
                "Make sure Matrix_Operations.c is included in the build."
            )
        except AttributeError:
            raise ImportError(
                f"Wrapper '{wrapper_name}' not found in coordinate_transform_wrapper.\n"
                "Check that your Cython wrapper is up to date with matrix version."
            )


# ==============================================================================
# Clarke   abc → αβ
# ==============================================================================

class ClarkeTransformBlock(TransformBlockBase):
    """
    Power-invariant Clarke transform.

        α = (2ia − ib − ic) / 3
        β = (ib − ic) / √3

    Input port 0 : [ia, ib, ic]
    Output       : [α, β]

    C Backend: Uses matrix multiplication with pre-computed 3x2 Clarke matrix.
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.output_label = "[alpha,beta]"
        self.is_dynamic  = False
        self.vector_size = 2
        if use_c_backend:
            self._load_wrapper('ClarkeTransformWrapper')

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

    # -- C backend (matrix-based) ---------------------------------------------

    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """Compute using matrix-based C implementation."""
        u = np.zeros(3, dtype=np.float32)
        if input_values and len(input_values[0].value) >= 3:
            u[:3] = input_values[0].value[:3].astype(np.float32, copy=False)

        self._c_wrapper.set_inputs(u)
        self._c_wrapper.compute()
        y = self._c_wrapper.get_outputs()

        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def __repr__(self) -> str:
        be = "C (matrix)" if self.use_c_backend else "Python"
        return f"ClarkeTransformBlock('{self.name}', backend={be})"


# ==============================================================================
# Inverse Clarke   αβ → abc
# ==============================================================================

class InvClarkeTransformBlock(TransformBlockBase):
    """
    Inverse Clarke transform.

        a = α
        b = -α/2 + (√3/2)β
        c = -α/2 - (√3/2)β

    Input port 0 : [α, β]
    Output       : [a, b, c]

    C Backend: Uses matrix multiplication with pre-computed 2x3 inverse matrix.
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.output_label = "[a,b,c]"
        self.is_dynamic  = False
        self.vector_size = 3
        if use_c_backend:
            self._load_wrapper('InvClarkeTransformWrapper')

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
        """Compute using matrix-based C implementation."""
        u = np.zeros(2, dtype=np.float32)
        if input_values and len(input_values[0].value) >= 2:
            u[:2] = input_values[0].value[:2].astype(np.float32, copy=False)

        self._c_wrapper.set_inputs(u)
        self._c_wrapper.compute()
        y = self._c_wrapper.get_outputs()

        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def __repr__(self) -> str:
        be = "C (matrix)" if self.use_c_backend else "Python"
        return f"InvClarkeTransformBlock('{self.name}', backend={be})"


# ==============================================================================
# Park   αβ → dq
# ==============================================================================

class ParkTransformBlock(TransformBlockBase):
    """
    Park transform (rotating reference frame).

        d =  α·cosθ + β·sinθ
        q = -α·sinθ + β·cosθ

    Input port 0 : [α, β]
    Input port 1 : [θ]        (rotor angle in radians)
    Output       : [d, q]

    C Backend: Updates 2x2 rotation matrix each cycle using current angle,
              then performs matrix multiplication.
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.output_label = "[d,q]"
        self.is_dynamic  = False
        self.vector_size = 2
        if use_c_backend:
            self._load_wrapper('ParkTransformWrapper')

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
        """Compute using matrix-based C implementation."""
        u = np.zeros(2, dtype=np.float32)
        theta = 0.0

        if input_values and len(input_values) >= 2:
            if len(input_values[0].value) >= 2:
                u[:2] = input_values[0].value[:2].astype(np.float32, copy=False)
            if input_values[1].value.size > 0:
                theta = float(input_values[1].value[0])

        self._c_wrapper.set_inputs(u)
        self._c_wrapper.set_theta(theta)  # Updates rotation matrix internally
        self._c_wrapper.compute()         # Performs matrix multiplication
        y = self._c_wrapper.get_outputs()

        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def __repr__(self) -> str:
        be = "C (matrix)" if self.use_c_backend else "Python"
        return f"ParkTransformBlock('{self.name}', backend={be})"


# ==============================================================================
# Inverse Park   dq → αβ
# ==============================================================================

class InvParkTransformBlock(TransformBlockBase):
    """
    Inverse Park transform.

        α = d·cosθ - q·sinθ
        β = d·sinθ + q·cosθ

    Input port 0 : [d, q]
    Input port 1 : [θ]        (rotor angle in radians)
    Output       : [α, β]

    C Backend: Updates 2x2 inverse rotation matrix each cycle using current angle,
              then performs matrix multiplication.
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.output_label = "[alpha,beta]"
        self.is_dynamic  = False
        self.vector_size = 2
        if use_c_backend:
            self._load_wrapper('InvParkTransformWrapper')

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
        """Compute using matrix-based C implementation."""
        u = np.zeros(2, dtype=np.float32)
        theta = 0.0

        if input_values and len(input_values) >= 2:
            if len(input_values[0].value) >= 2:
                u[:2] = input_values[0].value[:2].astype(np.float32, copy=False)
            if input_values[1].value.size > 0:
                theta = float(input_values[1].value[0])

        self._c_wrapper.set_inputs(u)
        self._c_wrapper.set_theta(theta)  # Updates inverse matrix internally
        self._c_wrapper.compute()          # Performs matrix multiplication
        y = self._c_wrapper.get_outputs()

        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def __repr__(self) -> str:
        be = "C (matrix)" if self.use_c_backend else "Python"
        return f"InvParkTransformBlock('{self.name}', backend={be})"


# ==============================================================================
# Optional: Combined Clarke-Park for optimized FOC
# ==============================================================================

class ClarkeParkTransformBlock(TransformBlockBase):
    """
    Combined Clarke+Park transform (abc → dq in one step).

    Optimized transform that goes directly from three-phase to rotating frame:
        [d; q] = [R(θ)] * [C] * [A; B; C]

    Input port 0 : [ia, ib, ic]
    Input port 1 : [θ]        (rotor angle in radians)
    Output       : [d, q]

    C Backend Only: Uses optimized combined matrix multiplication.
    """

    def __init__(self, name: str, use_c_backend: bool = False, dtype=None) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)
        self.output_label = "[d,q]"
        self.is_dynamic = False
        self.vector_size = 2

        if not use_c_backend:
            raise ValueError("ClarkeParkTransformBlock requires C backend (use_c_backend=True)")

        self._load_wrapper('ClarkeParkTransformWrapper')

    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """Not implemented - Python backend not available for combined transform."""
        raise NotImplementedError(
            "ClarkeParkTransformBlock only available with C backend. "
            "Set use_c_backend=True and ensure wrapper is compiled."
        )

    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """Compute using optimized combined matrix-based C implementation."""
        u = np.zeros(3, dtype=np.float32)
        theta = 0.0

        if input_values and len(input_values) >= 2:
            if len(input_values[0].value) >= 3:
                u[:3] = input_values[0].value[:3].astype(np.float32, copy=False)
            if input_values[1].value.size > 0:
                theta = float(input_values[1].value[0])

        self._c_wrapper.set_inputs(u)
        self._c_wrapper.set_theta(theta)
        self._c_wrapper.compute()
        y = self._c_wrapper.get_outputs()

        self.output = VectorSignal(y, self.name, dtype=self.dtype)
        return self.output

    def __repr__(self) -> str:
        return f"ClarkeParkTransformBlock('{self.name}', backend='C (optimized matrix)')"


# ==============================================================================
# Factory function for easy access
# ==============================================================================

def create_transform_blocks(
    use_c_backend: bool = False,
    dtype=None,
    include_combined: bool = False
) -> Dict[str, Any]:
    """
    Create all transform blocks with consistent naming.

    Args:
        use_c_backend: Whether to use C backend
        dtype: Data type for signals
        include_combined: Whether to include the combined ClarkePark block

    Returns:
        Dictionary with block instances
    """
    blocks = {
        'clarke': ClarkeTransformBlock('clarke', use_c_backend, dtype),
        'inv_clarke': InvClarkeTransformBlock('inv_clarke', use_c_backend, dtype),
        'park': ParkTransformBlock('park', use_c_backend, dtype),
        'inv_park': InvParkTransformBlock('inv_park', use_c_backend, dtype),
    }

    if include_combined:
        if not use_c_backend:
            raise ValueError(
                "ClarkeParkTransformBlock requires use_c_backend=True. "
                "Pass use_c_backend=True or include_combined=False."
            )
        blocks['clarke_park'] = ClarkeParkTransformBlock(
            'clarke_park', use_c_backend, dtype
        )

    return blocks


# ==============================================================================
# Test / example usage
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing coordinate transform blocks")
    print("=" * 60)

    # Test Python backend (unchanged)
    print("\n--- Python Backend ---")
    blocks_py = create_transform_blocks(use_c_backend=False)

    # Test Clarke
    test_input = VectorSignal(np.array([1.0, 0.5, -0.5]), 'input')
    result = blocks_py['clarke'].compute_py(0, 0, [test_input])
    print(f"Clarke [1, 0.5, -0.5] -> [{result.value[0]:.3f}, {result.value[1]:.3f}]")

    # Test inverse Clarke
    test_input = VectorSignal(np.array([1.0, 0.0]), 'input')
    result = blocks_py['inv_clarke'].compute_py(0, 0, [test_input])
    print(f"Inverse Clarke [1, 0] -> [{result.value[0]:.3f}, {result.value[1]:.3f}, {result.value[2]:.3f}]")

    # Test Park
    test_ab = VectorSignal(np.array([1.0, 0.0]), 'ab')
    test_theta = VectorSignal(np.array([np.pi/2]), 'theta')
    result = blocks_py['park'].compute_py(0, 0, [test_ab, test_theta])
    print(f"Park [1, 0] @ 90° -> [{result.value[0]:.3f}, {result.value[1]:.3f}]")

    # Test inverse Park
    test_dq = VectorSignal(np.array([1.0, 0.0]), 'dq')
    result = blocks_py['inv_park'].compute_py(0, 0, [test_dq, test_theta])
    print(f"Inverse Park [1, 0] @ 90° -> [{result.value[0]:.3f}, {result.value[1]:.3f}]")

    # If C wrapper is available, test C backend
    try:
        print("\n--- C Backend (Matrix-based) ---")
        blocks_c = create_transform_blocks(use_c_backend=True, include_combined=True)

        # Test Clarke with C backend
        test_abc_main = VectorSignal(np.array([1.0, 0.5, -0.5], dtype=np.float32), 'abc_main')
        result = blocks_c['clarke'].compute_c(0, 0, [test_abc_main])
        print(f"Clarke [1, 0.5, -0.5] -> [{result.value[0]:.3f}, {result.value[1]:.3f}] (C matrix)")

        # Test Park with C backend
        result = blocks_c['park'].compute_c(0, 0, [test_ab, test_theta])
        print(f"Park [1, 0] @ 90° -> [{result.value[0]:.3f}, {result.value[1]:.3f}] (C matrix)")

        # Test combined ClarkePark if available
        if 'clarke_park' in blocks_c:
            test_abc = VectorSignal(np.array([1.0, 0.5, -0.5]), 'abc')
            result = blocks_c['clarke_park'].compute_c(0, 0, [test_abc, test_theta])
            print(f"ClarkePark [1, 0.5, -0.5] @ 90° -> [{result.value[0]:.3f}, {result.value[1]:.3f}] (optimized)")

    except ImportError as e:
        print(f"\nC backend not available: {e}")
        print("Compile the C wrapper first with build_all.bat")
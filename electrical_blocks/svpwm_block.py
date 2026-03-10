"""
svpwm_block.py
==============

Space Vector PWM (SVPWM) block for EmbedSim.
Converts three-phase voltage commands to duty cycles for PWM generation.

Ports
-----
    port 0 : [v_a, v_b, v_c]    — phase voltages w.r.t. virtual neutral [V]

Output
------
    [duty_a, duty_b, duty_c]     — duty cycles in range [0, 1]

Algorithm
---------
    duty_x = v_x / v_dc + 0.5
    duty_x = clamp(duty_x, 0.0, 1.0)

FIX (v1.0.1)
------------
    Removed the stale-fallback guard in normal operation:
        OLD: if abs(v_a) < 1.0 and abs(v_b) < 1.0 and abs(v_c) < 1.0:
                 out = self._last_open_loop_duties.copy()   # ← froze loop
    This caused the control loop to stall permanently after t=0.2 s whenever
    the commanded voltages happened to be small (e.g. during SMO convergence).
    The block now always computes duties from the actual input voltages and
    keeps _last_open_loop_duties current so it is available if needed.

EmbedSim CodeGen attributes
---------------------------
    step_func    : SVPWM_Compute
    state_struct : SVPWM_Block_T
    C_SOURCES    : ['svpwm.c']
    C_HEADERS    : ['svpwm.h', 'Sys_Types.h']
    NUM_INPUTS   : 1   (port 0: [v_a, v_b, v_c])
    OUTPUT_SIZE  : 3   ([duty_a, duty_b, duty_c])
"""

import sys
import numpy as np
from typing import List, Optional

from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks    import VectorSignal

# Debug flag
DEBUG = False


# ==============================================================================
# Pure-Python SVPWM implementation (no C dependency)
# ==============================================================================

class _PySVPWM:
    """
    Pure Python implementation of Space Vector PWM.
    Used when C wrapper is not available or use_c_backend=False.
    """

    def __init__(self, v_dc: float = 48.0):
        self._v_dc = float(v_dc)
        self.duty_a = 0.5
        self.duty_b = 0.5
        self.duty_c = 0.5

        self._step = 0
        # Stores last open-loop duties for use during transition blend only.
        # Updated every open-loop step so the transition starts from a valid value.
        self._last_open_loop_duties = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    @property
    def v_dc(self):
        return self._v_dc

    @v_dc.setter
    def v_dc(self, value):
        self._v_dc = float(value)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def compute(self, v_a: float, v_b: float, v_c: float, t: float = 0.0) -> np.ndarray:
        """
        Compute duty cycles from phase voltages.

        Parameters
        ----------
        v_a, v_b, v_c : float
            Phase voltages w.r.t. virtual neutral [V]
        t : float, optional
            Current time [s]

        Returns
        -------
        np.ndarray, shape (3,), dtype float32
            [duty_a, duty_b, duty_c] in [0, 1]
        """
        self._step += 1

        # ── Open-loop startup (t < 0.15 s) ───────────────────────────────────
        if t < 0.15:
            freq = 10.0
            theta = 2.0 * np.pi * freq * t
            va_open = 20.0 * np.sin(theta)
            vb_open = 20.0 * np.sin(theta - 2.0 * np.pi / 3.0)
            vc_open = 20.0 * np.sin(theta + 2.0 * np.pi / 3.0)

            d_a = self._clamp(va_open / self._v_dc + 0.5, 0.0, 1.0)
            d_b = self._clamp(vb_open / self._v_dc + 0.5, 0.0, 1.0)
            d_c = self._clamp(vc_open / self._v_dc + 0.5, 0.0, 1.0)

            out = np.array([d_a, d_b, d_c], dtype=np.float32)
            self._last_open_loop_duties = out.copy()

            if DEBUG and self._step % 1000 == 0:
                print(f"SVPWM t={t:.3f}: OPEN LOOP - duties=({d_a:.3f},{d_b:.3f},{d_c:.3f})")

        # ── Transition blend (0.15 s → 0.20 s) ───────────────────────────────
        elif t < 0.2:
            d_a_cl = self._clamp(v_a / self._v_dc + 0.5, 0.0, 1.0)
            d_b_cl = self._clamp(v_b / self._v_dc + 0.5, 0.0, 1.0)
            d_c_cl = self._clamp(v_c / self._v_dc + 0.5, 0.0, 1.0)
            closed_loop = np.array([d_a_cl, d_b_cl, d_c_cl], dtype=np.float32)

            alpha = (t - 0.15) / 0.05   # ramps 0 → 1
            out = (1.0 - alpha) * self._last_open_loop_duties + alpha * closed_loop
            out = out.astype(np.float32)

            if DEBUG and self._step % 1000 == 0:
                print(f"SVPWM t={t:.3f}: TRANSITION - alpha={alpha:.2f}")

        # ── Normal closed-loop operation (t ≥ 0.20 s) ────────────────────────
        else:
            d_a = self._clamp(v_a / self._v_dc + 0.5, 0.0, 1.0)
            d_b = self._clamp(v_b / self._v_dc + 0.5, 0.0, 1.0)
            d_c = self._clamp(v_c / self._v_dc + 0.5, 0.0, 1.0)
            out = np.array([d_a, d_b, d_c], dtype=np.float32)
            # FIX: always update from actual computed duties — never fall back
            # to stale open-loop values.  The old guard
            #     if abs(v_a) < 1.0 and abs(v_b) < 1.0 and abs(v_c) < 1.0:
            #         out = self._last_open_loop_duties.copy()
            # was causing the loop to freeze permanently whenever commanded
            # voltages were small during SMO convergence.
            self._last_open_loop_duties = out.copy()

        self.duty_a, self.duty_b, self.duty_c = float(out[0]), float(out[1]), float(out[2])
        return out

    def compute_array(self, vabc: np.ndarray, t: float = 0.0) -> np.ndarray:
        return self.compute(vabc[0], vabc[1], vabc[2], t)

    def reset(self):
        self.duty_a = 0.5
        self.duty_b = 0.5
        self.duty_c = 0.5
        self._step = 0
        self._last_open_loop_duties = np.array([0.5, 0.5, 0.5], dtype=np.float32)


# ==============================================================================
# SVPWMBlock  —  EmbedSim VectorBlock
# ==============================================================================

class SVPWMBlock(SimBlockBase):
    """
    Space Vector PWM modulator block.

    Converts three-phase voltage commands to duty cycles for PWM generation.
    Includes an open-loop startup pattern for initial motor excitation.

    Parameters
    ----------
    name         : str   — unique block identifier
    v_dc         : float — DC bus voltage [V] (default 48.0)
    use_c_backend: bool  — use compiled svpwm_wrapper.pyd
    """

    # ── CodeGen marker attributes ─────────────────────────────────────────
    import pathlib as _pl
    PYX_FILE:    str  = str(_pl.Path(__file__).parent / 'c_src' / 'svpwm_wrapper.pyx')
    step_func:   str  = 'SVPWM_Compute'
    state_struct: str = 'SVPWM_Block_T'
    NUM_INPUTS:  int  = 1
    OUTPUT_SIZE: int  = 3
    C_SOURCES:   list = ['svpwm.c']
    C_HEADERS:   list = ['svpwm.h', 'Sys_Types.h']

    def __init__(
        self,
        name:          str,
        v_dc:          float = 48.0,
        use_c_backend: bool  = False,
        dtype                = None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.output_label = "[duty_a,duty_b,duty_c]"
        self.is_dynamic   = False
        self.vector_size  = 3
        self.state        = None
        self._v_dc        = float(v_dc)

        self._debug_count = 0
        self._last_t      = 0.0

        if use_c_backend:
            self._impl = self._load_c_wrapper(v_dc)
        else:
            self._impl = _PySVPWM(v_dc)

    # ── C loader ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_c_wrapper(v_dc):
        try:
            import svpwm_wrapper as sw
            return sw.SVPWMWrapper(v_dc=v_dc)
        except ImportError:
            raise ImportError(
                "Cython wrapper 'svpwm_wrapper' not found.\n"
                "Compile with: python setup_svpwm.py build_ext --inplace\n"
                "Or set use_c_backend=False to use the Python backend."
            )

    # ── Input parsing ────────────────────────────────────────────────────

    def _parse_inputs(self, input_values):
        """Extract (v_a, v_b, v_c) safely."""
        v_a = v_b = v_c = 0.0

        if not input_values or len(input_values) < 1:
            return v_a, v_b, v_c

        if input_values[0] is not None:
            val = input_values[0].value
            if len(val) >= 3:
                v_a, v_b, v_c = float(val[0]), float(val[1]), float(val[2])
            elif len(val) == 2:
                v_a, v_b = float(val[0]), float(val[1])
            elif len(val) == 1:
                v_a = float(val[0])

        return v_a, v_b, v_c

    # ── Compute dispatch ─────────────────────────────────────────────────

    def compute(self, t, dt, input_values=None):
        self._debug_count += 1
        self._last_t = t

        if self.use_c_backend:
            return self.compute_c(t, dt, input_values)
        return self.compute_py(t, dt, input_values)

    # -- Python backend -------------------------------------------------------

    def compute_py(
        self,
        t:  float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        v_a, v_b, v_c = self._parse_inputs(input_values)

        if DEBUG and self._debug_count % 1000 == 0:
            print(f"SVPWM t={t:.3f}: v_abc=({v_a:.2f}, {v_b:.2f}, {v_c:.2f})")

        out = self._impl.compute(v_a, v_b, v_c, t)

        if DEBUG and self._debug_count % 1000 == 0:
            print(f"SVPWM t={t:.3f}: duties=({out[0]:.3f}, {out[1]:.3f}, {out[2]:.3f})")

        self.output = VectorSignal(out.astype(np.float32), self.name, dtype=self.dtype)
        return self.output

    # -- C backend ------------------------------------------------------------

    def compute_c(
        self,
        t:  float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        v_a, v_b, v_c = self._parse_inputs(input_values)

        if DEBUG and self._debug_count % 1000 == 0:
            print(f"SVPWM t={t:.3f}: v_abc=({v_a:.2f}, {v_b:.2f}, {v_c:.2f})")

        # Use Python impl for open-loop / transition phases (time-based pattern)
        if t < 0.2:
            out = self._impl.compute(v_a, v_b, v_c, t)
        else:
            out = self._impl.compute(v_a, v_b, v_c)

        if DEBUG and self._debug_count % 1000 == 0:
            print(f"SVPWM t={t:.3f}: duties=({out[0]:.3f}, {out[1]:.3f}, {out[2]:.3f})")

        self.output = VectorSignal(out.astype(np.float32), self.name, dtype=self.dtype)
        return self.output

    # ── Block lifecycle ─────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._impl.reset()
        self._debug_count = 0
        self._last_t      = 0.0

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def duty_a(self) -> float:
        return self._impl.duty_a if hasattr(self._impl, 'duty_a') else 0.5

    @property
    def duty_b(self) -> float:
        return self._impl.duty_b if hasattr(self._impl, 'duty_b') else 0.5

    @property
    def duty_c(self) -> float:
        return self._impl.duty_c if hasattr(self._impl, 'duty_c') else 0.5

    @property
    def v_dc(self) -> float:
        return self._v_dc

    @v_dc.setter
    def v_dc(self, value: float):
        self._v_dc = float(value)
        if hasattr(self._impl, 'v_dc'):
            self._impl.v_dc = self._v_dc

    def set_v_dc(self, v_dc: float) -> None:
        self.v_dc = v_dc

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return f"SVPWMBlock('{self.name}', v_dc={self._v_dc:.1f}V, backend={be})"


# ==============================================================================
# Factory function
# ==============================================================================

def create_svpwm_block(
    name: str = "svpwm",
    v_dc: float = 48.0,
    use_c_backend: bool = False,
) -> SVPWMBlock:
    return SVPWMBlock(name=name, v_dc=v_dc, use_c_backend=use_c_backend)


# ==============================================================================
# Test / example usage
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SVPWM block")
    print("=" * 60)

    svpwm = SVPWMBlock("test_svpwm", v_dc=48.0, use_c_backend=False)

    test_voltages = [
        (0.0,  0.0,   0.0),
        (24.0, -12.0, -12.0),
        (20.0,  20.0,  20.0),
        (48.0,   0.0,   0.0),
    ]

    print("\n--- Python Backend ---")
    for va, vb, vc in test_voltages:
        input_sig = VectorSignal(np.array([va, vb, vc], dtype=np.float32), "test")
        result = svpwm.compute_py(t=0.25, dt=1e-4, input_values=[input_sig])
        print(f"Input ({va:5.1f}, {vb:5.1f}, {vc:5.1f}) V -> "
              f"Output ({result.value[0]:.3f}, {result.value[1]:.3f}, {result.value[2]:.3f})")

    print("\n--- Open-loop Startup Pattern ---")
    for t in [0.025, 0.075, 0.125, 0.175, 0.225]:
        result = svpwm.compute_py(t=t, dt=1e-4, input_values=None)
        print(f"t={t:.3f}s -> Output "
              f"({result.value[0]:.3f}, {result.value[1]:.3f}, {result.value[2]:.3f})")

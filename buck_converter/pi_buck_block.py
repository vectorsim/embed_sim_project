"""
pi_buck_block.py
================

PI Buck converter voltage controller block for EmbedSim.

Derives from SimBlockBase so it participates in CodeGenStart/CodeGenEnd
regions and its C_SOURCES / C_HEADERS attributes are picked up by
PYXInspector when generating embedsim_loop.c.

Position in control chain
-------------------------
    V_ref ──► [PI_BuckBlock] ──► duty ──► [BuckConverterBlock]
    V_meas ──►

Ports
-----
    port 0 : [V_ref]     — voltage reference  (scalar wrapped in vector) [V]
    port 1 : [V_meas]    — measured voltage   (from BuckConverter)       [V]

Output
------
    [duty]               — PWM duty cycle [0-1]

RK4 integration
---------------
    state[0] = integrator accumulator  (integrated by EmbedSim engine)
    get_derivative() returns [error]   so RK4 integrates it correctly

Backends
--------
    Python : _PyPI_Buck  (always available)
    C      : pi_buck_wrapper.pyd  (compile with setup_pi_buck.py)

Author : EmbedSim Framework
Version: 1.0.0
"""

import sys
import numpy as np
from typing import List, Optional

from _path_utils import get_embedsim_import_path

sys.path.insert(0, get_embedsim_import_path())

from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks import VectorSignal


# ==============================================================================
# Pure-Python implementation
# ==============================================================================

class _PyPI_Buck:
    """Minimal Python mirror of PI_Buck_Block_T."""

    def __init__(self, Kp: float, Ki: float, duty_max: float, duty_min: float, Ts: float):
        self.Kp = np.float32(Kp)
        self.Ki = np.float32(max(Ki, 1e-9))
        self.duty_max = np.float32(duty_max)
        self.duty_min = np.float32(duty_min)
        self.Ts = np.float32(Ts)
        self.integ = np.float32(0.0)
        self.last_duty = np.float32(0.0)

    def compute(self, V_ref: float, V_meas: float, dt: float) -> float:
        e = np.float32(V_ref - V_meas)
        lim = self.duty_max / self.Ki
        sample_time = dt if dt > 0 else self.Ts
        self.integ = np.float32(np.clip(self.integ + e * sample_time, -lim, lim))
        duty = np.clip(self.Kp * e + self.Ki * self.integ,
                       self.duty_min, self.duty_max)
        self.last_duty = np.float32(duty)
        return float(duty)

    def reset(self):
        self.integ = np.float32(0.0)
        self.last_duty = np.float32(0.0)

    def get_integrator(self): return float(self.integ)


# ==============================================================================
# PI_BuckBlock  —  EmbedSim SimBlockBase (CodeGen-ready, RK4-compatible)
# ==============================================================================

class PI_BuckBlock(SimBlockBase):
    """
    Proportional-Integral voltage controller for Buck Converter.

    Derives from SimBlockBase — participates in CodeGenStart/CodeGenEnd
    regions.  CodeGenEnd.generate_pyx_stub() will include C_SOURCES and
    C_HEADERS when emitting embedsim_loop.c.

    Parameters
    ----------
    name      : str   — unique block identifier
    Kp        : float — proportional gain      (default 0.1  1/V)
    Ki        : float — integral gain          (default 5.0  1/(V·s))
    duty_max  : float — maximum duty cycle     (default 0.95)
    duty_min  : float — minimum duty cycle     (default 0.05)
    Ts        : float — nominal sample time    (default 100µs = 1e-4)
    use_c_backend : bool — use compiled pi_buck_wrapper.pyd

    Ports
    -----
    port 0 : [V_ref]    — voltage reference    [V]
    port 1 : [V_meas]   — measured voltage     [V]

    Output
    ------
    [duty]               — PWM duty cycle [0-1]

    RK4 state
    ---------
    state[0] = error integrator
    """

    # ── CodeGen marker attributes (read by PYXInspector / CodeGenEnd) ────────
    import pathlib as _pl
    #: Absolute path to .pyx — works regardless of working directory
    PYX_FILE: str = str(_pl.Path(__file__).parent / 'c_src' / 'pi_buck_wrapper.pyx')
    #: step_func / state_struct filled by PYXInspector via __init_subclass__
    step_func: str = 'PI_Buck_Compute'
    state_struct: str = 'PI_Buck_Block_T'
    NUM_INPUTS: int = 2
    OUTPUT_SIZE: int = 1
    C_SOURCES: list = ['pi_buck_controller.c']
    C_HEADERS: list = ['pi_buck_controller.h', 'Sys_Types.h']

    def __init__(
            self,
            name: str,
            Kp: float = 0.1,
            Ki: float = 5.0,
            duty_max: float = 0.95,
            duty_min: float = 0.05,
            Ts: float = 1e-4,
            t_enable: float = 0.0,
            use_c_backend: bool = False,
            dtype=None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.output_label = "[duty]"
        self.vector_size = 1
        self.is_dynamic = True  # has integrator state → RK4

        self._Kp = float(Kp)
        self._Ki = float(Ki)
        self._duty_max = float(duty_max)
        self._duty_min = float(duty_min)
        self._Ts = float(Ts)
        self._t_enable = float(t_enable)

        # ── RK4-compatible state: [integrator] ────────────────────────────────
        self.state = np.zeros(1, dtype=np.float32)
        self.k1 = self.k2 = self.k3 = self.k4 = np.zeros(1, dtype=np.float32)

        if use_c_backend:
            self._impl = self._load_c_wrapper(Kp, Ki, duty_max, duty_min, Ts)
        else:
            self._impl = _PyPI_Buck(Kp, Ki, duty_max, duty_min, Ts)

    # ── C loader ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_c_wrapper(Kp, Ki, duty_max, duty_min, Ts):
        try:
            import pi_buck_wrapper as pbw
            w = pbw.PI_BuckWrapper()
            w.set_params(Kp=Kp, Ki=Ki, duty_max=duty_max, duty_min=duty_min, Ts=Ts)
            return w
        except ImportError:
            raise ImportError(
                "Cython wrapper 'pi_buck_wrapper' not found.\n"
                "Compile with: python setup_pi_buck.py build_ext --inplace\n"
                "Or set use_c_backend=False to use the Python backend."
            )

    # ── Input helpers ─────────────────────────────────────────────────────────

    def _get_voltages(self, input_values):
        V_ref = 0.0
        V_meas = 0.0
        if not input_values:
            return V_ref, V_meas
        if input_values[0] is not None:
            v = input_values[0].value
            V_ref = float(v[0]) if len(v) >= 1 else 0.0
        if len(input_values) > 1 and input_values[1] is not None:
            v = input_values[1].value
            V_meas = float(v[0]) if len(v) >= 1 else 0.0
        return V_ref, V_meas

    # ── RK4 interface ─────────────────────────────────────────────────────────

    def get_derivative(self, t: float,
                       input_values: Optional[List[VectorSignal]] = None
                       ) -> np.ndarray:
        """dx/dt = [error]  — frozen at zero while t < t_enable."""
        if t < self._t_enable:
            self.state[0] = 0.0
            return np.zeros(1, dtype=np.float32)
        V_ref, V_meas = self._get_voltages(input_values)
        e = np.float32(V_ref - V_meas)
        return np.array([e], dtype=np.float32)

    # ── Compute dispatch ──────────────────────────────────────────────────────

    def compute(self, t, dt, input_values=None):
        if self.use_c_backend:
            return self.compute_c(t, dt, input_values)
        return self.compute_py(t, dt, input_values)

    # -- Python backend -------------------------------------------------------

    def compute_py(
            self,
            t: float,
            dt: float,
            input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        if t < self._t_enable:
            self.output = VectorSignal(
                np.array([0.0], dtype=np.float32), self.name, dtype=self.dtype
            )
            return self.output
        V_ref, V_meas = self._get_voltages(input_values)
        # Use RK4-integrated state[0] as the authoritative integrator
        lim = np.float32(self._duty_max / max(self._Ki, 1e-9))
        integ = np.float32(np.clip(self.state[0], -lim, lim))
        e = np.float32(V_ref - V_meas)
        sample_time = dt if dt > 0 else self._Ts
        duty = float(np.clip(
            self._Kp * e + self._Ki * integ,
            self._duty_min, self._duty_max
        ))
        self.output = VectorSignal(
            np.array([duty], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    # -- C backend ------------------------------------------------------------

    def compute_c(
            self,
            t: float,
            dt: float,
            input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        V_ref, V_meas = self._get_voltages(input_values)
        # Sync state[0] → C wrapper integrator before compute
        self._impl.set_integrator(np.float32(self.state[0]))
        duty = self._impl.compute(
            np.float32(V_ref), np.float32(V_meas), np.float32(dt)
        )
        # Sync C wrapper integrator → state[0] after compute
        self.state[0] = np.float32(self._impl.get_integrator())
        self.output = VectorSignal(
            np.array([duty], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    # ── Block lifecycle ───────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self.state = np.zeros(1, dtype=np.float32)
        if hasattr(self, '_impl'):
            self._impl.reset()

    # ── Runtime parameter update ──────────────────────────────────────────────

    def set_params(self, Kp=None, Ki=None, duty_max=None, duty_min=None, Ts=None) -> None:
        """Update PI parameters at runtime."""
        if Kp is not None: self._Kp = float(Kp)
        if Ki is not None: self._Ki = float(Ki)
        if duty_max is not None: self._duty_max = float(duty_max)
        if duty_min is not None: self._duty_min = float(duty_min)
        if Ts is not None: self._Ts = float(Ts)

        if self.use_c_backend:
            self._impl.set_params(
                Kp=self._Kp, Ki=self._Ki,
                duty_max=self._duty_max, duty_min=self._duty_min,
                Ts=self._Ts
            )
        else:
            self._impl.Kp = np.float32(self._Kp)
            self._impl.Ki = np.float32(max(self._Ki, 1e-9))
            self._impl.duty_max = np.float32(self._duty_max)
            self._impl.duty_min = np.float32(self._duty_min)
            self._impl.Ts = np.float32(self._Ts)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def integrator(self) -> float:
        """Current integrator state."""
        return float(self.state[0])

    @property
    def last_duty(self) -> float:
        """Last computed duty cycle."""
        if self.use_c_backend:
            return float(self._impl.get_last_output())
        return float(self._impl.last_duty)

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return (f"PI_BuckBlock('{self.name}', "
                f"Kp={self._Kp}, Ki={self._Ki}, "
                f"duty=[{self._duty_min},{self._duty_max}], "
                f"backend={be})")
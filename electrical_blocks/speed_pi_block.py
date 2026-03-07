"""
speed_pi_block.py
=================

Speed PI controller block for EmbedSim PMSM FOC outer loop.

Derives from SimBlockBase so it participates in CodeGenStart/CodeGenEnd
regions and its C_SOURCES / C_HEADERS attributes are picked up by
PYXInspector (feature 05121967) when generating embedsim_loop.c.

Position in FOC chain
---------------------
    omega_ref ──► [SpeedPIBlock] ──► [id_ref, iq_ref] ──► [SMCBlock]
    omega_meas ──►

Ports
-----
    port 0 : [omega_ref]    — speed reference  (scalar wrapped in vector) [rad/s]
    port 1 : [omega_meas]   — measured speed   (from VectorDelay)         [rad/s]

Output
------
    [id_ref, iq_ref]        — id_ref = 0 (MTPA), iq_ref = PI output      [A]

RK4 integration
---------------
    state[0] = integrator accumulator  (integrated by EmbedSim engine)
    get_derivative() returns [error]   so RK4 integrates it correctly

Backends
--------
    Python : _PySpeedPI  (always available)
    C      : speed_pi_wrapper.pyd  (compile with setup_speed_pi.py)

Author : EmbedSim Framework
Version: 1.0.0
"""

import sys
import numpy as np
from typing import List, Optional

from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks    import VectorSignal


# ==============================================================================
# Pure-Python implementation
# ==============================================================================

class _PySpeedPI:
    """Minimal Python mirror of SpeedPI_Block_T."""

    def __init__(self, Kp: float, Ki: float, i_max: float):
        self.Kp     = np.float32(Kp)
        self.Ki     = np.float32(max(Ki, 1e-9))
        self.i_max  = np.float32(i_max)
        self.integ  = np.float32(0.0)

    def compute(self, omega_ref: float, omega_meas: float, dt: float):
        e           = np.float32(omega_ref - omega_meas)
        lim         = self.i_max / self.Ki
        self.integ  = np.float32(np.clip(self.integ + e * dt, -lim, lim))
        iq_ref      = np.clip(self.Kp * e + self.Ki * self.integ,
                               -self.i_max, self.i_max)
        return float(0.0), float(iq_ref)

    def reset(self):
        self.integ = np.float32(0.0)

    def get_integrator(self): return float(self.integ)


# ==============================================================================
# SpeedPIBlock  —  EmbedSim SimBlockBase (CodeGen-ready, RK4-compatible)
# ==============================================================================

class SpeedPIBlock(SimBlockBase):
    """
    Proportional-Integral speed controller for PMSM FOC outer loop.

    Derives from SimBlockBase — participates in CodeGenStart/CodeGenEnd
    regions.  CodeGenEnd.generate_pyx_stub() will include C_SOURCES and
    C_HEADERS when emitting embedsim_loop.c.

    Parameters
    ----------
    name      : str   — unique block identifier
    Kp        : float — proportional gain      (default 0.5  A·s/rad)
    Ki        : float — integral gain          (default 5.0  A/rad)
    i_max     : float — output clamp           (default 10.0 A)
    use_c_backend : bool — use compiled speed_pi_wrapper.pyd

    Ports
    -----
    port 0 : [omega_ref]   — speed reference   [rad/s]
    port 1 : [omega_meas]  — measured speed    [rad/s]

    Output
    ------
    [id_ref, iq_ref]       — current references [A]

    RK4 state
    ---------
    state[0] = error integrator
    """

    # ── CodeGen marker attributes (read by PYXInspector / CodeGenEnd) ────────
    import pathlib as _pl
    #: Absolute path to .pyx — works regardless of working directory
    PYX_FILE:     str  = str(_pl.Path(__file__).parent / 'c_src' / 'speed_pi_wrapper.pyx')
    #: step_func / state_struct filled by PYXInspector via __init_subclass__
    step_func:    str  = 'SpeedPI_Compute'
    state_struct: str  = 'SpeedPI_Block_T'
    NUM_INPUTS:   int  = 2
    OUTPUT_SIZE:  int  = 2
    C_SOURCES:    list = ['speed_pi_controller.c']
    C_HEADERS:    list = ['speed_pi_controller.h', 'Sys_Types.h']

    def __init__(
        self,
        name:          str,
        Kp:            float = 0.5,
        Ki:            float = 5.0,
        i_max:         float = 10.0,
        use_c_backend: bool  = False,
        dtype                = None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.output_label = "[id_ref,iq_ref]"
        self.vector_size  = 2
        self.is_dynamic   = True          # has integrator state → RK4

        self._Kp    = float(Kp)
        self._Ki    = float(Ki)
        self._i_max = float(i_max)

        # ── RK4-compatible state: [integrator] ────────────────────────────────
        self.state = np.zeros(1, dtype=np.float32)
        self.k1 = self.k2 = self.k3 = self.k4 = np.zeros(1, dtype=np.float32)

        if use_c_backend:
            self._impl = self._load_c_wrapper(Kp, Ki, i_max)
        else:
            self._impl = _PySpeedPI(Kp, Ki, i_max)

    # ── C loader ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_c_wrapper(Kp, Ki, i_max):
        try:
            import speed_pi_wrapper as spw
            w = spw.SpeedPIWrapper()
            w.set_params(Kp=Kp, Ki=Ki, i_max=i_max)
            return w
        except ImportError:
            raise ImportError(
                "Cython wrapper 'speed_pi_wrapper' not found.\n"
                "Compile with: python setup_speed_pi.py build_ext --inplace\n"
                "Or set use_c_backend=False to use the Python backend."
            )

    # ── Input helpers ─────────────────────────────────────────────────────────

    def _get_omega(self, input_values):
        # port 0: [omega_ref]  — scalar from VectorStep (index 0)
        # port 1: full 7-element motor vector from delay_omega — index [2] = omega_m
        omega_ref  = float(input_values[0].value[0]) \
                     if input_values else 0.0
        omega_meas = float(input_values[1].value[2]) \
                     if input_values and len(input_values) > 1 else 0.0
        return omega_ref, omega_meas

    # ── RK4 interface ─────────────────────────────────────────────────────────

    def get_derivative(self, t: float,
                       input_values: Optional[List[VectorSignal]] = None
                       ) -> np.ndarray:
        """dx/dt = [error]  — RK4 engine integrates the accumulator."""
        omega_ref, omega_meas = self._get_omega(input_values)
        e = np.float32(omega_ref - omega_meas)
        return np.array([e], dtype=np.float32)

    # ── Compute dispatch ──────────────────────────────────────────────────────

    def compute(self, t, dt, input_values=None):
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
        omega_ref, omega_meas = self._get_omega(input_values)
        # Use RK4-integrated state[0] as the authoritative integrator
        lim    = np.float32(self._i_max / max(self._Ki, 1e-9))
        integ  = np.float32(np.clip(self.state[0], -lim, lim))
        e      = np.float32(omega_ref - omega_meas)
        iq_ref = float(np.clip(
            self._Kp * e + self._Ki * integ,
            -self._i_max, self._i_max
        ))
        self.output = VectorSignal(
            np.array([0.0, iq_ref], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    # -- C backend ------------------------------------------------------------

    def compute_c(
        self,
        t:  float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        omega_ref, omega_meas = self._get_omega(input_values)
        # Sync state[0] → C wrapper integrator before compute
        self._impl.set_integrator(np.float32(self.state[0]))
        id_ref, iq_ref = self._impl.compute(
            np.float32(omega_ref), np.float32(omega_meas), np.float32(dt)
        )
        # Sync C wrapper integrator → state[0] after compute
        self.state[0] = np.float32(self._impl.get_integrator())
        self.output = VectorSignal(
            np.array([id_ref, iq_ref], dtype=np.float32), self.name, dtype=self.dtype
        )
        return self.output

    # ── Block lifecycle ───────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self.state = np.zeros(1, dtype=np.float32)
        self._impl.reset()

    # ── Runtime parameter update ──────────────────────────────────────────────

    def set_params(self, Kp=None, Ki=None, i_max=None) -> None:
        """Update PI parameters at runtime."""
        if Kp    is not None: self._Kp    = float(Kp)
        if Ki    is not None: self._Ki    = float(Ki)
        if i_max is not None: self._i_max = float(i_max)
        if self.use_c_backend:
            self._impl.set_params(
                Kp=self._Kp, Ki=self._Ki, i_max=self._i_max
            )
        else:
            self._impl.Kp    = np.float32(self._Kp)
            self._impl.Ki    = np.float32(max(self._Ki, 1e-9))
            self._impl.i_max = np.float32(self._i_max)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def integrator(self) -> float:
        """Current integrator state."""
        return float(self.state[0])

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return (f"SpeedPIBlock('{self.name}', "
                f"Kp={self._Kp}, Ki={self._Ki}, i_max={self._i_max}, "
                f"backend={be})")

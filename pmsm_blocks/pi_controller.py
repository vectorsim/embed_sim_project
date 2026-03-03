"""
pi_controller.py
================

PIControllerBlock — Scalar PI controller with anti-windup clamping.

Control law:
    output(t) = Kp · e(t) + Ki · ∫e(t) dt

Anti-windup (back-calculation type):
    The integrator is frozen when the output is saturated AND the error
    would drive it further into saturation:
        dI/dt = 0  if  |output| ≥ limit  AND  e · I > 0

Inputs  (port 0): [error]         — single-element VectorSignal
Output          : [control_output] — single-element VectorSignal

State (internal): [integrator_state]   — one floating-point value

C backend interface:
    pmsm_pi_compute(InputSignals*, OutputSignals*)
    InputSignals  { double error;                             }
    OutputSignals { double output;                            }

    The C implementation must maintain its own integrator state
    across calls (e.g. in a static variable or passed via a context struct).

Parameters
----------
    Kp    : float  — Proportional gain         default 1.0
    Ki    : float  — Integral gain             default 10.0
    limit : float  — Output saturation limit   default 100.0
    initial:float  — Initial integrator value  default 0.0

Example
-------
    from pmsm_blocks import PIControllerBlock

    speed_pi = PIControllerBlock("speed_pi", Kp=1.0, Ki=20.0, limit=100.0)
    error_sum >> speed_pi >> iq_ref
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional

from ._path_utils import setup_embedsim_path
setup_embedsim_path()

from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks import VectorSignal


# ==============================================================================
# PIControllerBlock
# ==============================================================================

class PIControllerBlock(SimBlockBase):
    """
    Scalar PI controller with anti-windup — dual Python / C backend.

    State
        self.state[0] — accumulated integral of error

    Output
        Single-element VectorSignal: [control_output]

    Supports Euler integration via integrate_state() and
    get_derivative() for RK4 (external integrator in EmbedSim engine).
    """

    def __init__(
        self,
        name:    str,
        Kp:      float = 1.0,
        Ki:      float = 10.0,
        limit:   float = 100.0,
        initial: float = 0.0,
        use_c_backend: bool = False,
        dtype = None,
    ) -> None:
        """
        Args:
            name:          Block identifier
            Kp:            Proportional gain
            Ki:            Integral gain
            limit:         Symmetric output saturation ±limit
            initial:       Initial integrator state
            use_c_backend: False = Python (default), True = compiled C
            dtype:         Array dtype (default float32 from embedsim)
        """
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.Kp:    float = float(Kp)
        self.Ki:    float = float(Ki)
        self.limit: float = float(limit)

        # ODE state: integral accumulator
        self.state:      np.ndarray = np.array([float(initial)], dtype=np.float64)
        self.derivative: np.ndarray = np.zeros(1,               dtype=np.float64)

        self.is_dynamic  = True
        self.vector_size = 1

        # RK4 staging (allocated once)
        self.k1 = np.zeros(1, dtype=np.float64)
        self.k2 = np.zeros(1, dtype=np.float64)
        self.k3 = np.zeros(1, dtype=np.float64)
        self.k4 = np.zeros(1, dtype=np.float64)

        self._dt: float = 0.0001   # updated from compute dt arg; passed to C wrapper

        if use_c_backend:
            self._load_wrapper()

    # -- Python backend --------------------------------------------------------

    def compute_py(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        Evaluate the PI controller for the current error.

        1. Read error from port 0.
        2. Compute raw output = Kp·e + Ki·I.
        3. Clip to ±limit.
        4. Update derivative (anti-windup).
        """
        if not input_values or len(input_values[0].value) < 1:
            self.output = VectorSignal([0.0], self.name, dtype=self.dtype)
            return self.output

        error  = float(input_values[0].value[0])
        I      = float(self.state[0])

        # Raw PI output
        raw    = self.Kp * error + self.Ki * I
        output = float(np.clip(raw, -self.limit, self.limit))

        # Anti-windup: freeze integrator if saturated and error winding up
        saturated = abs(raw) >= self.limit
        winding_up = error * I > 0.0
        if saturated and winding_up:
            self.derivative[0] = 0.0
        else:
            self.derivative[0] = error

        self.output = VectorSignal(
            np.array([output], dtype=np.float64), self.name, dtype=self.dtype
        )
        return self.output

    # -- ODE hooks -------------------------------------------------------------

    def get_derivative(
        self,
        t: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> np.ndarray:
        """Return dI/dt (with anti-windup applied)."""
        if not input_values or len(input_values[0].value) < 1:
            return np.zeros(1, dtype=np.float64)

        error  = float(input_values[0].value[0])
        I      = float(self.state[0])
        raw    = self.Kp * error + self.Ki * I
        saturated  = abs(raw) >= self.limit
        winding_up = error * I > 0.0

        if saturated and winding_up:
            return np.zeros(1, dtype=np.float64)
        return np.array([error], dtype=np.float64)

    def integrate_state(self, dt: float, solver: str = "euler") -> None:
        """Advance integrator by dt (Euler only; RK4 handled by engine)."""
        if solver == "euler":
            self.state = self.state + self.derivative * dt

    # -- C backend -------------------------------------------------------------

    def compute_c(
        self,
        t: float,
        dt: float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        C-backend compute — uses Python state, same logic as compute_py.

        The C wrapper's internal integrator is NOT used.
        EmbedSim's RK4 drives state integration via get_derivative().
        compute_c only evaluates the PI output from self.state[0].
        """
        if not input_values or len(input_values[0].value) < 1:
            self.output = VectorSignal([0.0], self.name, dtype=self.dtype)
            return self.output

        error = float(input_values[0].value[0])
        I     = float(self.state[0])

        raw    = self.Kp * error + self.Ki * I
        output = float(np.clip(raw, -self.limit, self.limit))

        saturated  = abs(raw) >= self.limit
        winding_up = error * I > 0.0
        if saturated and winding_up:
            self.derivative[0] = 0.0
        else:
            self.derivative[0] = error

        self.output = VectorSignal(
            np.array([output], dtype=np.float64), self.name, dtype=self.dtype
        )
        return self.output

    def _load_wrapper(self) -> None:
        try:
            from pi_controller_wrapper import PmsmPiWrapper
            self._wrapper = PmsmPiWrapper()
            self._wrapper.set_parameters(self.Kp, self.Ki, self.limit, dt=self._dt)
            # is_dynamic stays True — EmbedSim RK4 integrates self.state[0]
        except ImportError:
            raise ImportError(
                "Cython wrapper 'pi_controller_wrapper' not found.\n"
                "Generate with CodeGenEnd.generate_pyx_stub() and compile."
            )

    # -- Repr ------------------------------------------------------------------

    def __repr__(self) -> str:
        be = "C" if self.use_c_backend else "Python"
        return (
            f"PIControllerBlock('{self.name}', backend={be}, "
            f"Kp={self.Kp}, Ki={self.Ki}, limit={self.limit})"
        )

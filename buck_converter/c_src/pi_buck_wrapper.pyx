# pi_buck_wrapper.pyx
# ====================
#
# Cython bridge for PI_Buck_Block_T — following the same pattern as
# speed_pi_wrapper.pyx and smc_wrapper.pyx.
#
# Usage:
#   from pi_buck_wrapper import PI_BuckWrapper
#   pi = PI_BuckWrapper()
#   pi.set_params(Kp=0.1, Ki=5.0, duty_max=0.95, duty_min=0.05, Ts=1e-4)
#   duty = pi.compute(V_ref, V_meas, dt)
#
# Author : EmbedSim Framework
# Version: 1.0.0

import numpy as np
cimport numpy as cnp

ctypedef float         real32_T
ctypedef unsigned char uint8_T

# ─────────────────────────────────────────────────────────────────────────────
# C declarations
# ─────────────────────────────────────────────────────────────────────────────
cdef extern from "pi_buck_controller.h":

    ctypedef struct PI_Buck_Params_T:
        real32_T Kp
        real32_T Ki
        real32_T duty_max
        real32_T duty_min
        real32_T Ts

    ctypedef struct PI_Buck_State_T:
        real32_T integrator
        real32_T prev_error
        real32_T last_output

    ctypedef struct PI_Buck_Block_T:
        PI_Buck_Params_T params
        PI_Buck_State_T  state

    ctypedef struct PI_Buck_Input_T:
        real32_T V_ref
        real32_T V_meas

    ctypedef struct PI_Buck_Output_T:
        real32_T duty

    void PI_Buck_Init      (PI_Buck_Block_T* pPI)
    void PI_Buck_SetParams (PI_Buck_Block_T* pPI,
                            real32_T Kp, real32_T Ki,
                            real32_T duty_max, real32_T duty_min,
                            real32_T Ts)
    void PI_Buck_ResetState(PI_Buck_Block_T* pPI)
    void PI_Buck_Compute   (PI_Buck_Block_T* pPI,
                            const PI_Buck_Input_T* pIn,
                            real32_T dt,
                            PI_Buck_Output_T* pOut)

# ─────────────────────────────────────────────────────────────────────────────
# Python-visible wrapper class
# ─────────────────────────────────────────────────────────────────────────────
cdef class PI_BuckWrapper:
    """
    Cython wrapper for PI_Buck_Block_T.
    Follows the same pattern as SpeedPIWrapper and SMCWrapper.
    """
    cdef PI_Buck_Block_T  _block
    cdef PI_Buck_Input_T  _in
    cdef PI_Buck_Output_T _out

    def __cinit__(self):
        PI_Buck_Init(&self._block)
        self._in.V_ref   = 0.0
        self._in.V_meas  = 0.0
        self._out.duty   = 0.0

    # ── Parameter setters ─────────────────────────────────────────────────────

    def set_params(self,
                   real32_T Kp       = 0.1,
                   real32_T Ki       = 5.0,
                   real32_T duty_max = 0.95,
                   real32_T duty_min = 0.05,
                   real32_T Ts       = 0.0001):
        """Set Kp, Ki, duty limits, and sample time."""
        PI_Buck_SetParams(&self._block, Kp, Ki, duty_max, duty_min, Ts)

    def reset(self):
        """Reset integrator state."""
        PI_Buck_ResetState(&self._block)

    # ── Input / compute / output ──────────────────────────────────────────────

    def set_inputs(self, real32_T V_ref, real32_T V_meas):
        self._in.V_ref  = V_ref
        self._in.V_meas = V_meas

    def set_inputs_array(self, cnp.ndarray u):
        """Set from numpy array [V_ref, V_meas]."""
        if u.shape[0] < 2:
            raise ValueError("Input array must have at least 2 elements")
        self._in.V_ref  = <real32_T> u[0]
        self._in.V_meas = <real32_T> u[1]

    def compute(self,
                real32_T V_ref,
                real32_T V_meas,
                real32_T dt) -> float:
        """
        Execute one PI step.

        Returns
        -------
        duty : float32  (PWM duty cycle [0-1])
        """
        self._in.V_ref  = V_ref
        self._in.V_meas = V_meas
        PI_Buck_Compute(&self._block, &self._in, dt, &self._out)
        return self._out.duty

    def get_output(self) -> float:
        """Return last duty cycle output."""
        return self._out.duty

    def get_integrator(self) -> float:
        """Return current integrator state."""
        return self._block.state.integrator

    def set_integrator(self, real32_T value):
        """Set integrator state (used to sync RK4 state → C wrapper)."""
        self._block.state.integrator = value

    def get_last_output(self) -> float:
        """Return last computed duty cycle."""
        return self._block.state.last_output
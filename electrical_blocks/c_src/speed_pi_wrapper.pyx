# speed_pi_wrapper.pyx
# ====================
#
# Cython bridge for SpeedPI_Block_T — WORLD 1 (Python simulation only).
# Identical pattern to smc_wrapper.pyx and coordinate_transform_wrapper.pyx.
#
# Usage:
#   from speed_pi_wrapper import SpeedPIWrapper
#   pi = SpeedPIWrapper()
#   pi.set_params(Kp=0.5, Ki=5.0, i_max=10.0)
#   id_ref, iq_ref = pi.compute(omega_ref, omega_meas, dt)
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
cdef extern from "speed_pi_controller.h":

    ctypedef struct SpeedPI_Params_T:
        real32_T Kp
        real32_T Ki
        real32_T i_max

    ctypedef struct SpeedPI_State_T:
        real32_T integrator
        real32_T prev_error

    ctypedef struct SpeedPI_Block_T:
        SpeedPI_Params_T params
        SpeedPI_State_T  state

    ctypedef struct SpeedPI_Input_T:
        real32_T omega_ref
        real32_T omega_meas

    ctypedef struct SpeedPI_Output_T:
        real32_T id_ref
        real32_T iq_ref

    void SpeedPI_Init      (SpeedPI_Block_T* pPI)
    void SpeedPI_SetParams (SpeedPI_Block_T* pPI,
                            real32_T Kp, real32_T Ki, real32_T i_max)
    void SpeedPI_ResetState(SpeedPI_Block_T* pPI)
    void SpeedPI_Compute   (SpeedPI_Block_T* pPI,
                            const SpeedPI_Input_T* pIn,
                            real32_T dt,
                            SpeedPI_Output_T* pOut)

# ─────────────────────────────────────────────────────────────────────────────
# Python-visible wrapper class
# ─────────────────────────────────────────────────────────────────────────────
cdef class SpeedPIWrapper:
    """
    Cython wrapper for SpeedPI_Block_T.
    Identical interface to SMCWrapper — drop-in for EmbedSim C backend.
    """
    cdef SpeedPI_Block_T  _block
    cdef SpeedPI_Input_T  _in
    cdef SpeedPI_Output_T _out

    def __cinit__(self):
        SpeedPI_Init(&self._block)
        self._in.omega_ref  = 0.0
        self._in.omega_meas = 0.0
        self._out.id_ref    = 0.0
        self._out.iq_ref    = 0.0

    # ── Parameter setters ─────────────────────────────────────────────────────

    def set_params(self,
                   real32_T Kp    = 0.5,
                   real32_T Ki    = 5.0,
                   real32_T i_max = 10.0):
        """Set Kp, Ki, i_max."""
        SpeedPI_SetParams(&self._block, Kp, Ki, i_max)

    def reset(self):
        """Reset integrator state."""
        SpeedPI_ResetState(&self._block)

    # ── Input / compute / output ──────────────────────────────────────────────

    def set_inputs(self, real32_T omega_ref, real32_T omega_meas):
        self._in.omega_ref  = omega_ref
        self._in.omega_meas = omega_meas

    def set_inputs_array(self, cnp.ndarray u):
        """Set from numpy array [omega_ref, omega_meas]."""
        if u.shape[0] < 2:
            raise ValueError("Input array must have at least 2 elements")
        self._in.omega_ref  = <real32_T> u[0]
        self._in.omega_meas = <real32_T> u[1]

    def compute(self,
                real32_T omega_ref,
                real32_T omega_meas,
                real32_T dt) -> tuple:
        """
        Execute one PI step.

        Returns
        -------
        (id_ref, iq_ref) : tuple of float32
        """
        self._in.omega_ref  = omega_ref
        self._in.omega_meas = omega_meas
        SpeedPI_Compute(&self._block, &self._in, dt, &self._out)
        return (self._out.id_ref, self._out.iq_ref)

    def get_outputs(self):
        """Return last outputs as numpy array [id_ref, iq_ref]."""
        return np.array([self._out.id_ref, self._out.iq_ref], dtype=np.float32)

    def get_integrator(self) -> float:
        """Return current integrator state."""
        return self._block.state.integrator

    def set_integrator(self, real32_T value):
        """Set integrator state (used to sync RK4 state[0] → C wrapper)."""
        self._block.state.integrator = value

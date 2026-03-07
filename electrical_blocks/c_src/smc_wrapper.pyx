# smc_wrapper.pyx
# ===============
#
# Cython bridge for the Sliding Mode Controller (SMC) — WORLD 1 (Python sim).
#
# Exposes SMC_Block_T to Python as a thin, zero-overhead wrapper class.
# real32_T == float on all supported platforms.
#
# On the Aurix target this file is never compiled; C sources are used directly.
#
# Usage:
#   from smc_wrapper import SMCWrapper
#   smc = SMCWrapper()
#   smc.set_params_d(lambda_=500.0, K_sw=24.0, phi=5.0, out_min=-24.0, out_max=24.0)
#   smc.set_params_q(lambda_=500.0, K_sw=24.0, phi=5.0, out_min=-24.0, out_max=24.0)
#   smc.compute(ref_d, ref_q, meas_d, meas_q, dt)  -> (v_d, v_q)
#
# Author : EmbedSim Framework
# Version: 1.0.0

import numpy as np
cimport numpy as cnp

# ─────────────────────────────────────────────────────────────────────────────
# Mirror real32_T / uint8_T from Sys_Types.h
# ─────────────────────────────────────────────────────────────────────────────
ctypedef float     real32_T
ctypedef unsigned char uint8_T

# ─────────────────────────────────────────────────────────────────────────────
# C declarations
# ─────────────────────────────────────────────────────────────────────────────
cdef extern from "sliding_mode_controller.h":

    cdef int SMC_NUM_CHANNELS

    ctypedef struct SMC_Params_T:
        real32_T lambda_  "lambda"
        real32_T K_sw
        real32_T phi
        real32_T out_min
        real32_T out_max

    ctypedef struct SMC_State_T:
        real32_T integral
        real32_T prev_error
        real32_T surface
        real32_T output

    ctypedef struct SMC_Block_T:
        SMC_Params_T params[2]   # SMC_NUM_CHANNELS = 2
        SMC_State_T  state[2]

    ctypedef struct SMC_Input_T:
        real32_T ref_d
        real32_T ref_q
        real32_T meas_d
        real32_T meas_q

    ctypedef struct SMC_Output_T:
        real32_T v_d
        real32_T v_q

    void SMC_Init      (SMC_Block_T* pSMC)
    void SMC_SetParams (SMC_Block_T* pSMC,
                        uint8_T channel,
                        real32_T lambda_,
                        real32_T K_sw,
                        real32_T phi,
                        real32_T out_min,
                        real32_T out_max)
    void SMC_ResetState(SMC_Block_T* pSMC, uint8_T channel)
    void SMC_Compute   (SMC_Block_T* pSMC,
                        const SMC_Input_T* pIn,
                        real32_T dt,
                        SMC_Output_T* pOut)

# ─────────────────────────────────────────────────────────────────────────────
# Python-visible wrapper class
# ─────────────────────────────────────────────────────────────────────────────
cdef class SMCWrapper:
    """
    Cython wrapper for SMC_Block_T.

    Two-channel Sliding Mode Controller operating on the d-q frame.
    Fully equivalent to the embedded C version.
    """
    cdef SMC_Block_T  _block
    cdef SMC_Input_T  _in
    cdef SMC_Output_T _out

    def __cinit__(self):
        SMC_Init(&self._block)
        self._in.ref_d  = 0.0
        self._in.ref_q  = 0.0
        self._in.meas_d = 0.0
        self._in.meas_q = 0.0
        self._out.v_d   = 0.0
        self._out.v_q   = 0.0

    # ── Parameter setters ─────────────────────────────────────────────────────

    def set_params_d(self,
                     real32_T lambda_  = 500.0,
                     real32_T K_sw     = 24.0,
                     real32_T phi      = 5.0,
                     real32_T out_min  = -24.0,
                     real32_T out_max  =  24.0):
        """Configure d-axis (channel 0) SMC parameters."""
        SMC_SetParams(&self._block, 0, lambda_, K_sw, phi, out_min, out_max)

    def set_params_q(self,
                     real32_T lambda_  = 500.0,
                     real32_T K_sw     = 24.0,
                     real32_T phi      = 5.0,
                     real32_T out_min  = -24.0,
                     real32_T out_max  =  24.0):
        """Configure q-axis (channel 1) SMC parameters."""
        SMC_SetParams(&self._block, 1, lambda_, K_sw, phi, out_min, out_max)

    def set_params(self,
                   uint8_T   channel,
                   real32_T  lambda_,
                   real32_T  K_sw,
                   real32_T  phi,
                   real32_T  out_min,
                   real32_T  out_max):
        """Configure any channel by index (0=d, 1=q)."""
        SMC_SetParams(&self._block, channel, lambda_, K_sw, phi, out_min, out_max)

    # ── State control ─────────────────────────────────────────────────────────

    def reset(self, channel: int = 255):
        """Reset integrators.  channel=255 resets all channels."""
        SMC_ResetState(&self._block, <uint8_T>channel)

    # ── Input setters ─────────────────────────────────────────────────────────

    def set_inputs(self,
                   real32_T ref_d,
                   real32_T ref_q,
                   real32_T meas_d,
                   real32_T meas_q):
        """Set d-q reference and measured currents."""
        self._in.ref_d  = ref_d
        self._in.ref_q  = ref_q
        self._in.meas_d = meas_d
        self._in.meas_q = meas_q

    def set_inputs_array(self, cnp.ndarray u):
        """Set inputs from numpy array [ref_d, ref_q, meas_d, meas_q]."""
        if u.shape[0] < 4:
            raise ValueError("Input array must have at least 4 elements")
        self._in.ref_d  = <real32_T> u[0]
        self._in.ref_q  = <real32_T> u[1]
        self._in.meas_d = <real32_T> u[2]
        self._in.meas_q = <real32_T> u[3]

    # ── Compute ───────────────────────────────────────────────────────────────

    def compute(self,
                real32_T ref_d,
                real32_T ref_q,
                real32_T meas_d,
                real32_T meas_q,
                real32_T dt) -> tuple:
        """
        Execute one SMC step.

        Returns
        -------
        (v_d, v_q) : tuple of float32
        """
        self._in.ref_d  = ref_d
        self._in.ref_q  = ref_q
        self._in.meas_d = meas_d
        self._in.meas_q = meas_q
        SMC_Compute(&self._block, &self._in, dt, &self._out)
        return (self._out.v_d, self._out.v_q)

    def get_outputs(self):
        """Return last outputs as numpy array [v_d, v_q]."""
        return np.array([self._out.v_d, self._out.v_q], dtype=np.float32)

    # ── State inspection ──────────────────────────────────────────────────────

    def get_surface(self, uint8_T channel) -> float:
        """Return last sliding surface value for the given channel."""
        if channel >= 2:
            raise ValueError("channel must be 0 (d) or 1 (q)")
        return self._block.state[channel].surface

    def get_integral(self, uint8_T channel) -> float:
        """Return current integrator value for the given channel."""
        if channel >= 2:
            raise ValueError("channel must be 0 (d) or 1 (q)")
        return self._block.state[channel].integral

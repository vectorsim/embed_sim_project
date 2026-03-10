# =============================================================================
# smo_wrapper.pyx
# =============================================================================
#
# Cython wrapper for the Sliding Mode Observer (SMO) C block.
#
# Exposes SMO_Block_T and SMO_Compute() to Python / EmbedSim.
#
# EmbedSim CodeGen attributes
# ----------------------------
#   step_func    : SMO_Compute
#   state_struct : SMO_Block_T
#   C_SOURCES    : ['smo.c']
#   C_HEADERS    : ['smo.h', 'Sys_Types.h']
#   NUM_INPUTS   : 2   (port0:[i_α,i_β]  port1:[v_α,v_β])
#   OUTPUT_SIZE  : 4   ([θ̂_e, ω̂_m, î_d, î_q])
#
# =============================================================================

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import  numpy as np
cimport numpy as cnp

cnp.import_array()

# --------------------------------------------------------------------------
# C declarations
# --------------------------------------------------------------------------
cdef extern from "Sys_Types.h":
    ctypedef float real32_T

cdef extern from "smo.h":
    ctypedef struct SMO_Params_T:
        real32_T R
        real32_T L
        real32_T K_smo
        real32_T wc_emf
        real32_T wc_spd
        real32_T phi
        real32_T p

    ctypedef struct SMO_Block_T:
        SMO_Params_T prm
        real32_T i_alpha_hat
        real32_T i_beta_hat
        real32_T e_alpha_hat
        real32_T e_beta_hat
        real32_T theta_e_hat
        real32_T theta_e_prev
        real32_T omega_e_filt
        real32_T y[4]

    void SMO_Init   (SMO_Block_T *blk, const SMO_Params_T *prm)
    void SMO_Compute(SMO_Block_T *blk,
                     real32_T dt,
                     real32_T i_alpha, real32_T i_beta,
                     real32_T v_alpha, real32_T v_beta,
                     real32_T *y)

# --------------------------------------------------------------------------
# Python-visible wrapper class
# --------------------------------------------------------------------------
cdef class SMOWrapper:
    """
    Thin Cython wrapper around SMO_Block_T (Sliding Mode Observer).

    Usage
    -----
    >>> w = SMOWrapper(R=0.5, L=0.0055, K_smo=20.0, wc_emf=3141.6,
    ...               wc_spd=502.7, phi=0.5, p=2.0)
    >>> y = w.compute(dt, i_alpha, i_beta, v_alpha, v_beta)
    >>> # y = [theta_e_hat, omega_m_hat, id_hat, iq_hat]
    """

    cdef SMO_Block_T  _blk
    cdef SMO_Params_T _prm

    def __cinit__(self,
                  float R       = 0.5,
                  float L       = 0.0055,
                  float K_smo   = 20.0,
                  float wc_emf  = 3141.593,   # 500 Hz * 2π
                  float wc_spd  = 502.655,    # 80  Hz * 2π
                  float phi     = 0.5,
                  float p       = 2.0):
        self._prm.R       = R
        self._prm.L       = L
        self._prm.K_smo   = K_smo
        self._prm.wc_emf  = wc_emf
        self._prm.wc_spd  = wc_spd
        self._prm.phi     = phi
        self._prm.p       = p
        SMO_Init(&self._blk, &self._prm)

    def reset(self):
        """Re-initialise all observer states to zero."""
        SMO_Init(&self._blk, &self._prm)

    def compute(self,
                float dt,
                float i_alpha,
                float i_beta,
                float v_alpha,
                float v_beta):
        """
        Execute one SMO step.

        Parameters
        ----------
        dt      : float  — timestep [s]
        i_alpha : float  — measured stator current α [A]
        i_beta  : float  — measured stator current β [A]
        v_alpha : float  — applied stator voltage α [V]  (z⁻¹ from previous step)
        v_beta  : float  — applied stator voltage β [V]

        Returns
        -------
        numpy.ndarray, shape (4,), dtype float32
            [θ̂_e [rad], ω̂_m [rad/s], î_d [A], î_q [A]]
        """
        cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(4, dtype=np.float32)
        SMO_Compute(&self._blk, dt,
                    i_alpha, i_beta, v_alpha, v_beta,
                    <real32_T *>out.data)
        return out

    def compute_array(self,
                      float dt,
                      cnp.ndarray[cnp.float32_t, ndim=1] i_alphabeta not None,
                      cnp.ndarray[cnp.float32_t, ndim=1] v_alphabeta not None):
        """
        Compute from packed [i_α, i_β] and [v_α, v_β] arrays.

        Returns
        -------
        np.ndarray, shape (4,), dtype float32
        """
        return self.compute(dt,
                            i_alphabeta[0], i_alphabeta[1],
                            v_alphabeta[0], v_alphabeta[1])

    # ── State read-back (for debugging / logging) ─────────────────────────────
    @property
    def theta_e_hat(self):
        return self._blk.theta_e_hat

    @property
    def omega_e_filt(self):
        return self._blk.omega_e_filt

    @property
    def i_alpha_hat(self):
        return self._blk.i_alpha_hat

    @property
    def i_beta_hat(self):
        return self._blk.i_beta_hat

    @property
    def e_alpha_hat(self):
        return self._blk.e_alpha_hat

    @property
    def e_beta_hat(self):
        return self._blk.e_beta_hat

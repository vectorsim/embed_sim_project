# distutils: language = c
# cython: language_level = 3
"""
smo_wrapper.pyx  --  pmsm_blocks/c_src/
Cython wrapper for smo.c (Sliding Mode Observer)
MISRA C:2012 / ASIL-D  --  real32_T interface
"""

import numpy as np
cimport numpy as np
from libc.string cimport memset

# --------------------------------------------------------------------------- #
#  C declarations                                                              #
# --------------------------------------------------------------------------- #
cdef extern from "smo.h":
    ctypedef float real32_T

    ctypedef struct SMO_Params_T:
        real32_T R
        real32_T L
        real32_T K_smo
        real32_T wc_emf
        real32_T wc_spd
        real32_T phi
        int      p

    ctypedef struct SMO_Block_T:
        SMO_Params_T prm
        real32_T i_alpha_hat
        real32_T i_beta_hat
        real32_T emf_alpha
        real32_T emf_beta
        real32_T theta_e
        real32_T omega_m
        real32_T omega_prev

    void SMO_Init   (SMO_Block_T *blk, SMO_Params_T *prm)
    void SMO_Compute(SMO_Block_T *blk,
                     real32_T dt,
                     real32_T i_alpha, real32_T i_beta,
                     real32_T v_alpha, real32_T v_beta,
                     real32_T *y)          # y[4]: theta_e, omega_m, id_hat, iq_hat

# --------------------------------------------------------------------------- #
#  Python wrapper class                                                        #
# --------------------------------------------------------------------------- #
cdef class SMOWrapper:
    """
    Sliding Mode Observer wrapper.

    Parameters
    ----------
    R, L      : float  -- stator resistance [Ohm] and inductance [H]
    K_smo     : float  -- SMO injection gain
    wc_emf    : float  -- back-EMF LPF bandwidth [rad/s]
    wc_spd    : float  -- speed LPF bandwidth [rad/s]
    phi       : float  -- boundary layer thickness [A]
    p         : int    -- pole pairs

    Usage
    -----
    >>> smo = SMOWrapper(R=0.5, L=5.5e-3, K_smo=20.0,
    ...                  wc_emf=3141.6, wc_spd=502.7, phi=0.5, p=2)
    >>> y = smo.compute(dt, i_alpha, i_beta, v_alpha, v_beta)
    >>> theta_e, omega_m, id_hat, iq_hat = y
    """

    cdef SMO_Block_T  _blk
    cdef SMO_Params_T _prm

    def __cinit__(self,
                  float R       = 0.5,
                  float L       = 5.5e-3,
                  float K_smo   = 20.0,
                  float wc_emf  = 3141.593,   # 2*pi*500
                  float wc_spd  = 502.655,    # 2*pi*80
                  float phi     = 0.5,
                  int   p       = 2):
        self._prm.R      = R
        self._prm.L      = L
        self._prm.K_smo  = K_smo
        self._prm.wc_emf = wc_emf
        self._prm.wc_spd = wc_spd
        self._prm.phi    = phi
        self._prm.p      = p
        SMO_Init(&self._blk, &self._prm)

    def reset(self):
        """Reset all observer states to zero."""
        SMO_Init(&self._blk, &self._prm)

    def compute(self,
                float dt,
                float i_alpha, float i_beta,
                float v_alpha, float v_beta):
        """
        Run one observer step.

        Returns
        -------
        numpy.ndarray, dtype=float32, shape=(4,)
            [theta_e_hat, omega_m_hat, id_hat, iq_hat]
        """
        cdef np.ndarray[np.float32_t, ndim=1] y = np.empty(4, dtype=np.float32)
        SMO_Compute(&self._blk, dt, i_alpha, i_beta, v_alpha, v_beta,
                    <real32_T*>y.data)
        return y

    def compute_array(self,
                      float dt,
                      np.ndarray[np.float32_t, ndim=1] i_ab not None,
                      np.ndarray[np.float32_t, ndim=1] v_ab not None):
        """Convenience overload accepting [i_alpha,i_beta] and [v_alpha,v_beta] arrays."""
        if i_ab.shape[0] < 2 or v_ab.shape[0] < 2:
            raise ValueError("i_ab and v_ab must each have at least 2 elements")
        return self.compute(dt, i_ab[0], i_ab[1], v_ab[0], v_ab[1])

    # ---- state read-back properties ---- #
    @property
    def theta_e(self):
        return self._blk.theta_e

    @property
    def omega_m(self):
        return self._blk.omega_m

    @property
    def i_alpha_hat(self):
        return self._blk.i_alpha_hat

    @property
    def i_beta_hat(self):
        return self._blk.i_beta_hat

    @property
    def emf_alpha(self):
        return self._blk.emf_alpha

    @property
    def emf_beta(self):
        return self._blk.emf_beta

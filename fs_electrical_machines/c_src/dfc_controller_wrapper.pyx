# dfc_controller_wrapper.pyx
# =============================================================================
# EmbedSim Differential Flatness Controller -- Cython wrapper
# =============================================================================
#
# \file      dfc_controller_wrapper.pyx
# \brief     Cython wrapper for embed_sim_dfc_controller.c / .h
#
# \details   Exposes DFCControllerWrapper (stateful, per-motor instance) and
#            the dfc_step() convenience function (stateless, single-step).
#            All C function calls are issued with the GIL released (nogil) so
#            the wrapper is safe to call from a Python simulation loop that
#            uses threading.
#
# \note      EKF speed observer declarations (embed_sim_ekf_speed.h) are NOT
#            included here.  They will be added in a separate wrapper file once
#            the DFC build is confirmed stable on hardware.
#
# Location:  fs_electrical_machines/c_src/dfc_controller_wrapper.pyx
# Import:    from fs_electrical_machines.dfc_controller_wrapper import DFCControllerWrapper
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# =============================================================================

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, exp, tanh, atan2

cnp.import_array()


# -----------------------------------------------------------------------------
# \brief  C declarations from embed_sim_matrix.h
# -----------------------------------------------------------------------------
cdef extern from "embed_sim_matrix.h":

    ctypedef int    MatrixElement   # int32_T
    ctypedef float  MatrixFloat     # real32_T

    ctypedef enum MatrixStatus_Type:
        MATRIX_SUCCESS                  = 0
        MATRIX_ERROR_NULL_PTR           = 1
        MATRIX_ERROR_DIMENSION_MISMATCH = 2
        MATRIX_ERROR_SINGULAR           = 3
        MATRIX_ERROR_SIZE_EXCEEDED      = 4
        MATRIX_ERROR_DIV_BY_ZERO        = 5
        MATRIX_ERROR_NOT_SQUARE         = 6
        MATRIX_ERROR_OUT_OF_BOUNDS      = 8

    MatrixElement Matrix_FloatToQ31(const MatrixFloat value) nogil
    MatrixFloat   Matrix_Q31ToFloat(const MatrixElement value) nogil


# -----------------------------------------------------------------------------
# \brief  C declarations from embed_sim_dfc_gains.h
#
# \note   DFC_GainSet_T is defined in embed_sim_dfc_gains.h, not in
#         embed_sim_dfc_controller.h.  The Cython declaration must list every
#         struct field in the same order as the C header so that Cython
#         computes the correct sizeof() when a local instance is built on the
#         stack (e.g. inside set_gains).
# -----------------------------------------------------------------------------
cdef extern from "embed_sim_dfc_gains.h":

    ctypedef struct DFC_GainSet_T:
        MatrixFloat kp_speed   # Speed P-gain [A/(rad/s)]
        MatrixFloat kp_id      # D-axis current P-gain [V/A]
        MatrixFloat kp_iq      # Q-axis current P-gain [V/A]


# -----------------------------------------------------------------------------
# \brief  C declarations from embed_sim_dfc_controller.h
#
# \note   Clarke_T / Park_T / InvPark_T are embedded inside DFC_State_T but
#         are not accessed from the Python layer.  Because cdef extern from
#         defers struct layout to the C compiler, Cython does not need to
#         enumerate those fields -- only fields that are read or written from
#         Cython must appear.  The absent transform fields do NOT corrupt the
#         layout of the fields that follow them.
#
#         The phantom theta_ref field present in the original wrapper has been
#         removed.  It did not exist in the C struct and shifted every
#         subsequent field offset in Cython's view, causing silent memory
#         corruption on all log_* reads.
# -----------------------------------------------------------------------------
cdef extern from "embed_sim_dfc_controller.h":

    # \brief SpeedFusion complementary filter state.
    ctypedef struct DFC_SpeedFusion_T:
        MatrixFloat theta_m_prev     # Previous mechanical angle [rad]
        MatrixFloat omega_enc_filt   # IIR-filtered encoder speed [rad/s]
        MatrixFloat omega_e_prev     # Previous fused electrical speed [rad/s]
        MatrixFloat alpha            # Fusion weight (0 = encoder, 1 = SMO)
        MatrixFloat omega_enc_mech   # Filtered encoder mechanical speed [rad/s]

    # \brief Sliding Mode Observer state (alphabeta frame).
    ctypedef struct DFC_SMO_T:
        MatrixFloat i_hat_alpha   # Estimated alpha-axis current [A]
        MatrixFloat i_hat_beta    # Estimated beta-axis current [A]
        MatrixFloat e_hat_alpha   # Filtered back-EMF alpha [V]
        MatrixFloat e_hat_beta    # Filtered back-EMF beta [V]
        MatrixFloat theta_e_hat   # Estimated electrical angle [rad]
        MatrixFloat omega_e_hat   # Estimated electrical speed [rad/s]
        MatrixFloat theta_e_prev  # Previous angle for speed extraction [rad]

    # \brief Input to DFC_Controller_Step().
    ctypedef struct DFC_Input_T:
        MatrixFloat omega_ref_mech  # Mechanical speed reference [rad/s]
        MatrixFloat theta_m         # Mechanical angle from encoder [rad]
        MatrixFloat ia              # Phase A current [A]
        MatrixFloat ib              # Phase B current [A]
        MatrixFloat ic              # Phase C current [A]

    # \brief Output from DFC_Controller_Step().
    ctypedef struct DFC_Output_T:
        MatrixFloat v_alpha   # Alpha-axis voltage reference [V]
        MatrixFloat v_beta    # Beta-axis voltage reference [V]

    # \brief Full Differential Flatness Controller state.
    ctypedef struct DFC_State_T:
        DFC_SpeedFusion_T fusion        # SpeedFusion state
        DFC_SMO_T         smo           # SMO state
        MatrixFloat       v_alpha_prev  # Alpha voltage delayed one step [V]
        MatrixFloat       v_beta_prev   # Beta voltage delayed one step [V]
        MatrixFloat       iq_ref_prev   # Previous iq_ref for derivative [A]
        MatrixFloat       diq_filt      # LPF-filtered diq_ref/dt [A/s]
        unsigned int      smo_warmup_cnt
        # Clarke_T / Park_T / InvPark_T -- opaque, not accessed from Cython
        MatrixFloat       log_speed_ref   # Speed reference at last log [RPM]
        MatrixFloat       log_iq_ref      # iq reference at last log [A]
        MatrixFloat       log_id          # Measured id at last log [A]
        MatrixFloat       log_iq          # Measured iq at last log [A]
        MatrixFloat       log_alpha       # Fusion weight at last log
        MatrixFloat       log_omega_e     # Fused electrical speed at last log [rad/s]
        unsigned int      log_counter
        MatrixFloat       log_next_time   # Next log threshold [s]

    void DFC_Controller_Init(
        DFC_State_T*      s,
        const MatrixFloat dt) nogil

    void DFC_Controller_Step(
        DFC_State_T*       s,
        const DFC_Input_T* u,
        const MatrixFloat  dt,
        DFC_Output_T*      y) nogil

    void DFC_Controller_Reset(
        DFC_State_T* s) nogil

    void DFC_Controller_GetDiagnostics(
        const DFC_State_T* s,
        MatrixFloat*       speed_ref_rpm,
        MatrixFloat*       iq_ref,
        MatrixFloat*       id,
        MatrixFloat*       iq,
        MatrixFloat*       alpha,
        MatrixFloat*       omega_e) nogil


# =============================================================================
# \class  DFCControllerWrapper
# \brief  Stateful per-motor wrapper for the Differential Flatness FOC Controller.
# =============================================================================
cdef class DFCControllerWrapper:
    """
    Differential Flatness FOC Controller for NANOTEC DB42S02.

    Architecture
    ------------
    SpeedFusion : complementary filter blending encoder IIR and SMO estimates.
    SMO         : Sliding Mode Observer for back-EMF / electrical speed.
    Voltage law : flatness feedforward with id/iq proportional correction terms.

    Parameters
    ----------
    v_dc : float
        DC bus voltage [V].  Default 17.0.
    p_poles : int
        Pole pairs.  Default 4.
    R_s : float
        Stator resistance [Ohm].  Default 0.285.
    L_d : float
        d-axis inductance [H].  Default 3.675e-4.
    L_q : float
        q-axis inductance [H].  Default 3.675e-4.
    lambda_pm : float
        Permanent magnet flux linkage [Wb].  Default 0.0014.
    i_max : float
        Maximum phase current [A].  Default 3.57.
    dt_s : float
        Nominal sampling time [s].  Default 50e-6 (20 kHz).
    kp_speed : float
        Speed P-gain [A/(rad/s)].  Default 0.4.
    kp_id : float
        D-axis current P-gain [V/A].  Default 0.4.
    kp_iq : float
        Q-axis current P-gain [V/A].  Default 8.0.

    Notes
    -----
    Motor parameters and gains are compile-time constants in the C headers
    (embed_sim_dfc_controller.h, embed_sim_dfc_gains.h).  The constructor
    arguments above are accepted for API documentation purposes; they do not
    override the C constants at runtime.  Edit the relevant #define and
    recompile to change them.

    Attributes (readonly)
    ---------------------
    v_alpha : float    Alpha-axis voltage reference [V].
    v_beta  : float    Beta-axis voltage reference [V].
    speed_est : float  Encoder IIR mechanical speed estimate [rad/s].
    iq_ref  : float    q-axis current reference at last diagnostic log [A].
    fusion_alpha : float  SpeedFusion weight (0 = encoder, 1 = SMO).
    omega_e : float    Fused electrical speed at last diagnostic log [rad/s].
    status  : int      0 = success.

    Examples
    --------
    >>> ctrl = DFCControllerWrapper()
    >>> ctrl.set_inputs_individual(209.4, 0.0, 0.0, 0.0, 0.0)
    >>> ctrl.compute(50e-6)
    >>> v_alpha, v_beta = ctrl.get_outputs()
    """

    cdef:
        DFC_State_T   _state
        DFC_Input_T   _input
        DFC_Output_T  _output
        bint          _initialized
        MatrixFloat   _dt

        readonly float  v_alpha
        readonly float  v_beta
        readonly float  speed_est
        readonly float  iq_ref
        readonly float  fusion_alpha
        readonly float  omega_e
        readonly int    status

    # -------------------------------------------------------------------------
    def __cinit__(self,
                  float v_dc      = 17.0,
                  int   p_poles   = 4,
                  float R_s       = 0.285,
                  float L_d       = 3.675e-4,
                  float L_q       = 3.675e-4,
                  float lambda_pm = 0.0014,
                  float i_max     = 3.57,
                  float dt_s      = 50.0e-6,
                  float kp_speed  = 0.4,
                  float kp_id     = 0.4,
                  float kp_iq     = 8.0):

        self._dt          = dt_s
        self._initialized = False
        self.v_alpha      = 0.0
        self.v_beta       = 0.0
        self.speed_est    = 0.0
        self.iq_ref       = 0.0
        self.fusion_alpha = 0.0
        self.omega_e      = 0.0
        self.status       = 0

        with nogil:
            DFC_Controller_Init(&self._state, dt_s)

        self._initialized = True

    # -------------------------------------------------------------------------
    cpdef void set_inputs(self, float[:] u) except *:
        """
        Set controller inputs from a contiguous float32 memoryview.

        Parameters
        ----------
        u : float[5]
            [omega_ref_mech (rad/s), theta_m (rad), ia (A), ib (A), ic (A)]
        """
        if u.shape[0] < 5:
            raise ValueError(
                f"Input array must have at least 5 elements, got {u.shape[0]}")

        self._input.omega_ref_mech = u[0]
        self._input.theta_m        = u[1]
        self._input.ia             = u[2]
        self._input.ib             = u[3]
        self._input.ic             = u[4]

    # -------------------------------------------------------------------------
    cpdef void set_inputs_individual(self,
                                     float omega_ref_mech,
                                     float theta_m,
                                     float ia,
                                     float ib,
                                     float ic) except *:
        """
        Set controller inputs as individual scalar arguments.

        Preferred over set_inputs() in tight simulation loops -- avoids
        NumPy memoryview coercion overhead.

        Parameters
        ----------
        omega_ref_mech : float  Mechanical speed reference [rad/s].
        theta_m        : float  Encoder mechanical angle [rad].
        ia, ib, ic     : float  Phase currents [A].
        """
        self._input.omega_ref_mech = omega_ref_mech
        self._input.theta_m        = theta_m
        self._input.ia             = ia
        self._input.ib             = ib
        self._input.ic             = ic

    # -------------------------------------------------------------------------
    cpdef void set_gains(self,
                         float kp_speed,
                         float kp_id,
                         float kp_iq) except *:
        """
        Request a runtime gain update.

        Not yet implemented -- gains are compile-time constants.
        Edit DFC_KP_SPEED / DFC_KP_ID / DFC_KP_IQ in embed_sim_dfc_gains.h
        and recompile to change them.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Runtime gain update requires DFC_GainSet_Apply() on the C side. "
            "Edit embed_sim_dfc_gains.h and recompile.")

    # -------------------------------------------------------------------------
    cpdef void compute(self, float dt) except *:
        """
        Execute one FOC step.

        Parameters
        ----------
        dt : float  Time step [s].  Typically 50e-6 for 20 kHz GTM ISR.
        """
        cdef MatrixFloat c_dt = dt
        cdef MatrixFloat speed_ref_rpm, iq_ref_diag, id_meas, iq_meas, alpha, omega_e_diag

        if not self._initialized:
            raise RuntimeError(
                "DFCControllerWrapper not initialised. Call __cinit__ first.")

        with nogil:
            DFC_Controller_Step(&self._state, &self._input, c_dt, &self._output)

        self.v_alpha = self._output.v_alpha
        self.v_beta  = self._output.v_beta

        with nogil:
            DFC_Controller_GetDiagnostics(&self._state,
                                          &speed_ref_rpm,
                                          &iq_ref_diag,
                                          &id_meas,
                                          &iq_meas,
                                          &alpha,
                                          &omega_e_diag)

        self.iq_ref       = iq_ref_diag
        self.fusion_alpha = alpha
        self.omega_e      = omega_e_diag
        self.speed_est    = self._state.fusion.omega_enc_mech
        self.status       = 0

    # -------------------------------------------------------------------------
    cpdef cnp.ndarray get_outputs(self):
        """
        Return voltage outputs as a float32 numpy array.

        Returns
        -------
        ndarray, shape (2,), dtype float32
            [v_alpha, v_beta]
        """
        cdef cnp.ndarray[cnp.float32_t, ndim=1] y = np.empty(2, dtype=np.float32)
        y[0] = self.v_alpha
        y[1] = self.v_beta
        return y

    # -------------------------------------------------------------------------
    cpdef cnp.ndarray get_diagnostics(self):
        """
        Return the latest diagnostic snapshot as a float32 numpy array.

        Returns
        -------
        ndarray, shape (7,), dtype float32
            [speed_est_rad_s, iq_ref_A, iq_meas_A, id_meas_A,
             speed_ref_rpm, fusion_alpha, omega_e_rad_s]
        """
        cdef:
            MatrixFloat speed_ref_rpm, iq_ref_diag, id_meas, iq_meas, alpha, omega_e_diag
            cnp.ndarray[cnp.float32_t, ndim=1] diag

        with nogil:
            DFC_Controller_GetDiagnostics(&self._state,
                                          &speed_ref_rpm,
                                          &iq_ref_diag,
                                          &id_meas,
                                          &iq_meas,
                                          &alpha,
                                          &omega_e_diag)

        diag = np.empty(7, dtype=np.float32)
        diag[0] = self._state.fusion.omega_enc_mech
        diag[1] = iq_ref_diag
        diag[2] = iq_meas
        diag[3] = id_meas
        diag[4] = speed_ref_rpm
        diag[5] = alpha
        diag[6] = omega_e_diag
        return diag

    # -------------------------------------------------------------------------
    cpdef void reset(self) except *:
        """
        Reset all integrators and state.  Call on motor stop or fault.
        """
        with nogil:
            DFC_Controller_Reset(&self._state)

        self.v_alpha      = 0.0
        self.v_beta       = 0.0
        self.speed_est    = 0.0
        self.iq_ref       = 0.0
        self.fusion_alpha = 0.0
        self.omega_e      = 0.0
        self.status       = 0

    # -------------------------------------------------------------------------
    cpdef void get_smo_state(self,
                             float[:] e_alpha_beta,
                             float[:] i_hat_alpha_beta) except *:
        """
        Read SMO internal state for debugging.

        Parameters
        ----------
        e_alpha_beta     : float[2]  Out: [e_alpha_filt, e_beta_filt] [V].
        i_hat_alpha_beta : float[2]  Out: [i_hat_alpha, i_hat_beta]   [A].
        """
        if e_alpha_beta.shape[0] < 2 or i_hat_alpha_beta.shape[0] < 2:
            raise ValueError("Output arrays must have at least 2 elements.")

        e_alpha_beta[0]     = self._state.smo.e_hat_alpha
        e_alpha_beta[1]     = self._state.smo.e_hat_beta
        i_hat_alpha_beta[0] = self._state.smo.i_hat_alpha
        i_hat_alpha_beta[1] = self._state.smo.i_hat_beta

    # -------------------------------------------------------------------------
    def __repr__(self):
        return (f"DFCControllerWrapper("
                f"v_alpha={self.v_alpha:.3f} V, "
                f"v_beta={self.v_beta:.3f} V, "
                f"speed_est={self.speed_est:.2f} rad/s, "
                f"iq_ref={self.iq_ref:.3f} A, "
                f"alpha={self.fusion_alpha:.3f})")


# =============================================================================
# \brief  Stateless single-step convenience wrapper.
# =============================================================================
def dfc_step(float omega_ref_mech,
             float theta_m,
             float ia,
             float ib,
             float ic,
             float dt,
             float v_dc      = 17.0,
             int   p_poles   = 4,
             float R_s       = 0.285,
             float L_d       = 3.675e-4,
             float L_q       = 3.675e-4,
             float lambda_pm = 0.0014,
             float i_max     = 3.57,
             float kp_speed  = 0.4,
             float kp_id     = 0.4,
             float kp_iq     = 8.0) -> tuple:
    """
    Single-step DFC controller calculation.

    Creates a fresh controller instance, executes one step, and returns
    outputs.  Not intended for continuous simulation -- use
    DFCControllerWrapper directly to preserve state across steps.

    Parameters
    ----------
    omega_ref_mech : float  Speed reference [rad/s].
    theta_m        : float  Encoder angle [rad].
    ia, ib, ic     : float  Phase currents [A].
    dt             : float  Time step [s].
    Remaining arguments are accepted for API compatibility; the C compile-time
    constants in embed_sim_dfc_controller.h and embed_sim_dfc_gains.h govern
    actual behaviour.

    Returns
    -------
    tuple : (v_alpha, v_beta, speed_est, fusion_alpha, omega_e)
    """
    cdef DFCControllerWrapper ctrl = DFCControllerWrapper(
        v_dc, p_poles, R_s, L_d, L_q, lambda_pm,
        i_max, dt, kp_speed, kp_id, kp_iq)

    ctrl.set_inputs_individual(omega_ref_mech, theta_m, ia, ib, ic)
    ctrl.compute(dt)

    return ctrl.v_alpha, ctrl.v_beta, ctrl.speed_est, ctrl.fusion_alpha, ctrl.omega_e


# =============================================================================
__version__ = "1.1.0"
__author__  = "EmbedSim Team"

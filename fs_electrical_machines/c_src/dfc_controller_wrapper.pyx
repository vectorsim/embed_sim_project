# dfc_controller_wrapper.pyx
# =============================================================================
# EmbedSim Differential Flatness Controller Cython Wrapper for fs_electrical_machines
# =============================================================================
# Cython wrapper for the DFC FOC Controller C implementation.
#
# Location: fs_electrical_machines/c_src/dfc_controller_wrapper.pyx
#
# The compiled module will be available as:
#   from fs_electrical_machines.dfc_controller_wrapper import DFCControllerWrapper
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# =============================================================================

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, exp, tanh, atan2

# Initialize NumPy C API
cnp.import_array()

# -----------------------------------------------------------------------------
# C declarations from embed_sim_matrix.h
# -----------------------------------------------------------------------------
cdef extern from "embed_sim_matrix.h":
    ctypedef int       MatrixElement   # int32_T
    ctypedef float     MatrixFloat     # real32_T

    ctypedef enum MatrixStatus_Type:
        MATRIX_SUCCESS          = 0
        MATRIX_ERROR_NULL_PTR   = 1
        MATRIX_ERROR_DIMENSION_MISMATCH = 2
        MATRIX_ERROR_SINGULAR   = 3
        MATRIX_ERROR_SIZE_EXCEEDED = 4
        MATRIX_ERROR_DIV_BY_ZERO = 5
        MATRIX_ERROR_NOT_SQUARE = 6
        MATRIX_ERROR_OUT_OF_BOUNDS = 8

    MatrixElement Matrix_FloatToQ31(const MatrixFloat value) nogil
    MatrixFloat   Matrix_Q31ToFloat(const MatrixElement value) nogil


# -----------------------------------------------------------------------------
# C declarations from embed_sim_dfc_controller.h
# -----------------------------------------------------------------------------
cdef extern from "embed_sim_dfc_controller.h":

    # Gain structure
    ctypedef struct DFC_GainSet_T:
        MatrixFloat kp_speed
        MatrixFloat kp_id
        MatrixFloat kp_iq

    # SpeedFusion structure
    ctypedef struct DFC_SpeedFusion_T:
        MatrixFloat theta_m_prev
        MatrixFloat omega_enc_filt
        MatrixFloat omega_e_prev
        MatrixFloat alpha
        MatrixFloat omega_enc_mech

    # SMO structure
    ctypedef struct DFC_SMO_T:
        MatrixFloat i_hat_alpha
        MatrixFloat i_hat_beta
        MatrixFloat e_hat_alpha
        MatrixFloat e_hat_beta
        MatrixFloat theta_e_hat
        MatrixFloat omega_e_hat
        MatrixFloat theta_e_prev

    # Input structure
    ctypedef struct DFC_Input_T:
        MatrixFloat omega_ref_mech
        MatrixFloat theta_m
        MatrixFloat ia
        MatrixFloat ib
        MatrixFloat ic

    # Output structure
    ctypedef struct DFC_Output_T:
        MatrixFloat v_alpha
        MatrixFloat v_beta

    # State structure
    ctypedef struct DFC_State_T:
        DFC_SpeedFusion_T fusion
        DFC_SMO_T smo
        MatrixFloat v_alpha_prev
        MatrixFloat v_beta_prev
        MatrixFloat theta_ref
        MatrixFloat iq_ref_prev
        MatrixFloat diq_filt
        unsigned int smo_warmup_cnt
        MatrixFloat log_speed_ref
        MatrixFloat log_iq_ref
        MatrixFloat log_id
        MatrixFloat log_iq
        MatrixFloat log_alpha
        MatrixFloat log_omega_e
        unsigned int log_counter
        MatrixFloat log_next_time

    # Global gains
    extern DFC_GainSet_T g_dfc_gains

    # Function prototypes
    void DFC_Controller_Init(DFC_State_T* s, const MatrixFloat dt) nogil
    void DFC_Controller_Step(
        DFC_State_T* s,
        const DFC_Input_T* u,
        const MatrixFloat dt,
        DFC_Output_T* y) nogil
    void DFC_Controller_Reset(DFC_State_T* s) nogil
    void DFC_Controller_GetDiagnostics(
        const DFC_State_T* s,
        MatrixFloat* speed_ref_rpm,
        MatrixFloat* iq_ref,
        MatrixFloat* id,
        MatrixFloat* iq,
        MatrixFloat* alpha,
        MatrixFloat* omega_e) nogil
    void DFC_GainSet_SetFromSchedule(const DFC_GainSet_T* const src) nogil


# -----------------------------------------------------------------------------
# DFC Controller Wrapper Class
# -----------------------------------------------------------------------------
cdef class DFCControllerWrapper:
    """
    Differential Flatness FOC Controller Wrapper for NANOTEC DB42S02.

    Implements differential flatness control with:
        - SpeedFusion: complementary filter (encoder + SMO)
        - Sliding Mode Observer for back-EMF estimation
        - Flatness voltage law with feedforward cancellation

    Parameters
    ----------
    v_dc : float
        DC bus voltage [V] (default: 17.0)
    p_poles : int
        Number of pole pairs (default: 4)
    R_s : float
        Stator resistance [Ω] (default: 0.285)
    L_d : float
        d-axis inductance [H] (default: 0.0003675)
    L_q : float
        q-axis inductance [H] (default: 0.0003675)
    lambda_pm : float
        Permanent magnet flux linkage [Wb] (default: 0.0014)
    i_max : float
        Maximum current [A] (default: 3.57)
    dt_s : float
        Sampling time [s] (default: 50e-6)
    kp_speed : float
        Speed P-gain [A/(rad/s)] (default: 0.119)
    kp_id : float
        D-axis current P-gain [V/A] (default: 2.0)
    kp_iq : float
        Q-axis current P-gain [V/A] (default: 2.0)

    Attributes
    ----------
    v_alpha, v_beta : float
        Alpha/beta voltage references [V]
    speed_est : float
        Estimated mechanical speed [rad/s]
    iq_ref : float
        q-axis current reference [A]
    fusion_alpha : float
        Current fusion weight (0=encoder, 1=SMO)
    omega_e : float
        Fused electrical speed [rad/s]
    status : int
        Last operation status code (0 = success)

    Examples
    --------
    >>> controller = DFCControllerWrapper()
    >>> inputs = np.array([209.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    >>> controller.set_inputs(inputs)
    >>> controller.compute(50e-6)
    >>> v_alpha, v_beta = controller.get_outputs()
    """

    cdef:
        DFC_State_T      _state
        DFC_Input_T      _input
        DFC_Output_T     _output
        bint             _initialized
        MatrixFloat      _dt

        # Python-visible read-only attributes
        readonly float   v_alpha
        readonly float   v_beta
        readonly float   speed_est
        readonly float   iq_ref
        readonly float   fusion_alpha
        readonly float   omega_e
        readonly int     status

    def __cinit__(self,
                  float v_dc = 17.0,
                  int p_poles = 4,
                  float R_s = 0.285,
                  float L_d = 0.0003675,
                  float L_q = 0.0003675,
                  float lambda_pm = 0.0014,
                  float i_max = 3.57,
                  float dt_s = 50e-6,
                  float kp_speed = 0.119,
                  float kp_id = 2.0,
                  float kp_iq = 2.0):
        """
        Initialize the DFC controller wrapper.
        """
        # Store dt for later use
        self._dt = dt_s
        self._initialized = False
        self.v_alpha = 0.0
        self.v_beta = 0.0
        self.speed_est = 0.0
        self.iq_ref = 0.0
        self.fusion_alpha = 0.0
        self.omega_e = 0.0
        self.status = 0

        # Set gains before init (so Init picks them up)
        cdef DFC_GainSet_T gains
        gains.kp_speed = kp_speed
        gains.kp_id = kp_id
        gains.kp_iq = kp_iq

        with nogil:
            DFC_GainSet_SetFromSchedule(&gains)

        # Initialize C state
        with nogil:
            DFC_Controller_Init(&self._state, dt_s)

        # Mark as initialized
        self._initialized = True

    # -------------------------------------------------------------------------
    cpdef void set_inputs(self, float[:] u) except *:
        """
        Set controller inputs.

        Parameters
        ----------
        u : float[5]
            Input array: [ω_ref_mech (rad/s), θ_m (rad), ia (A), ib (A), ic (A)]
        """
        if u.shape[0] < 5:
            raise ValueError(f"Input array must have at least 5 elements, got {u.shape[0]}")

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
        Set controller inputs using individual arguments.

        Parameters
        ----------
        omega_ref_mech : float
            Mechanical speed reference [rad/s]
        theta_m : float
            Mechanical angle from encoder [rad]
        ia, ib, ic : float
            Phase currents [A]
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
        Update controller gains at runtime.

        Parameters
        ----------
        kp_speed : float
            Speed P-gain [A/(rad/s)]
        kp_id : float
            D-axis current P-gain [V/A]
        kp_iq : float
            Q-axis current P-gain [V/A]
        """
        cdef DFC_GainSet_T gains
        gains.kp_speed = kp_speed
        gains.kp_id = kp_id
        gains.kp_iq = kp_iq

        with nogil:
            DFC_GainSet_SetFromSchedule(&gains)

    # -------------------------------------------------------------------------
    cpdef void compute(self, float dt) except *:
        """
        Execute one control step.

        Parameters
        ----------
        dt : float
            Time step [s] (typically 50e-6 for 20 kHz)
        """
        cdef MatrixFloat c_dt = dt

        if not self._initialized:
            raise RuntimeError("Controller not initialized. Call __cinit__ first.")

        # Call C controller step
        with nogil:
            DFC_Controller_Step(&self._state, &self._input, c_dt, &self._output)

        # Update Python-visible attributes
        self.v_alpha = self._output.v_alpha
        self.v_beta  = self._output.v_beta
        self.iq_ref  = self._state.log_iq_ref
        self.fusion_alpha = self._state.log_alpha
        self.omega_e = self._state.log_omega_e

        # Get diagnostic speed (mechanical)
        cdef MatrixFloat speed_ref_rpm, iq_ref_diag, id_meas, iq_meas, alpha, omega_e_diag
        with nogil:
            DFC_Controller_GetDiagnostics(&self._state,
                                          &speed_ref_rpm,
                                          &iq_ref_diag,
                                          &id_meas,
                                          &iq_meas,
                                          &alpha,
                                          &omega_e_diag)

        # Convert RPM to rad/s for speed estimate
        # speed_est from encoder IIR is stored in fusion.omega_enc_mech
        self.speed_est = self._state.fusion.omega_enc_mech
        self.status = 0  # Success

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
        Return diagnostic data as a float32 numpy array.

        Returns
        -------
        ndarray, shape (7,), dtype float32
            [speed_est_rad_s, iq_ref, iq_meas, id_meas, speed_ref_rpm, fusion_alpha, omega_e_rad_s]
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
        diag[0] = self._state.fusion.omega_enc_mech  # speed_est_rad_s
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
        Reset the controller state.

        Clears all integrators, SMO state, and diagnostic counters.
        """
        with nogil:
            DFC_Controller_Reset(&self._state)

        self.v_alpha = 0.0
        self.v_beta = 0.0
        self.speed_est = 0.0
        self.iq_ref = 0.0
        self.fusion_alpha = 0.0
        self.omega_e = 0.0
        self.status = 0

    # -------------------------------------------------------------------------
    cpdef void get_smo_state(self, float[:] e_alpha_beta, float[:] i_hat_alpha_beta) except *:
        """
        Get SMO internal state for debugging.

        Parameters
        ----------
        e_alpha_beta : float[2]
            Output: [e_alpha_filt, e_beta_filt] (filtered back-EMF) [V]
        i_hat_alpha_beta : float[2]
            Output: [i_hat_alpha, i_hat_beta] (estimated currents) [A]
        """
        if e_alpha_beta.shape[0] < 2 or i_hat_alpha_beta.shape[0] < 2:
            raise ValueError("Output arrays must have at least 2 elements")

        e_alpha_beta[0] = self._state.smo.e_hat_alpha
        e_alpha_beta[1] = self._state.smo.e_hat_beta
        i_hat_alpha_beta[0] = self._state.smo.i_hat_alpha
        i_hat_alpha_beta[1] = self._state.smo.i_hat_beta

    # -------------------------------------------------------------------------
    def __repr__(self):
        return (f"DFCControllerWrapper("
                f"v_alpha={self.v_alpha:.3f}, v_beta={self.v_beta:.3f}, "
                f"speed_est={self.speed_est:.3f}, iq_ref={self.iq_ref:.3f}, "
                f"alpha={self.fusion_alpha:.3f})")


# -----------------------------------------------------------------------------
# Convenience function - uses individual arguments to avoid memoryview issues
# -----------------------------------------------------------------------------
def dfc_step(float omega_ref_mech,
             float theta_m,
             float ia,
             float ib,
             float ic,
             float dt,
             float v_dc=17.0,
             int p_poles=4,
             float R_s=0.285,
             float L_d=0.0003675,
             float L_q=0.0003675,
             float lambda_pm=0.0014,
             float i_max=3.57,
             float kp_speed=0.119,
             float kp_id=2.0,
             float kp_iq=2.0) -> tuple:
    """
    Single-step DFC controller calculation (stateless convenience wrapper).

    Creates a controller instance, executes one step, and returns outputs.

    Parameters
    ----------
    omega_ref_mech : float
        Mechanical speed reference [rad/s]
    theta_m : float
        Mechanical angle from encoder [rad]
    ia, ib, ic : float
        Phase currents [A]
    dt : float
        Time step [s]
    v_dc, p_poles, R_s, L_d, L_q, lambda_pm, i_max :
        Motor parameters
    kp_speed, kp_id, kp_iq :
        Controller gains

    Returns
    -------
    tuple : (v_alpha, v_beta, speed_est, fusion_alpha, omega_e)
        Voltage references [V], estimated speed [rad/s], fusion weight, electrical speed [rad/s]
    """
    cdef DFCControllerWrapper ctrl = DFCControllerWrapper(
        v_dc, p_poles, R_s, L_d, L_q, lambda_pm,
        i_max, dt, kp_speed, kp_id, kp_iq)

    # Use individual argument method to avoid memoryview coercion
    ctrl.set_inputs_individual(omega_ref_mech, theta_m, ia, ib, ic)
    ctrl.compute(dt)

    return ctrl.v_alpha, ctrl.v_beta, ctrl.speed_est, ctrl.fusion_alpha, ctrl.omega_e


# -----------------------------------------------------------------------------
# Version info
# -----------------------------------------------------------------------------
__version__ = "1.0.0"
__author__ = "EmbedSim Team"
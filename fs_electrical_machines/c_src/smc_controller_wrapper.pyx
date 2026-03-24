# smc_controller_wrapper.pyx
# =============================================================================
# EmbedSim SMC Controller Cython Wrapper for fs_electrical_machines
# =============================================================================
# Cython wrapper for the SMC FOC Controller C implementation.
#
# Location: fs_electrical_machines/c_src/smc_controller_wrapper.pyx
#
# The compiled module will be available as:
#   from fs_electrical_machines.smc_controller_wrapper import SMCControllerWrapper
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# =============================================================================

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs

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
# C declarations from embed_sim_smc_controller.h
# -----------------------------------------------------------------------------
cdef extern from "embed_sim_smc_controller.h":

    # Input structure
    ctypedef struct SMC_Input_T:
        MatrixFloat omega_ref_mech
        MatrixFloat theta_m
        MatrixFloat ia
        MatrixFloat ib
        MatrixFloat ic

    # Output structure
    ctypedef struct SMC_Output_T:
        MatrixFloat v_alpha
        MatrixFloat v_beta

    # State structure
    ctypedef struct SMC_Controller_T:
        MatrixFloat int_spd
        MatrixFloat int2_spd
        MatrixFloat iq_ref
        MatrixFloat id_ref
        MatrixFloat vd
        MatrixFloat vq
        MatrixFloat log_speed
        MatrixFloat log_speed_ref
        MatrixFloat log_iq_meas
        MatrixFloat log_id_meas
        unsigned int log_counter
        MatrixFloat log_next_time

    # Function prototypes
    void SMC_Controller_Init(SMC_Controller_T* s, const MatrixFloat dt) nogil
    void SMC_Controller_Step(
        SMC_Controller_T* s,
        const SMC_Input_T* u,
        const MatrixFloat dt,
        SMC_Output_T* y) nogil
    void SMC_Controller_Reset(SMC_Controller_T* s) nogil
    void SMC_Controller_GetDiagnostics(
        const SMC_Controller_T* s,
        MatrixFloat* speed,
        MatrixFloat* speed_ref,
        MatrixFloat* iq,
        MatrixFloat* id) nogil


# -----------------------------------------------------------------------------
# SMC Controller Wrapper Class
# -----------------------------------------------------------------------------
cdef class SMCControllerWrapper:
    """
    SMC FOC Controller Wrapper for NANOTEC DB42S02.

    Implements pure Sliding Mode Control (SMC) with integral sliding surface:
        - Speed SMC: s = e + λ·∫e + γ·∫∫e
        - Current SMC: equivalent control + switching with boundary layer

    Parameters
    ----------
    v_dc : float
        DC bus voltage [V] (default: 17.0)
    p_poles : int
        Number of pole pairs (default: 4)
    R_s : float
        Stator resistance [Ω] (default: 0.19)
    L_d : float
        d-axis inductance [H] (default: 0.000125)
    L_q : float
        q-axis inductance [H] (default: 0.000125)
    lambda_pm : float
        Permanent magnet flux linkage [Wb] (default: 0.0014)
    J_rotor : float
        Rotor inertia [kg·m²] (default: 2.4e-6)
    B_friction : float
        Friction coefficient [N·m·s/rad] (default: 1e-6)
    i_max : float
        Maximum current [A] (default: 3.57)
    dt_s : float
        Sampling time [s] (default: 50e-6)

    Attributes
    ----------
    v_alpha, v_beta : float
        Alpha/beta voltage references [V]
    speed_est : float
        Estimated mechanical speed [rad/s]
    iq_ref : float
        q-axis current reference [A]
    status : int
        Last operation status code (0 = success)

    Examples
    --------
    >>> controller = SMCControllerWrapper()
    >>> inputs = np.array([209.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    >>> controller.set_inputs(inputs)
    >>> controller.compute(50e-6)
    >>> v_alpha, v_beta = controller.get_outputs()
    """

    cdef:
        SMC_Controller_T _state
        SMC_Input_T      _input
        SMC_Output_T     _output
        bint             _initialized
        MatrixFloat      _dt

        # Python-visible read-only attributes
        readonly float   v_alpha
        readonly float   v_beta
        readonly float   speed_est
        readonly float   iq_ref
        readonly int     status

    def __cinit__(self,
                  float v_dc = 17.0,
                  int p_poles = 4,
                  float R_s = 0.19,
                  float L_d = 0.000125,
                  float L_q = 0.000125,
                  float lambda_pm = 0.0014,
                  float J_rotor = 2.4e-6,
                  float B_friction = 1e-6,
                  float i_max = 3.57,
                  float dt_s = 50e-6):
        """
        Initialize the SMC controller wrapper.
        """
        # Store dt for later use
        self._dt = dt_s
        self._initialized = False
        self.v_alpha = 0.0
        self.v_beta = 0.0
        self.speed_est = 0.0
        self.iq_ref = 0.0
        self.status = 0

        # Initialize C state (dt is unused in C init but kept for API)
        with nogil:
            SMC_Controller_Init(&self._state, dt_s)

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
            SMC_Controller_Step(&self._state, &self._input, c_dt, &self._output)

        # Update Python-visible attributes
        self.v_alpha = self._output.v_alpha
        self.v_beta  = self._output.v_beta
        self.iq_ref  = self._state.iq_ref

        # Get diagnostic speed
        cdef MatrixFloat speed, speed_ref, iq, id
        with nogil:
            SMC_Controller_GetDiagnostics(&self._state, &speed, &speed_ref, &iq, &id)
        self.speed_est = speed
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
        ndarray, shape (5,), dtype float32
            [speed_est, iq_ref, iq_meas, id_meas, speed_ref]
        """
        cdef:
            MatrixFloat speed, speed_ref, iq, id
            cnp.ndarray[cnp.float32_t, ndim=1] diag

        with nogil:
            SMC_Controller_GetDiagnostics(&self._state, &speed, &speed_ref, &iq, &id)

        diag = np.empty(5, dtype=np.float32)
        diag[0] = speed
        diag[1] = self._state.iq_ref
        diag[2] = self._state.log_iq_meas
        diag[3] = self._state.log_id_meas
        diag[4] = speed_ref
        return diag

    # -------------------------------------------------------------------------
    cpdef void reset(self) except *:
        """
        Reset the controller state.

        Clears all integrators and diagnostic counters.
        """
        with nogil:
            SMC_Controller_Reset(&self._state)

        self.v_alpha = 0.0
        self.v_beta = 0.0
        self.speed_est = 0.0
        self.iq_ref = 0.0
        self.status = 0

    # -------------------------------------------------------------------------
    def __repr__(self):
        return (f"SMCControllerWrapper("
                f"v_alpha={self.v_alpha:.3f}, v_beta={self.v_beta:.3f}, "
                f"speed_est={self.speed_est:.3f}, iq_ref={self.iq_ref:.3f})")


# -----------------------------------------------------------------------------
# Convenience function - uses individual arguments to avoid memoryview issues
# -----------------------------------------------------------------------------
def smc_step(float omega_ref_mech,
             float theta_m,
             float ia,
             float ib,
             float ic,
             float dt,
             float v_dc=17.0,
             int p_poles=4,
             float R_s=0.19,
             float L_d=0.000125,
             float L_q=0.000125,
             float lambda_pm=0.0014,
             float J_rotor=2.4e-6,
             float B_friction=1e-6,
             float i_max=3.57) -> tuple:
    """
    Single-step SMC controller calculation (stateless convenience wrapper).

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
    v_dc, p_poles, R_s, L_d, L_q, lambda_pm, J_rotor, B_friction, i_max :
        Motor parameters

    Returns
    -------
    tuple : (v_alpha, v_beta, speed_est)
        Voltage references [V] and estimated speed [rad/s]
    """
    cdef SMCControllerWrapper ctrl = SMCControllerWrapper(
        v_dc, p_poles, R_s, L_d, L_q, lambda_pm,
        J_rotor, B_friction, i_max, dt)

    # Use individual argument method to avoid memoryview coercion
    ctrl.set_inputs_individual(omega_ref_mech, theta_m, ia, ib, ic)
    ctrl.compute(dt)

    return ctrl.v_alpha, ctrl.v_beta, ctrl.speed_est


# -----------------------------------------------------------------------------
# Version info
# -----------------------------------------------------------------------------
__version__ = "1.0.0"
__author__ = "EmbedSim Team"
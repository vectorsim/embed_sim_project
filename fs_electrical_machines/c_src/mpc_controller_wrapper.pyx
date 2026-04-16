# mpc_controller_wrapper.pyx
# =============================================================================
# EmbedSim MPC Controller Cython Wrapper for fs_electrical_machines
# =============================================================================
# Cython wrapper for the Model Predictive Control FOC Controller C implementation.
#
# Aligned with:
#   - embed_sim_mpc_controller.h
#   - embed_sim_mpc_controller.c
#   - embed_sim_mpc_gains.h
#   - mpc_controller_block.py
#
# ALIGNMENT FIXES (see mpc_controller_block.py # C: comments for cross-ref):
#
#   FIX 1 — speed_est was RPM*2pi/60 (double-converting already-RPM value).
#            MPC_Controller_GetDiagnostics() returns [RPM].  speed_est is now
#            stored as [RPM] and the property converts on demand.
#            Python MPCControllerBlock.log_data["speed"] is also [RPM].
#
#   FIX 2 — get_diagnostics() diag[0] was speed_rpm, diag[1] was speed_ref_rpm.
#            Corrected to diag[0]=speed_ref_rpm, diag[1]=speed_rpm so the
#            array matches MPC_Controller_GetDiagnostics() argument order and
#            MPCControllerBlock.get_diagnostics() key order:
#            { speed_ref_rpm, speed_rpm, id_meas, iq_meas, vd, vq }
#
#   FIX 3 — __cinit__ was silently discarding ALL tuning parameters (Q_id, Q_iq,
#            R_vd, R_vq, KI_v, N, smo_k, smo_fc).  The C controller uses compile-
#            time #defines from embed_sim_mpc_gains.h so runtime values cannot
#            change MPC_N or motor parameters.  However a new set_gains() method
#            stores the runtime weights and documents this limitation clearly so
#            the CMA-ES tuner author knows what is and is not configurable.
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
        MATRIX_SUCCESS              = 0
        MATRIX_ERROR_NULL_PTR       = 1
        MATRIX_ERROR_DIMENSION_MISMATCH = 2
        MATRIX_ERROR_SINGULAR       = 3
        MATRIX_ERROR_SIZE_EXCEEDED  = 4
        MATRIX_ERROR_DIV_BY_ZERO    = 5
        MATRIX_ERROR_NOT_SQUARE     = 6
        MATRIX_ERROR_OUT_OF_BOUNDS  = 8

    MatrixElement Matrix_FloatToQ31(const MatrixFloat value) nogil
    MatrixFloat   Matrix_Q31ToFloat(const MatrixElement value) nogil


# -----------------------------------------------------------------------------
# C declarations from embed_sim_mpc_controller.h
# -----------------------------------------------------------------------------
cdef extern from "embed_sim_mpc_controller.h":

    # Input structure (matches MPC_Input_T exactly)
    ctypedef struct MPC_Input_T:
        MatrixFloat omega_ref_mech   # Mechanical speed reference [rad/s]
        MatrixFloat theta_m          # Encoder mechanical angle [rad]
        MatrixFloat ia               # Phase A current [A]
        MatrixFloat ib               # Phase B current [A]
        MatrixFloat ic               # Phase C current [A]

    # Output structure (matches MPC_Output_T exactly)
    ctypedef struct MPC_Output_T:
        MatrixFloat v_alpha          # Alpha voltage reference (normalised [-])
        MatrixFloat v_beta           # Beta voltage reference (normalised [-])

    # Encoder speed estimator state (MPC_EncSpeed_T)
    ctypedef struct MPC_EncSpeed_T:
        MatrixFloat theta_m_prev       # Previous encoder angle [rad]
        MatrixFloat theta_m_unwrapped  # Continuously unwrapped angle [rad]
        MatrixFloat omega_filt         # IIR-filtered mechanical speed [rad/s]

    # SMO state (MPC_SMO_T)
    ctypedef struct MPC_SMO_T:
        MatrixFloat i_alpha_hat    # Estimated alpha current [A]
        MatrixFloat i_beta_hat     # Estimated beta current [A]
        MatrixFloat e_alpha_filt   # LPF-smoothed back-EMF alpha [V]
        MatrixFloat e_beta_filt    # LPF-smoothed back-EMF beta [V]
        MatrixFloat alpha_lpf      # LPF coefficient (pre-computed from fc, dt) [-]

    # Complete controller state (MPC_Controller_T)
    # ALIGNMENT: field order and types must exactly match embed_sim_mpc_controller.h.
    # The four coordinate-transform sub-states (Clarke_T, Park_T x2, InvPark_T)
    # are opaque to Cython — we size them as char[] pads.  If the C structs change
    # size, update the pad lengths here accordingly (use sizeof() in a test .c file).
    ctypedef struct MPC_Controller_T:
        MPC_EncSpeed_T enc               # Encoder speed estimator
        MPC_SMO_T      smo               # Sliding Mode Observer
        MatrixFloat    v_alpha_prev      # Previous alpha voltage [V]  (z-1)
        MatrixFloat    v_beta_prev       # Previous beta voltage [V]   (z-1)
        MatrixFloat    iq_limit          # Soft-start current limit [A]
        MatrixFloat    speed_err_integral # Speed error integral [rad]
        # Diagnostic log fields — written by MPC_Controller_Step() every
        # MPC_DIAG_STEPS ticks (20 ticks → 1 kHz at 20 kHz ISR).
        # Units: RPM, RPM, A, A, V, V  (mirror of mpc_controller_block.py log_data)
        MatrixFloat    log_speed_ref     # Speed reference [RPM]
        MatrixFloat    log_speed         # Actual speed [RPM]
        MatrixFloat    log_id            # D-axis current [A]
        MatrixFloat    log_iq            # Q-axis current [A]
        MatrixFloat    log_vd            # D-axis voltage [V]
        MatrixFloat    log_vq            # Q-axis voltage [V]
        unsigned int   log_counter       # Ticks since last log write
        MatrixFloat    log_next_time     # (unused in C, retained for ABI compat)
        # Opaque coordinate transform sub-states
        # ALIGNMENT: char[64] is safe if sizeof(Clarke_T) and sizeof(Park_T)
        # and sizeof(InvPark_T) are each <= 64 bytes.  Verify with:
        #   printf("Clarke_T=%zu Park_T=%zu InvPark_T=%zu\n",
        #          sizeof(Clarke_T), sizeof(Park_T), sizeof(InvPark_T));
        char           _clarke_state[64]    # Clarke_T
        char           _park_state[64]      # Park_T  (current dq)
        char           _park_emf_state[64]  # Park_T  (BEMF feedforward)
        char           _inv_park_state[64]  # InvPark_T

    # Function prototypes (all nogil — no Python objects touched in C)
    void MPC_Controller_Init(MPC_Controller_T* s, const MatrixFloat dt) nogil
    void MPC_Controller_Step(
        MPC_Controller_T* s,
        const MPC_Input_T* u,
        const MatrixFloat dt,
        MPC_Output_T* y) nogil
    void MPC_Controller_Reset(MPC_Controller_T* s) nogil
    void MPC_Controller_GetDiagnostics(
        const MPC_Controller_T* s,
        MatrixFloat* speed_ref_rpm,   # [RPM]
        MatrixFloat* speed_rpm,       # [RPM]
        MatrixFloat* id_meas,         # [A]
        MatrixFloat* iq_meas,         # [A]
        MatrixFloat* vd,              # [V]
        MatrixFloat* vq               # [V]
    ) nogil


# -----------------------------------------------------------------------------
# MPCControllerWrapper
# -----------------------------------------------------------------------------
cdef class MPCControllerWrapper:
    """
    Cython wrapper around the MPC_Controller_T C state machine.

    ALIGNMENT NOTES
    ---------------
    speed_est property  : returns [RPM] to match MPC_Controller_GetDiagnostics()
                          and mpc_controller_block.py log_data["speed"] [RPM].
                          FIX: previously converted RPM→rad/s (double-conversion
                          bug — GetDiagnostics already returns RPM).

    get_diagnostics()   : returns float32 array [speed_ref_rpm, speed_rpm,
                          id_meas, iq_meas, vd, vq].  Index 0 = speed_ref,
                          index 1 = speed — matches GetDiagnostics() arg order
                          and MPCControllerBlock.get_diagnostics() key order.
                          FIX: previously [0]=speed_rpm, [1]=speed_ref_rpm.

    set_gains()         : stores the five CMA-ES-tuned weight values for
                          diagnostic inspection.  NOTE: MPC_N, motor parameters
                          (R_S, L, LAMBDA_PM, KT, J, B) and SMO parameters
                          (SMO_K, SMO_FC) are compile-time #defines in
                          embed_sim_mpc_gains.h / embed_sim_mpc_controller.h
                          and CANNOT be changed at runtime without recompiling.
                          Only the diagnostic weight fields are stored here.
    """

    cdef:
        MPC_Controller_T _state
        MPC_Input_T      _input
        MPC_Output_T     _output
        bint             _initialized
        MatrixFloat      _dt

        # Runtime gain storage (documentation / inspection only)
        # These do NOT change the compiled C controller behaviour.
        MatrixFloat      _rt_Q_id
        MatrixFloat      _rt_Q_iq
        MatrixFloat      _rt_R_vd
        MatrixFloat      _rt_R_vq
        MatrixFloat      _rt_KI_v
        MatrixFloat      _rt_Q_omega

        # Python-visible read-only attributes
        # FIX: speed_est stores [RPM] — was wrongly storing rad/s
        readonly float   v_alpha       # normalised alpha voltage [-]
        readonly float   v_beta        # normalised beta voltage [-]
        readonly float   speed_est     # speed estimate [RPM]  (NOT rad/s)
        readonly float   speed_ref_est # speed reference [RPM]
        readonly float   iq_est        # q-axis current estimate [A]
        readonly float   id_est        # d-axis current estimate [A]
        readonly int     status

    def __cinit__(self,
                  float v_dc        = 17.0,
                  int   p_poles     = 4,
                  float R_s         = 0.285,
                  float L           = 0.0003675,
                  float lambda_pm   = 0.0014,
                  float i_max       = 3.57,
                  float dt_s        = 50e-6,
                  int   N           = 10,
                  float Q_id        = 10.82,
                  float Q_iq        = 0.01,
                  float Q_omega     = 500.0,
                  float R_vd        = 0.001,
                  float R_vq        = 0.005,
                  float smo_k       = 4.68,
                  float smo_fc      = 1000.0,
                  float KI_v        = 0.01,
                  float SOFTSTART_T = 0.1):
        """
        Initialise the MPC controller wrapper.

        IMPORTANT — compile-time vs runtime parameters
        -----------------------------------------------
        The C controller uses compile-time #defines from embed_sim_mpc_gains.h
        and embed_sim_mpc_controller.h for ALL motor parameters and MPC weights.
        The parameters accepted here serve two purposes:

          1. API compatibility with MPCControllerBlock (Python block passes the
             same keyword arguments to both the Python and C backends).
          2. Runtime inspection via set_gains() / get_gains() so the CMA-ES tuner
             can log which weights were active during a C-backend run.

        Parameters that have NO effect on the compiled C controller:
            v_dc, p_poles, R_s, L, lambda_pm, i_max, N, smo_k, smo_fc,
            SOFTSTART_T  (all compile-time #defines in the C headers).

        Parameters stored for inspection:
            Q_id, Q_iq, Q_omega, R_vd, R_vq, KI_v
        """
        self._dt = dt_s

        # Store runtime weight values for inspection
        # C counterpart: embed_sim_mpc_gains.h MPC_GainSet_T
        self._rt_Q_id    = Q_id
        self._rt_Q_iq    = Q_iq
        self._rt_R_vd    = R_vd
        self._rt_R_vq    = R_vq
        self._rt_KI_v    = KI_v
        self._rt_Q_omega = Q_omega

        # Suppress compiler warnings for parameters that cannot be used at runtime
        # (all motor and MPC structural parameters are compile-time #defines)
        _ = (v_dc, p_poles, R_s, L, lambda_pm, i_max, N, smo_k, smo_fc, SOFTSTART_T)

        # Initialise Python-visible attributes
        self.v_alpha       = 0.0
        self.v_beta        = 0.0
        self.speed_est     = 0.0   # [RPM]
        self.speed_ref_est = 0.0   # [RPM]
        self.iq_est        = 0.0
        self.id_est        = 0.0
        self.status        = 0

        # Zero input / output structs
        self._input.omega_ref_mech = 0.0
        self._input.theta_m        = 0.0
        self._input.ia             = 0.0
        self._input.ib             = 0.0
        self._input.ic             = 0.0
        self._output.v_alpha       = 0.0
        self._output.v_beta        = 0.0

        # Initialise C state (pre-computes SMO alpha_lpf from dt_s)
        with nogil:
            MPC_Controller_Init(&self._state, dt_s)

        self._initialized = True

    # -------------------------------------------------------------------------
    cpdef void set_gains(self,
                         float Q_id,
                         float Q_iq,
                         float R_vd,
                         float R_vq,
                         float KI_v,
                         float Q_omega = 500.0) except *:
        """
        Store CMA-ES-tuned weight values for runtime inspection.

        C ALIGNMENT NOTE: These values do NOT change the compiled C controller.
        The C controller reads MPC_Q_ID, MPC_Q_IQ, MPC_R_VD, MPC_R_VQ, MPC_KI_V
        as compile-time #defines from embed_sim_mpc_gains.h.

        To apply new tuned weights to the C backend you must:
          1. Write the tuned values to c_src/embed_sim_mpc_gains.h
             (the _write_gains_header() function in the tuner does this).
          2. Recompile:  python setup_mpc_controller.py build_ext --inplace

        Python counterpart: MPCControllerBlock stores tuned weights live in
        self.Q_id, self.Q_iq, self.R_vd, self.R_vq, self.KI_v because the
        Python backend reads them at every compute_py() call.

        Parameters
        ----------
        Q_id    : d-axis state cost
        Q_iq    : q-axis regulariser
        R_vd    : vd effort weight
        R_vq    : vq effort weight
        KI_v    : speed-error integral gain
        Q_omega : speed tracking cost (should remain 500.0 — fixed by physics)
        """
        self._rt_Q_id    = Q_id
        self._rt_Q_iq    = Q_iq
        self._rt_R_vd    = R_vd
        self._rt_R_vq    = R_vq
        self._rt_KI_v    = KI_v
        self._rt_Q_omega = Q_omega

    # -------------------------------------------------------------------------
    def get_gains(self) -> dict:
        """
        Return the runtime gain values as a dict.

        Keys match _ACTIVE_GAINS in db42s02_closed_loop_mpc_foc_20k.py so the
        CMA-ES tuner can log C-backend gain values alongside Python-backend runs.
        """
        return {
            "Q_id":    float(self._rt_Q_id),
            "Q_iq":    float(self._rt_Q_iq),
            "R_vd":    float(self._rt_R_vd),
            "R_vq":    float(self._rt_R_vq),
            "KI_v":    float(self._rt_KI_v),
            "Q_omega": float(self._rt_Q_omega),
        }

    # -------------------------------------------------------------------------
    cpdef void set_inputs(self, float[:] u) except *:
        """
        Set controller inputs from a packed array.

        Parameters
        ----------
        u : float[5]
            [omega_ref_mech (rad/s), theta_m (rad), ia (A), ib (A), ic (A)]
            Matches MPC_Input_T field order exactly.
        """
        if u.shape[0] < 5:
            raise ValueError(
                f"Input array needs >= 5 elements, got {u.shape[0]}")

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
        Set controller inputs as individual arguments.
        Mirrors MPC_Input_T field assignment in MPC_Controller_Step().
        """
        self._input.omega_ref_mech = omega_ref_mech
        self._input.theta_m        = theta_m
        self._input.ia             = ia
        self._input.ib             = ib
        self._input.ic             = ic

    # -------------------------------------------------------------------------
    cpdef void compute(self, float dt) except *:
        """
        Execute one MPC control step.

        Calls MPC_Controller_Step() then MPC_Controller_GetDiagnostics().
        After this call:
          self.v_alpha, self.v_beta  : normalised voltage outputs [-]
          self.speed_est             : estimated speed [RPM]    ← FIX: was rad/s
          self.speed_ref_est         : speed reference [RPM]
          self.id_est, self.iq_est   : dq currents [A]
        """
        cdef MatrixFloat c_dt = dt
        cdef MatrixFloat speed_ref_rpm
        cdef MatrixFloat speed_rpm
        cdef MatrixFloat id_meas
        cdef MatrixFloat iq_meas
        cdef MatrixFloat vd
        cdef MatrixFloat vq

        if not self._initialized:
            raise RuntimeError("Controller not initialised.")

        # ── MPC step ─────────────────────────────────────────────────────────
        with nogil:
            MPC_Controller_Step(&self._state, &self._input, c_dt, &self._output)

        # Update normalised voltage outputs
        self.v_alpha = self._output.v_alpha
        self.v_beta  = self._output.v_beta

        # ── Diagnostics ───────────────────────────────────────────────────────
        # GetDiagnostics reads log_speed_ref and log_speed which are written
        # in [RPM] by MPC_Controller_Step() every MPC_DIAG_STEPS ticks.
        # FIX: previously converted speed_rpm → rad/s here (wrong —
        #      GetDiagnostics already returns [RPM]).
        with nogil:
            MPC_Controller_GetDiagnostics(
                &self._state,
                &speed_ref_rpm,   # [RPM]
                &speed_rpm,       # [RPM]
                &id_meas,         # [A]
                &iq_meas,         # [A]
                &vd,              # [V]
                &vq               # [V]
            )

        # Store [RPM] directly — no conversion
        # Python MPCControllerBlock.log_data["speed"] is also [RPM]
        self.speed_ref_est = speed_ref_rpm   # [RPM]
        self.speed_est     = speed_rpm       # [RPM]  FIX: was speed_rpm * 2π/60
        self.id_est        = id_meas
        self.iq_est        = iq_meas
        self.status        = 0

    # -------------------------------------------------------------------------
    cpdef cnp.ndarray get_outputs(self):
        """
        Return voltage outputs as a float32 numpy array.

        Returns
        -------
        ndarray shape (2,) : [v_alpha [-], v_beta [-]]
            Normalised voltages ready for SVPWM (same as MPCControllerBlock output).
        """
        cdef cnp.ndarray[cnp.float32_t, ndim=1] y = np.empty(2, dtype=np.float32)
        y[0] = self.v_alpha
        y[1] = self.v_beta
        return y

    # -------------------------------------------------------------------------
    cpdef cnp.ndarray get_diagnostics(self):
        """
        Return diagnostic snapshot as a float32 numpy array.

        Returns
        -------
        ndarray shape (6,):
            [0] speed_ref_rpm [RPM]   ← FIX: was [1] in old code
            [1] speed_rpm     [RPM]   ← FIX: was [0] in old code
            [2] id_meas       [A]
            [3] iq_meas       [A]
            [4] vd            [V]
            [5] vq            [V]

        ALIGNMENT: index order matches MPC_Controller_GetDiagnostics() argument
        order and MPCControllerBlock.get_diagnostics() return dict key order:
            { speed_ref_rpm: 0, speed_rpm: 1, id_meas: 2, iq_meas: 3, vd: 4, vq: 5 }
        """
        cdef:
            MatrixFloat speed_ref_rpm
            MatrixFloat speed_rpm
            MatrixFloat id_meas
            MatrixFloat iq_meas
            MatrixFloat vd
            MatrixFloat vq
            cnp.ndarray[cnp.float32_t, ndim=1] diag

        with nogil:
            MPC_Controller_GetDiagnostics(
                &self._state,
                &speed_ref_rpm,
                &speed_rpm,
                &id_meas,
                &iq_meas,
                &vd,
                &vq
            )

        diag = np.empty(6, dtype=np.float32)
        # FIX: index 0 = speed_ref_rpm, index 1 = speed_rpm
        # Old code had these swapped: diag[0]=speed_rpm, diag[1]=speed_ref_rpm
        diag[0] = speed_ref_rpm   # [RPM]
        diag[1] = speed_rpm       # [RPM]
        diag[2] = id_meas         # [A]
        diag[3] = iq_meas         # [A]
        diag[4] = vd              # [V]
        diag[5] = vq              # [V]
        return diag

    # -------------------------------------------------------------------------
    cpdef void reset(self) except *:
        """
        Reset all controller state.

        Calls MPC_Controller_Reset() which zeroes all dynamic state while
        PRESERVING smo.alpha_lpf (pre-computed from dt at Init time).
        FIX in MPC_Controller_Reset(): old code called Init(s, 0.0) which
        set alpha_lpf = 0 → SMO permanently blind after reset.
        """
        with nogil:
            MPC_Controller_Reset(&self._state)

        self.v_alpha       = 0.0
        self.v_beta        = 0.0
        self.speed_est     = 0.0
        self.speed_ref_est = 0.0
        self.id_est        = 0.0
        self.iq_est        = 0.0
        self.status        = 0

    # -------------------------------------------------------------------------
    def __repr__(self):
        return (
            f"MPCControllerWrapper("
            f"v_alpha={self.v_alpha:.4f}, v_beta={self.v_beta:.4f}, "
            f"speed_est={self.speed_est:.1f} RPM, "
            f"id={self.id_est:.3f}A, iq={self.iq_est:.3f}A)"
        )


# -----------------------------------------------------------------------------
# Convenience function  —  single-step call without persistent state
# -----------------------------------------------------------------------------
def mpc_step(float omega_ref_mech,
             float theta_m,
             float ia,
             float ib,
             float ic,
             float dt,
             float v_dc        = 17.0,
             int   p_poles     = 4,
             float R_s         = 0.285,
             float L           = 0.0003675,
             float lambda_pm   = 0.0014,
             float i_max       = 3.57,
             int   N           = 10,
             float Q_id        = 10.82,
             float Q_iq        = 0.01,
             float Q_omega     = 500.0,
             float R_vd        = 0.001,
             float R_vq        = 0.005,
             float smo_k       = 4.68,
             float smo_fc      = 1000.0,
             float KI_v        = 0.01,
             float SOFTSTART_T = 0.1) -> tuple:
    """
    Stateless single-step MPC calculation.

    Creates a fresh controller state, runs one step, returns outputs.
    For repeated calls prefer MPCControllerWrapper to preserve state.

    Returns
    -------
    tuple : (v_alpha [-], v_beta [-], speed_est [RPM], id_est [A], iq_est [A])
    """
    # Motor/structural parameters are compile-time — suppress unused warnings
    _ = (v_dc, p_poles, R_s, L, lambda_pm, i_max, N, Q_id, Q_iq,
         Q_omega, R_vd, R_vq, smo_k, smo_fc, KI_v, SOFTSTART_T)

    cdef MPCControllerWrapper ctrl = MPCControllerWrapper(dt_s=dt)
    ctrl.set_inputs_individual(omega_ref_mech, theta_m, ia, ib, ic)
    ctrl.compute(dt)

    # Returns speed_est in [RPM] — consistent with get_diagnostics() [0..1]
    return (ctrl.v_alpha, ctrl.v_beta,
            ctrl.speed_est, ctrl.id_est, ctrl.iq_est)


# -----------------------------------------------------------------------------
# Version info
# -----------------------------------------------------------------------------
__version__ = "1.2.0"
__author__  = "EmbedSim Team"
#
# v1.2.0 — Removed spurious duplicate of the entire module that was appended
#           after __version__ (lines 584-683 in the original file).  The
#           duplicate caused a Cython redefinition error for MPC_Input_T,
#           MPC_Output_T, MPC_SMO_T, MPC_EncSpeed_T and MPC_Controller_T
#           on every build.  No logic changes.

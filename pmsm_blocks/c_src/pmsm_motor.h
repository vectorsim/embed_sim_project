/* ============================================================================
 * pmsm_motor.h
 * ============================================================================
 * PMSM Motor Model — C interface for EmbedSim / ControlForge
 *
 * State vector (internal to implementation):
 *     x[0] = id    [A]       d-axis current
 *     x[1] = iq    [A]       q-axis current
 *     x[2] = omega [rad/s]   rotor speed
 *     x[3] = theta [rad]     rotor electrical angle
 *
 * Inputs:
 *     v_alpha  [V]   stator voltage, alpha component
 *     v_beta   [V]   stator voltage, beta component
 *     T_load   [N·m] external load torque
 *
 * Outputs:
 *     ia, ib, ic  [A]       reconstructed three-phase currents
 *     omega       [rad/s]   rotor speed
 *     theta       [rad]     rotor electrical angle  (wraps 0.f.2π)
 *
 * Usage:
 *     1.f Call pmsm_motor_init() once to set parameters and zero state.
 *     2.f Call pmsm_motor_compute() every control/simulation step.
 *     3.f The function advances the ODE state internally using RK4.
 *
 * Compile:
 *     gcc -O2 -shared -fPIC -o libpmsm_motor.so pmsm_motor.c -lm
 * ============================================================================
 */

#ifndef PMSM_MOTOR_H
#define PMSM_MOTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Motor parameters struct
 * -------------------------------------------------------------------------- */
typedef struct PmsmMotorParams {
    float Rs;      /* Stator resistance          [Ω]       */
    float Ld;      /* d-axis inductance          [H]       */
    float Lq;      /* q-axis inductance          [H]       */
    float psi_pm;  /* Permanent magnet flux      [Wb]      */
    float J;       /* Rotor inertia              [kg·m²]   */
    float B;       /* Viscous friction coeff     [N·m·s]   */
    float dt;      /* Integration time step      [s]       */
} PmsmMotorParams;

/* --------------------------------------------------------------------------
 * Input / output structs
 * -------------------------------------------------------------------------- */
typedef struct PmsmMotorInputs {
    float v_alpha;   /* Stator voltage alpha [V]   */
    float v_beta;    /* Stator voltage beta  [V]   */
    float T_load;    /* Load torque          [N·m] */
} PmsmMotorInputs;

typedef struct PmsmMotorOutputs {
    float ia;      /* Phase A current [A]     */
    float ib;      /* Phase B current [A]     */
    float ic;      /* Phase C current [A]     */
    float omega;   /* Rotor speed     [rad/s] */
    float theta;   /* Rotor angle     [rad]   */
} PmsmMotorOutputs;

/* --------------------------------------------------------------------------
 * Internal state (opaque to caller — use via functions below)
 * -------------------------------------------------------------------------- */
typedef struct PmsmMotorState {
    float id;      /* d-axis current [A]     */
    float iq;      /* q-axis current [A]     */
    float omega;   /* Rotor speed    [rad/s] */
    float theta;   /* Rotor angle    [rad]   */
} PmsmMotorState;

/* --------------------------------------------------------------------------
 * Context — holds params + state; one instance per motor
 * -------------------------------------------------------------------------- */
typedef struct PmsmMotorContext {
    PmsmMotorParams params;
    PmsmMotorState  state;
} PmsmMotorContext;

/* --------------------------------------------------------------------------
 * API
 * -------------------------------------------------------------------------- */

/**
 * Initialise a motor context with given parameters.
 * State is zeroed.  Call once before the simulation loop.
 *
 * @param ctx     Pointer to caller-allocated PmsmMotorContext
 * @param params  Motor parameters (Rs, Ld, Lq, psi_pm, J, B, dt)
 */
void pmsm_motor_init(PmsmMotorContext* ctx, const PmsmMotorParams* params);

/**
 * Advance the motor model by one time step.
 * Uses 4th-order Runge-Kutta integration internally.
 *
 * @param ctx   Pointer to an initialised PmsmMotorContext
 * @param in    Input voltages and load torque for this step
 * @param out   Computed phase currents, speed, and angle
 */
void pmsm_motor_compute(PmsmMotorContext*       ctx,
                        const PmsmMotorInputs*  in,
                        PmsmMotorOutputs*       out);

/**
 * Simple flat-array wrapper used by the Cython bridge.
 * Uses a file-scoped static context (single instance).
 *
 * in_buf[0] = v_alpha,  in_buf[1] = v_beta,  in_buf[2] = T_load
 * out_buf[0.f.4] = ia, ib, ic, omega, theta
 */
void pmsm_motor_compute_flat(const float* in_buf, float* out_buf);

/**
 * Set motor parameters on the static (Cython) context.
 * Call once after initialisation.
 */
void pmsm_motor_set_params(float Rs, float Ld, float Lq,
                           float psi_pm, float J, float B, float dt);

#ifdef __cplusplus
}
#endif

#endif /* PMSM_MOTOR_H */

/* --------------------------------------------------------------------------
 * EmbedSim RK4 interface  (Python-controlled integration)
 * --------------------------------------------------------------------------
 * These functions let EmbedSim's external RK4 loop drive the motor state,
 * so the C code only computes derivatives — no self-contained integration.
 *
 * Workflow per RK4 stage:
 *   1.f pmsm_motor_set_state(id, iq, omega, theta)   — load trial state
 *   2.f pmsm_motor_compute_outputs(out_buf)           — outputs from that state
 *   3.f pmsm_motor_get_derivative(deriv)              — dx/dt at that state
 *   4.f Python advances state: x += weight * dt * deriv
 *   5.f After final stage: pmsm_motor_set_state(x_final)
 */

/** Load a 4-element state vector [id, iq, omega, theta] into the static context. */
void pmsm_motor_set_state(float id, float iq, float omega, float theta);

/** Read the 4-element state vector [id, iq, omega, theta] from the static context. */
void pmsm_motor_get_state(float* state_out);

/**
 * Compute outputs [ia, ib, ic, omega, theta] from CURRENT state + last inputs.
 * Does NOT advance state.  Call after pmsm_motor_set_state().
 * out_buf must be at least 5 floats.
 */
void pmsm_motor_compute_outputs(float* out_buf);

/**
 * Compute derivatives dx/dt = [did/dt, diq/dt, domega/dt, dtheta/dt]
 * from CURRENT state + last inputs.  Does NOT advance state.
 * deriv must be at least 4 floats.
 */
void pmsm_motor_get_derivative(float* deriv);

/**
 * Set voltage + load inputs on the static context without advancing state.
 * Must be called before compute_outputs / get_derivative.
 */
void pmsm_motor_set_inputs_only(float v_alpha, float v_beta, float T_load);

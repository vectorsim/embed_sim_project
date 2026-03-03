/* ============================================================================
 * pi_controller.h
 * ============================================================================
 * Scalar PI controller with back-calculation anti-windup
 *
 * Control law:
 *   output = Kp · e + Ki · I
 *   I(t+dt) = I(t) + e · dt          (when NOT saturated or winding up)
 *   I(t+dt) = I(t)                   (frozen when saturated AND winding up)
 *
 * One PiControllerContext per controller instance.
 *
 * Compile (together with pi_controller.c):
 *   gcc -O2 -shared -fPIC -o libpi_controller.so pi_controller.c
 * ============================================================================
 */

#ifndef PI_CONTROLLER_H
#define PI_CONTROLLER_H

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Parameters struct
 * -------------------------------------------------------------------------- */
typedef struct PiControllerParams {
    float Kp;     /* Proportional gain          */
    float Ki;     /* Integral gain              */
    float limit;  /* Symmetric output clamp  ±limit */
    float dt;     /* Integration time step [s]  */
} PiControllerParams;

/* --------------------------------------------------------------------------
 * State
 * -------------------------------------------------------------------------- */
typedef struct PiControllerState {
    float integrator;  /* Accumulated integral of error */
} PiControllerState;

/* --------------------------------------------------------------------------
 * Context = params + state  (one per controller instance)
 * -------------------------------------------------------------------------- */
typedef struct PiControllerContext {
    PiControllerParams params;
    PiControllerState  state;
} PiControllerContext;

/* --------------------------------------------------------------------------
 * Input / output structs
 * -------------------------------------------------------------------------- */
typedef struct PiControllerInputs  { float error;  } PiControllerInputs;
typedef struct PiControllerOutputs { float output; } PiControllerOutputs;

/* --------------------------------------------------------------------------
 * API
 * -------------------------------------------------------------------------- */

/** Initialise context with given parameters; integrator zeroed. */
void pi_controller_init(PiControllerContext*       ctx,
                         const PiControllerParams*  params);

/** Compute one step and update integrator (Euler). */
void pi_controller_compute(PiControllerContext*        ctx,
                            const PiControllerInputs*   in,
                            PiControllerOutputs*        out);

/* Flat-array wrapper for Cython bridge.
 * Uses a file-scoped static context.
 * in_buf[0]  = error
 * out_buf[0] = control output
 */
void pi_controller_compute_flat(const float* in_buf, float* out_buf);

/** Set parameters on the static context. */
void pi_controller_set_params(float Kp, float Ki, float limit, float dt);

#ifdef __cplusplus
}
#endif

#endif /* PI_CONTROLLER_H */

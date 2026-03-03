/* ============================================================================
 * pmsm_motor.c
 * ============================================================================
 * PMSM Motor Model — C implementation
 *
 * Electrical equations (dq frame, rotor-field-oriented):
 *
 *   did/dt  = (vd - Rs·id + Lq·ω·iq) / Ld
 *   diq/dt  = (vq - Rs·iq - Ld·ω·id - ψ_pm·ω) / Lq
 *
 * Mechanical equations:
 *
 *   Te      = 1.5f · ψ_pm · iq
 *   dω/dt   = (Te - B·ω - T_load) / J
 *   dθ/dt   = ω
 *
 * Three-phase current reconstruction (inverse Park):
 *
 *   ia = id·cosf(θ)       − iq·sinf(θ)
 *   ib = id·cosf(θ−2π/3) − iq·sinf(θ−2π/3)
 *   ic = id·cosf(θ+2π/3) − iq·sinf(θ+2π/3)
 *
 * Integration: classical 4th-order Runge-Kutta (RK4).
 *
 * Compile (Linux/macOS):
 *   gcc -O2 -shared -fPIC -o libpmsm_motor.so pmsm_motor.c -lm
 *
 * Compile (Windows MSVC):
 *   cl /O2 /LD pmsm_motor.c /Fe:pmsm_motor.dll
 * ============================================================================
 */

#include "pmsm_motor.h"
#include <math.h>
#include <string.h>   /* memset */

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */
#ifndef M_PI
#define M_PI  3.14159265f
#endif

#define TWO_PI      (2.0f * M_PI)
#define PI_2_3      (2.0f * M_PI / 3.0f)   /* 120 degrees in radians */
#define THREE_HALFS (1.5f)

/* --------------------------------------------------------------------------
 * Internal helper: ODE right-hand side  dx/dt = f(state, inputs, params)
 *
 * state_in[4]  = [id, iq, omega, theta]  (current state for this RK stage)
 * vd, vq       = dq voltages (computed from v_alpha/v_beta + current theta)
 * T_load       = external load torque
 * deriv[4]     = output: [did/dt, diq/dt, domega/dt, dtheta/dt]
 * -------------------------------------------------------------------------- */
static void _pmsm_deriv(const float state_in[4],
                         float vd, float vq, float T_load,
                         const PmsmMotorParams* p,
                         float deriv[4])
{
    const float id    = state_in[0];
    const float iq    = state_in[1];
    const float omega = state_in[2];

    /* Electrical dynamics */
    deriv[0] = (vd - p->Rs * id    + p->Lq * omega * iq) / p->Ld;
    deriv[1] = (vq - p->Rs * iq    - p->Ld * omega * id
                   - p->psi_pm * omega) / p->Lq;

    /* Electromagnetic torque */
    const float Te = THREE_HALFS * p->psi_pm * iq;

    /* Mechanical dynamics */
    deriv[2] = (Te - p->B * omega - T_load) / p->J;
    deriv[3] = omega;
}

/* --------------------------------------------------------------------------
 * Initialise context
 * -------------------------------------------------------------------------- */
void pmsm_motor_init(PmsmMotorContext* ctx, const PmsmMotorParams* params)
{
    memcpy(&ctx->params, params, sizeof(PmsmMotorParams));
    memset(&ctx->state,  0,      sizeof(PmsmMotorState));
}

/* --------------------------------------------------------------------------
 * Compute one time step — full context version
 * -------------------------------------------------------------------------- */
void pmsm_motor_compute(PmsmMotorContext*       ctx,
                        const PmsmMotorInputs*  in,
                        PmsmMotorOutputs*       out)
{
    const PmsmMotorParams* p  = &ctx->params;
    const float           dt = p->dt;

    /* Current state as flat array for RK4 */
    float x[4] = {
        ctx->state.id,
        ctx->state.iq,
        ctx->state.omega,
        ctx->state.theta
    };

    /* ── Park transform: αβ → dq using CURRENT theta ──────────────────────
     * We use the angle at the START of the step for all four RK4 stages.
     * A more accurate approach would update theta per stage, but for the
     * typical dt used in FOC (≤ 0.1f ms) this is indistinguishable.
     * ─────────────────────────────────────────────────────────────────────*/
    const float theta  = x[3];
    const float cos_th = cosf(theta);
    const float sin_th = sinf(theta);
    const float vd     =  in->v_alpha * cos_th + in->v_beta * sin_th;
    const float vq     = -in->v_alpha * sin_th + in->v_beta * cos_th;

    /* ── RK4 integration ───────────────────────────────────────────────── */
    float k1[4], k2[4], k3[4], k4[4];
    float x_tmp[4];
    int i;

    /* k1 = f(x) */
    _pmsm_deriv(x, vd, vq, in->T_load, p, k1);

    /* k2 = f(x + dt/2 · k1) */
    for (i = 0; i < 4; i++) x_tmp[i] = x[i] + 0.5f * dt * k1[i];
    _pmsm_deriv(x_tmp, vd, vq, in->T_load, p, k2);

    /* k3 = f(x + dt/2 · k2) */
    for (i = 0; i < 4; i++) x_tmp[i] = x[i] + 0.5f * dt * k2[i];
    _pmsm_deriv(x_tmp, vd, vq, in->T_load, p, k3);

    /* k4 = f(x + dt · k3) */
    for (i = 0; i < 4; i++) x_tmp[i] = x[i] + dt * k3[i];
    _pmsm_deriv(x_tmp, vd, vq, in->T_load, p, k4);

    /* Combine: x_new = x + dt/6 · (k1 + 2k2 + 2k3 + k4) */
    float x_new[4];
    for (i = 0; i < 4; i++) {
        x_new[i] = x[i] + (dt / 6.0f) *
                   (k1[i] + 2.0f*k2[i] + 2.0f*k3[i] + k4[i]);
    }

    /* Wrap theta to [0, 2π) */
    x_new[3] = fmodf(x_new[3], TWO_PI);
    if (x_new[3] < 0.0f) x_new[3] += TWO_PI;

    /* Store updated state */
    ctx->state.id    = x_new[0];
    ctx->state.iq    = x_new[1];
    ctx->state.omega = x_new[2];
    ctx->state.theta = x_new[3];

    /* ── Three-phase current reconstruction (inverse Park) ─────────────── */
    const float id_new    = x_new[0];
    const float iq_new    = x_new[1];
    const float theta_new = x_new[3];

    const float cos_t     = cosf(theta_new);
    const float sin_t     = sinf(theta_new);
    const float cos_m120  = cosf(theta_new - PI_2_3);
    const float sin_m120  = sinf(theta_new - PI_2_3);
    const float cos_p120  = cosf(theta_new + PI_2_3);
    const float sin_p120  = sinf(theta_new + PI_2_3);

    out->ia    = id_new * cos_t    - iq_new * sin_t;
    out->ib    = id_new * cos_m120 - iq_new * sin_m120;
    out->ic    = id_new * cos_p120 - iq_new * sin_p120;
    out->omega = x_new[2];
    out->theta = x_new[3];
}

/* --------------------------------------------------------------------------
 * Static context for the flat-array / Cython interface
 * (one motor instance — sufficient for most simulations)
 * -------------------------------------------------------------------------- */
static PmsmMotorContext _static_ctx;
static int              _static_ctx_ready = 0;

void pmsm_motor_set_params(float Rs, float Ld, float Lq,
                           float psi_pm, float J, float B, float dt)
{
    PmsmMotorParams p;
    p.Rs     = Rs;
    p.Ld     = Ld;
    p.Lq     = Lq;
    p.psi_pm = psi_pm;
    p.J      = J;
    p.B      = B;
    p.dt     = dt;
    pmsm_motor_init(&_static_ctx, &p);
    _static_ctx_ready = 1;
}

void pmsm_motor_compute_flat(const float* in_buf, float* out_buf)
{
    /* Lazy init with safe defaults if set_params was never called */
    if (!_static_ctx_ready) {
        PmsmMotorParams p = {0.5f, 0.002f, 0.002f, 0.3f, 0.001f, 0.001f, 0.0001f};
        pmsm_motor_init(&_static_ctx, &p);
        _static_ctx_ready = 1;
    }

    PmsmMotorInputs  in;
    PmsmMotorOutputs out_s;

    in.v_alpha = in_buf[0];
    in.v_beta  = in_buf[1];
    in.T_load  = in_buf[2];

    pmsm_motor_compute(&_static_ctx, &in, &out_s);

    out_buf[0] = out_s.ia;
    out_buf[1] = out_s.ib;
    out_buf[2] = out_s.ic;
    out_buf[3] = out_s.omega;
    out_buf[4] = out_s.theta;
}

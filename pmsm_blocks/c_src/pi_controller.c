/* ============================================================================
 * pi_controller.c
 * ============================================================================
 * Scalar PI controller with back-calculation anti-windup
 *
 * Anti-windup logic:
 *   The integrator is clamped (dI/dt = 0) when both:
 *     1.f The raw output (before clipping) is saturated: |raw| >= limit
 *     2.f The error would drive the integrator further into saturation:
 *        e · I > 0   (error and integrator have same sign)
 *
 * Euler integration:   I(n+1) = I(n) + e(n) · dt
 *
 * Compile:
 *   gcc -O2 -shared -fPIC -o libpi_controller.so pi_controller.c
 * ============================================================================
 */

#include "pi_controller.h"
#include <string.h>  /* memset, memcpy */

/* ── Utility: clamp to ±limit ─────────────────────────────────────────────── */
static float _clamp(float v, float limit)
{
    if (v >  limit) return  limit;
    if (v < -limit) return -limit;
    return v;
}

/* --------------------------------------------------------------------------
 * Initialise
 * -------------------------------------------------------------------------- */
void pi_controller_init(PiControllerContext*      ctx,
                         const PiControllerParams* params)
{
    memcpy(&ctx->params, params, sizeof(PiControllerParams));
    memset(&ctx->state,  0,      sizeof(PiControllerState));
}

/* --------------------------------------------------------------------------
 * Compute one step
 * -------------------------------------------------------------------------- */
void pi_controller_compute(PiControllerContext*       ctx,
                            const PiControllerInputs*  in,
                            PiControllerOutputs*       out)
{
    const PiControllerParams* p = &ctx->params;
    const float error = in->error;
    const float I     = ctx->state.integrator;

    /* Raw PI output (pre-saturation) */
    const float raw    = p->Kp * error + p->Ki * I;

    /* Saturated output */
    out->output = _clamp(raw, p->limit);

    /* Anti-windup: freeze integrator when saturated and winding up */
    const int saturated   = (raw >  p->limit) || (raw < -p->limit);
    const int winding_up  = (error * I) > 0.0f;

    if (!(saturated && winding_up)) {
        ctx->state.integrator = I + error * p->dt;
    }
    /* else: integrator stays frozen */
}

/* --------------------------------------------------------------------------
 * Static (Cython) context
 * -------------------------------------------------------------------------- */
static PiControllerContext _static_ctx;
static int                 _static_ctx_ready = 0;

void pi_controller_set_params(float Kp, float Ki, float limit, float dt)
{
    PiControllerParams p = { Kp, Ki, limit, dt };
    pi_controller_init(&_static_ctx, &p);
    _static_ctx_ready = 1;
}

void pi_controller_compute_flat(const float* in_buf, float* out_buf)
{
    if (!_static_ctx_ready) {
        /* Safe defaults — should not happen if Python wrapper calls
         * set_params() on construction */
        PiControllerParams p = { 1.0f, 10.0f, 100.0f, 0.0001f };
        pi_controller_init(&_static_ctx, &p);
        _static_ctx_ready = 1;
    }

    PiControllerInputs  in  = { in_buf[0] };
    PiControllerOutputs out_s;
    pi_controller_compute(&_static_ctx, &in, &out_s);
    out_buf[0] = out_s.output;
}

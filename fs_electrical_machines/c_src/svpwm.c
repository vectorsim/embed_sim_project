/* svpwm.c
 * =============================================================================
 * Space Vector PWM switching time calculator.
 * EmbedSim / foc_generator / c_src
 *
 * Implements sector detection and T1/T2/T0 dwell times from:
 *
 *   T1 = Ts * (sqrt(3)*Vref/Vdc) * sin(n*pi/3 - alpha)
 *   T2 = Ts * (sqrt(3)*Vref/Vdc) * sin(alpha - (n-1)*pi/3)
 *   T0 = Ts - T1 - T2
 *
 * MISRA C:2012 compliant.
 * =============================================================================
 */

#include "svpwm.h"
#include <math.h>      /* sinf */

/* ── Constants ────────────────────────────────────────────────────────────── */
#define SVPWM_SQRT3          (1.7320508076f)
#define SVPWM_SQRT3_OVER_2   (0.8660254038f)
#define SVPWM_PI_OVER_3      (1.0471975512f)   /* 60 deg in radians          */
#define SVPWM_TWO_PI         (6.2831853072f)
#define SVPWM_VDC_MIN        (1.0e-6f)         /* Guard against divide-by-0  */


/* ── SVPWM_Init ───────────────────────────────────────────────────────────── */
void SVPWM_Init(void)
{
    /* Stateless block — nothing to initialise */
}


/* ── SVPWM_Step ───────────────────────────────────────────────────────────── */
void SVPWM_Step(const SVPWM_Input  *u,
                      SVPWM_Output *y)
{
    real32_T alpha_norm;
    real32_T modulation;
    uint8_T  sector;
    real32_T alpha_local;
    real32_T t1;
    real32_T t2;
    real32_T t0;

    /* ── Guard: degenerate Vdc ────────────────────────────────────────────── */
    if (u->Vdc < SVPWM_VDC_MIN)
    {
        y->T1     = 0.0f;
        y->T2     = 0.0f;
        y->T0     = u->Ts;
        y->sector = 1U;
        return;
    }

    /* ── Normalise alpha to [0, 2*pi) ────────────────────────────────────── */
    alpha_norm = u->alpha;
    while (alpha_norm < 0.0f)          { alpha_norm += SVPWM_TWO_PI; }
    while (alpha_norm >= SVPWM_TWO_PI) { alpha_norm -= SVPWM_TWO_PI; }

    /* ── Modulation index: sqrt(3) * Vref / Vdc ──────────────────────────── */
    modulation = (SVPWM_SQRT3 * u->Vref) / u->Vdc;

    /* ── Sector detection: 1..6 ──────────────────────────────────────────── */
    sector = (uint8_T)(alpha_norm / SVPWM_PI_OVER_3) + 1U;
    if (sector > 6U) { sector = 6U; }   /* MISRA 14.3: explicit clamp       */

    /* ── alpha relative to sector start ─────────────────────────────────── */
    alpha_local = alpha_norm - ((real32_T)(sector - 1U) * SVPWM_PI_OVER_3);

    /* ── Dwell times ─────────────────────────────────────────────────────── */
    /*   T1 = Ts * m * sin(pi/3 - alpha_local)                               */
    /*   T2 = Ts * m * sin(alpha_local)                                       */
    t1 = u->Ts * modulation * sinf(SVPWM_PI_OVER_3 - alpha_local);
    t2 = u->Ts * modulation * sinf(alpha_local);
    t0 = u->Ts - t1 - t2;

    /* ── Clamp T0 to zero (numerical noise at sector boundaries) ─────────── */
    if (t0 < 0.0f) { t0 = 0.0f; }

    y->T1     = t1;
    y->T2     = t2;
    y->T0     = t0;
    y->sector = sector;
}

/* svpwm.h
 * =============================================================================
 * Space Vector PWM switching time calculator — public API.
 * EmbedSim / foc_generator / c_src
 *
 * Include this header in any C module that calls SVPWM_Step().
 * =============================================================================
 */

#ifndef SVPWM_H
#define SVPWM_H

#include "Sys_Types.h"   /* real32_T, uint8_T */

#ifdef __cplusplus
extern "C" {
#endif


/* ── Input struct ─────────────────────────────────────────────────────────── */
typedef struct
{
    real32_T Vref;    /**< Reference voltage magnitude [V]  (0 .. Vdc/sqrt(3)) */
    real32_T alpha;   /**< Reference angle [rad]            (0 .. 2*pi)         */
    real32_T Vdc;     /**< DC bus voltage [V]                                    */
    real32_T Ts;      /**< Sample period [s]  (e.g. 1/10000 for 10 kHz)         */
} SVPWM_Input;


/* ── Output struct ────────────────────────────────────────────────────────── */
typedef struct
{
    real32_T T1;      /**< Active vector 1 on-time [s]  */
    real32_T T2;      /**< Active vector 2 on-time [s]  */
    real32_T T0;      /**< Zero vector on-time [s]       */
    uint8_T  sector;  /**< Active sector 1..6            */
} SVPWM_Output;


/* ── Public API ───────────────────────────────────────────────────────────── */

/**
 * SVPWM_Init
 * Call once at startup (stateless block — currently a no-op,
 * present for API consistency with stateful blocks).
 */
void SVPWM_Init(void);

/**
 * SVPWM_Step
 * Compute switching dwell times for one PWM period.
 *
 * @param u   Pointer to populated SVPWM_Input struct.
 * @param y   Pointer to SVPWM_Output struct to be filled.
 */
void SVPWM_Step(const SVPWM_Input *u,
                      SVPWM_Output *y);


#ifdef __cplusplus
}
#endif

#endif /* SVPWM_H */

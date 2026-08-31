/**********************************************************************************************************************
 * \file        cdd_evadc_app.h
 * \brief       EVADC driver interface for 3-phase FOC sensor readout on AURIX TC38x + AP32541 power board.
 *
 * \details     Five EVADC channels on four groups, all GTM-triggered.  Channel
 *              mapping per AP32541 v1.0 Table 12 / Table 15 (AppKit TC387):
 *
 *              Phase current sensing — triggered by ATOM0_CH7 via ADCTRIG0
 *              (duty 0.9, falling edge, XTMODE=1), one edge starts all three
 *              groups in parallel (simultaneous sampling).  Each group then
 *              converts 4 back-to-back samples accumulated in hardware
 *              (DRCTR=3, DMM=0) and scaled back in software — the same 4x
 *              oversampling scheme as the Infineon PmsmFoc reference:
 *                  G0_C0  AN00  VO1  Phase U current   → CddApp_T.Vu / .Iu
 *                  G3_C0  AN24  VO2  Phase V current   → CddApp_T.Vv / .Iv
 *                  G2_C0  AN16  VO3  Phase W current   → CddApp_T.Vw / .Iw
 *
 *              Reference + DC-link — triggered by the SAME ATOM0_CH7 edge via
 *              ADC_TRIG0[1] (GxQCTRL0.XTSEL = 0x8 = REQTRI, uniform per TC38x
 *              UM Appendix Table 292); one edge converts both G1 queue
 *              entries back-to-back at the PWM rate:
 *                  G1_C0  AN08  VRO      CSA zero-current reference  → module-internal
 *                  G1_C3  AN11  VOLT_DC  DC-link voltage             → CddApp_T.Vdc
 *
 *              Interrupts: ONE service request per conversion set, raised by
 *              G1RES3 (the last result).  The phase groups generate no SRs.
 *
 *              Conversion chain (CddEvadc_ConvertPhaseCurrents):
 *
 *                  Ix [A] = SIGN * (Vx - VRO_measured - Offset_x) / (G_CSA * R_shunt)
 *
 *                  R_shunt = 10 mOhm 1%      (AP32541 BoM #34, R17/R27/R37)
 *                  G_CSA   = 30.81 V/V typ   (TLE9180D gain code 100B, programmed
 *                                             via OP_GAIN1/2/3 in cdd_tle9180_app.c)
 *                  → 3.246 A/V, full scale ±8.1 A around VRO = 2.5 V, VAREF = 5 V.
 *
 *              VRO_measured is the live AN8 reading (tracks device spread and
 *              temperature drift of the reference buffer and cancels the ADC
 *              reference error at zero current).  Offset_x are the per-CSA
 *              residual output offsets (±100 mV uncal., ±10 mV after TLE9180
 *              auto-calibration, DS P_9.6.17/18), stored in CddApp_T.Vuo/
 *              .Vvo/.Vwo, determined ONCE at standstill by
 *              CddEvadc_CalibratePhaseOffsets() (never cyclically).
 *
 *              Vdc is the DC-link BUS voltage in volts, i.e. the pin voltage
 *              rescaled by the on-board 5.6k/56k divider (x11.0, AP32541 Eq. 1).
 *
 * \note        Commissioning order:
 *                  1. CddEvadc_Init()                    (after cdd_gpio_app, before GTM triggers)
 *                  2. Enable GTM ATOM triggers
 *                  3. TLE9180 in NORMAL mode, bridge DISABLED (/SOFF asserted)
 *                  4. CddEvadc_CalibratePhaseOffsets(&App, 64U)
 *                  5. Release /SOFF, start modulation
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_evadc_app.c
 *              - Rule  8.10: static used in declaration AND definition of internals
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_EVADC_APP_H_
#define CDD_EVADC_APP_H_

#include "cdd_config.h"           /* embed_sim_sys_types.h + embed_sim_compiler.h        */
#include "cdd_app.h"
#include "embed_sim_foc_types.h"  /* FocUvw_T (also pulls embed_sim_matrix.h:
                                   * MatrixFloat, MatrixStatus_Type)                     */

/**********************************************************************************************************************
 * Return Codes
 *********************************************************************************************************************/

/** \brief  CddEvadc_CalibratePhaseOffsets(): offsets stored, plausibility checks passed. */
#define CDD_EVADC_CAL_OK        (1U)
/** \brief  CddEvadc_CalibratePhaseOffsets(): stored offsets UNCHANGED — NumSamples==0,
 *          trigger/ISR timeout, VRO average outside 2.5 V +- 0.18 V, or a residual
 *          offset outside +-0.12 V (bridge not disabled / motor moving / CSA fault). */
#define CDD_EVADC_CAL_REJECTED  (0U)

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialises CONVCTRL, EVADC global config, calibration, and all
 *          measurement channels (G0/G3/G2 phase currents, G1 VRO + DC-link).
 *
 * \details Call after cdd_gpio_app and before GTM triggers are enabled.
 *          Blocks until hardware calibration completes for all groups.
 *
 * \return  void
 */
extern void CddEvadc_Init(void);

/**
 * \brief   Converts the sensed CSA output voltages into phase currents and
 *          updates the Isum plausibility signal.
 *
 * \details Ix = SIGN * (Vx - VRO_measured - Offset_x) * 3.246 A/V.  Call after
 *          CddEvadc_ReadSensorMeas() in the 20 kHz control task.  Isum should
 *          sit at noise level; a constant bias indicates offset drift (rerun
 *          calibration), a modulation-dependent Isum indicates the sampling
 *          instant leaving the low-side ON window.
 *
 * \param[in,out] CddAppPtr  Application data; Iu/Iv/Iw/Isum updated.  [A]
 * \return  void
 */
extern void CddEvadc_ConvertPhaseCurrents(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr);

extern void CddEvadc_CalibrateCurrentOffset(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr);

#endif /* CDD_EVADC_APP_H_ */

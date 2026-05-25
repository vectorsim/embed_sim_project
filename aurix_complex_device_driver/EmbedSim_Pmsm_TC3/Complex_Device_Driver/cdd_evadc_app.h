/**********************************************************************************************************************
 * \file        cdd_evadc_app.h
 * \brief       EVADC driver interface for 3-phase FOC sensor readout on AURIX TC3xx.
 *
 * \details     Four EVADC channels configured, all GTM-triggered:
 *
 *              Phase current sensing (triggered by ATOM0_CH4 via ADCTRIG0):
 *                  G0_C0  AN00  Phase U current   → CddEvadc_Meas_T.IPhU
 *                  G1_C0  AN08  Phase V current   → CddEvadc_Meas_T.IPhV
 *                  G2_C0  AN16  Phase W current   → CddEvadc_Meas_T.IPhW
 *
 *              DC-link voltage (triggered by ATOM0_CH3 via ADCTRIG3):
 *                  G8_C8  AN40  DC-link voltage   → CddEvadc_Meas_T.UDcLink
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_evadc_app.c
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_EVADC_APP_H_
#define CDD_EVADC_APP_H_

#include "cdd_config.h"   /* embed_sim_sys_types.h + embed_sim_compiler.h */

/**********************************************************************************************************************
 * Data Types
 *********************************************************************************************************************/

/**
 * \brief   Sensor measurement result structure.
 *
 * \details All values are in physical units after ADC full-scale scaling:
 *              IPhU/V/W  = phase current voltage  [V]  (0..ADC_MAX_VOLTAGE)
 *              UDcLink   = DC-link voltage          [V]
 */
typedef struct
{
    volatile real32_T   IPhU;       /**< \brief Phase U current sense voltage  [V] */
    volatile real32_T   IPhV;       /**< \brief Phase V current sense voltage  [V] */
    volatile real32_T   IPhW;       /**< \brief Phase W current sense voltage  [V] */
    volatile real32_T   UDcLink;    /**< \brief DC-link voltage                [V] */
} CddEvadc_Meas_T;

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialises CONVCTRL, EVADC global config, calibration, and all
 *          measurement channels.
 *
 * \details Call after cdd_gpio_app and before GTM triggers are enabled.
 *          Blocks until hardware calibration completes for all groups.
 *
 * \return  void
 */
extern void CddEvadc_Init(void);

/**
 * \brief   Reads all four ADC result registers into the measurement structure.
 *
 * \details Each channel read only if the Valid Flag (VF) is set.
 *          If VF is not set the previous value is retained.
 *          Call from the control loop ISR (GTM_Atom_00_Ch_00_Isr).
 *
 * \param[in,out] MeasPtr  Pointer to the measurement result structure.
 * \return  void
 */
extern void CddEvadc_ReadSensorMeas(
    P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr);

#endif /* CDD_EVADC_APP_H_ */

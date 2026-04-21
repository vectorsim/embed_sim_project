/**********************************************************************************************************************
 * \file        cdd_evadc_app.h
 * \brief       EVADC driver interface for 3-phase FOC sensor readout on AURIX TC3xx.
 *
 * \details     Four EVADC channels are configured, all GTM-triggered:
 *
 *              Phase current sensing (triggered by ATOM0_CH4 via ADCTRIG0):
 *                  G0_C0  AN00  Phase U current   -> Meas.IPhU
 *                  G1_C0  AN08  Phase V current   -> Meas.IPhV
 *                  G2_C0  AN16  Phase W current   -> Meas.IPhW
 *
 *              DC-link voltage (triggered by ATOM0_CH3 via ADCTRIG3):
 *                  G8_C8  AN40  DC-link voltage   -> Meas.UDcLink
 *
 *              Resolver channels (G3 AN24, G11 AN19) are not present on the
 *              DB42S02 motorkit.  Position/speed acquisition uses the GPT12
 *              incremental encoder on P02.6/7/8 via cdd_gpt12_app.
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

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"

/**********************************************************************************************************************
 * Data Types
 *********************************************************************************************************************/

/**
 * \brief   Sensor measurement result structure.
 *
 * \details All values are in physical units after ADC full-scale scaling:
 *              IPhU/V/W  = phase current voltage  [V]   (0..ADC_MAX_VOLTAGE)
 *              SinP/CosP = resolver signal voltage [V]
 *              UDcLink   = DC-link voltage          [V]  (scaled by resistor divider)
 */
typedef struct
{
    volatile real32_T   IPhU;       /**< \brief Phase U current sense voltage  [V] */
    volatile real32_T   IPhV;       /**< \brief Phase V current sense voltage  [V] */
    volatile real32_T   IPhW;       /**< \brief Phase W current sense voltage  [V] */
    volatile real32_T   UDcLink;    /**< \brief DC-link voltage                [V] */
} EVADC_Meas_T;

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialises CONVCTRL, EVADC global config, calibration, and all
 *          six measurement channels.
 *
 * \details Call after cdd_gpio_app and before GTM triggers are enabled.
 *          Blocks until hardware calibration completes for all groups.
 *
 * \return  None
 */
extern void Initialize_EVADC_Module(void);

/**
 * \brief   Reads all six ADC result registers into the measurement structure.
 *
 * \details Each channel is read only if the Valid Flag (VF) is set.
 *          If VF is not set the previous value is retained.
 *          Call from the control loop ISR (GTM_ATOM_00_CH_00_ISR).
 *
 * \param   Meas_Ptr   Pointer to the measurement result structure
 * \return  None
 */
extern void Read_EVADC_Sensor_Measurement(volatile EVADC_Meas_T * const Meas_Ptr);

#endif /* CDD_EVADC_APP_H_ */

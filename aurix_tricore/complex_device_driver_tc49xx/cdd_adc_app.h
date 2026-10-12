/**********************************************************************************************************************
 * \file      cdd_adc_app.h
 * \brief     Public interface for the TMADC && DSADC application module .
 *
 * \details
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter → Uppercase)
 *              - Output pointers: PascalCasePtr
 *              - Local variables: lowerPascalCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   1.0.0
 * \date      2026-10-04
 * \author    Paul Abraham
 *            EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef COMPLEX_DEVICE_DRIVER_CDD_ADC_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_ADC_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"
#include "cdd_adc_app.h"
#include "cdd_app.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief Initializes the TMADC application module.
 * \details Configures three phase current channels and Vdc.
 *          Must be called before \ref CddAdc_ReadSensor.
 */
extern void CddAdc_Init(void);

/**
 * \brief Reads all ADC channels and updates the inverter control structure.
 * \param[in,out] InvPtr Pointer to the inverter control data.
 * \details Reads Phase U, Phase V, Phase W, and Vdc, and sets valid flags.
 */
extern void CddAdc_ReadSensor(P2VAR(CddInverterCtrl_T, AUTOMATIC, CDD_APPL_DATA) InvPtr);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_ADC_APP_H_ */

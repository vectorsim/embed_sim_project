/**********************************************************************************************************************
 * \file      cdd_app.c
 * \brief     Public interface for the CddSys low-level system utility module.
 *
 * \details   Declares the CddSys function API used across the CDD layer for
 *            TriCore-specific primitives:
 *              - Atomic bit-field load-modify-store (LDMST instruction).
 *              - Binary semaphore via compare-and-swap (CMPSWAP.W instruction).
 *              - Single and multi-instruction NOP delays.
 *              - Interrupt-enable intrinsic.
 *
 *
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
 * \author   Paul Abraham
 *             EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/


/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "cdd_app.h"
#include "cdd_sys_util.h"
#include "cdd_egtm_app.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Central runtime state of the inverter control application.
 *
 * \details Single global instance shared between the control-loop ISR
 *          (ControlLoop_ISR), the fast shut-off ISR (TIM0_CH4_ISR), and the
 *          application layer. Declared extern in cdd_app.h; defined here so
 *          that exactly one translation unit owns the storage.
 *
 * \note    See CddInverterCtrl_T in cdd_app.h for the per-member semantics,
 *          valid ranges, and the functions that read or write each field.
 *
 * \see     CddInverterCtrl_T
 */
 CddInverterCtrl_T  Inv_G;

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

 void Cdd_Init(void)
 {
     /* Init EGTM Module*/
     CddEgtm_InitModule();

 }


 void Cdd_Start(void)
 {
     CddEgtm_Start();

 }

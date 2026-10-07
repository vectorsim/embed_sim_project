/**********************************************************************************************************************
 * \file      cdd_egtm_app.h
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
 * \author    Paul Abraham
 *            EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef COMPLEX_DEVICE_DRIVER_CDD_EGTM_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_EGTM_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"

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
 * \brief   Initialises the eGTM module for three-phase inverter control.
 *
 * \details Full eGTM bring-up for the inverter:
 *            -# Enables the eGTM clock and CMU0 global clock, and programs the
 *               CMU to the CMU0 target frequency derived from the cluster
 *               clock.
 *            -# Initialises the Inv_G global with default dead time, duty
 *               cycles, shut-off state, frequency mode, and CMU0 frequency.
 *            -# Computes Inv_G.PeriodTicks for the selected frequency mode.
 *            -# Configures the TIM0 channel 4 fast shut-off path.
 *            -# Configures the ATOM CLS0 master channel and the three
 *               complementary PWM phases with dead time.
 *
 * \return  void
 *
 * \post    eGTM is configured but outputs remain inactive until the host
 *          trigger is issued by CddEgtm_Start().
 *
 * \see     CddEgtm_InitTim0Ch4
 * \see     CddEgtm_InitAtomCls0
 * \see     CddEgtm_Start
 */
extern void CddEgtm_InitModule(void);

/**
 * \brief   Starts all enabled eGTM ATOM channels synchronously.
 *
 * \details Sets HOST_TRIG in EGTM_CLS0_ATOM_AGC_GLB_CTRL, which latches the
 *          shadow registers and starts every enabled channel at the same
 *          eGTM clock edge. From this point the master channel drives the
 *          control-loop ISR and the PWM phases drive the gate outputs.
 *
 * \return  void
 *
 * \pre     CddEgtm_InitModule() has completed.
 *
 * \see     CddEgtm_InitModule
 * \see     CddEgtm_InitAtomCls0
 */
extern void CddEgtm_Start(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_EGTM_APP_H_ */

/**********************************************************************************************************************
 * \file      cdd_egtm_app.h
 * \brief     Public interface for the CddEgtm application module (eGTM inverter control).
 *
 * \details   Declares the public API used by the CDD layer to initialise and start
 *            the Infineon eGTM (enhanced Generic Timer Module) for three-phase
 *            inverter PWM generation on AURIX TriCore:
 *              - Clock enable and CMU0 frequency setup.
 *              - ATOM cluster 0 master + three complementary phases with dead-time.
 *              - TIM0 channel 4 fast shut-off input path.
 *              - Synchronous start of all enabled ATOM channels via HOST_TRIG.
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
 * \note      MISRA C:2012 compliance notes for this module:
 *              - Rule 8.5 / 8.6 : External declarations are provided only in this
 *                header; definitions reside in the corresponding .c file.
 *              - Rule 2.1       : No dead code in the public interface.
 *              - Directive 4.9  : Function-like macros are not exposed in this header.
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
 * \details Performs the complete eGTM bring-up sequence required before PWM
 *          generation can begin:
 *            -# Enables the eGTM module clock and the CMU0 global clock.
 *            -# Programs the CMU to the target frequency derived from the
 *               cluster clock and stores the resulting CMU0 frequency in
 *               Inv_G.CmuClk0Frequency.
 *            -# Initialises the Inv_G global structure with default dead-time,
 *               phase duty cycles, shut-off state flags, frequency mode and
 *               period ticks.
 *            -# Configures TIM0 channel 4 as the fast shut-off input (when
 *               CDD_TIM0_CH4_ON is enabled).
 *            -# Configures ATOM cluster 0: master channel CH0 plus the three
 *               complementary phase channels CH5/CH6/CH7 with dead-time
 *               insertion via CDTM DTM5.
 *
 * \return  void
 *
 * \post    eGTM hardware is fully configured.  Gate-drive outputs remain
 *          inactive until CddEgtm_Start() issues the HOST_TRIG.
 *
 * \note    This function must be called exactly once during system start-up,
 *          before any call to CddEgtm_Start().
 *
 * \see     CddEgtm_Start
 * \see     CddEgtm_Init_Tim0_Ch4_P00_01_FS
 * \see     CddEgtm_Init_AtomCls0_Inv
 *
 * \req     MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern void CddEgtm_Init(void);

/**
 * \brief   Starts all enabled eGTM ATOM channels synchronously.
 *
 * \details Writes HOST_TRIG = 1 to EGTM_CLS0_ATOM_AGC_GLB_CTRL.  This single
 *          write latches every enabled channel's shadow registers and starts
 *          the counters on the same eGTM clock edge, guaranteeing phase
 *          alignment between the master timing reference and the three PWM
 *          phases.
 *
 *          After the trigger:
 *            - Master channel CH0 generates CCU0/CCU1 events that drive the
 *              control-loop ISR.
 *            - Phase channels CH5/CH6/CH7 produce complementary high/low-side
 *              PWM with the programmed dead-time.
 *
 * \return  void
 *
 * \pre     CddEgtm_Init() has completed successfully.
 *
 * \note    The function is intentionally side-effect free apart from the
 *          single register write; it may be called only once after init.
 *
 * \see     CddEgtm_Init
 *
 * \req     MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern void CddEgtm_Start(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_EGTM_APP_H_ */

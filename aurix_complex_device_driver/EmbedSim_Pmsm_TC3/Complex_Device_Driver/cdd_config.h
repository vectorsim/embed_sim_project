/**********************************************************************************************************************
 * \file        cdd_config.h
 * \brief       System-wide configuration constants, macros, and interrupt vector
 *              table entry generation for AURIX TC3xx bare-metal targets.
 *
 * \details     Provides:
 *              - ES_INTERRUPT() trampoline macro (TASKING TriCore assembler syntax)
 *              - Mathematical and unit-conversion constants
 *              - Compile-time clock frequency constants
 *              - STM interrupt priority assignment
 *
 * \note        Assembler syntax:
 *              TASKING cctc+astc toolchain requires  HI:symbol / LO:symbol
 *              relocation syntax in inline __asm blocks.  The GCC TriCore
 *              syntax (@HI / @LO) is rejected by astc and must NOT be used.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  2.5  : All macros referenced by at least one translation unit
 *              - Rule 20.10 : Stringification (#) used only inside ES_INTERRUPT
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_CONFIG_H_
#define CDD_CONFIG_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "embed_sim_sys_types.h"


/**********************************************************************************************************************
 * Mathematical Constants
 *********************************************************************************************************************/
#define PI                      (3.141592653589793)     /**< \brief Pi                  [rad] */
#define RAD_360                 (6.283185307179586)     /**< \brief Full circle         [rad] */
#define RAD_270                 (4.712388980384690)     /**< \brief 270 degrees         [rad] */
#define RAD_240                 (4.188790204786391)     /**< \brief 240 degrees         [rad] */
#define RAD_120                 (2.094395102393195)     /**< \brief 120 degrees         [rad] */
#define RAD_90                  (1.570796326794897)     /**< \brief  90 degrees         [rad] */

/**********************************************************************************************************************
 * Numerical Utility
 *********************************************************************************************************************/

/** \brief  Boolean-like true expression — MISRA C:2012 Rule 14.4 compliant    */
#define MISRA_TRUE              (0x1U == 0x1U)

/** \brief  Floating-point zero-comparison epsilon                              */
#define EPSILON_ZERO            (1.0e-10f)

/**********************************************************************************************************************
 * Clock Frequency Constants  [Hz]  (ds1 P.1069)
 *********************************************************************************************************************/
#define MHZ_300                 (300000000.0f)   /**< \brief 300 MHz  [Hz] */
#define MHZ_200                 (200000000.0f)   /**< \brief 200 MHz  [Hz] */
#define MHZ_160                 (160000000.0f)   /**< \brief 160 MHz  [Hz] */
#define MHZ_100                 (100000000.0f)   /**< \brief 100 MHz  [Hz] */
#define MHZ_50                   (50000000.0f)   /**< \brief  50 MHz  [Hz] */
#define MHZ_20                   (20000000.0f)   /**< \brief  20 MHz  [Hz] */
#define MHZ_5                     (5000000.0f)   /**< \brief   5 MHz  [Hz] */
#define MHZ_1                     (1000000.0f)   /**< \brief   1 MHz  [Hz] */

/** \brief EVR (internal backup) oscillator frequency                           */
#define EVR_OSC_FREQUENCY       MHZ_100

/** \brief External crystal oscillator frequency                                */
#define XTAL_OSC_FREQUENCY      MHZ_20

/** \brief System clock oscillator frequency                                    */
#define SYSCLK_OSC_FREQUENCY    MHZ_20

/** \brief GTM CMU CLK0 target frequency                                        */
#define GTM_CMU_CLK0_FREQUENCY  MHZ_200

/**********************************************************************************************************************
 * Interrupt Priorities  (SRPN values)
 *********************************************************************************************************************/

/** \brief STM0 Compare-0 service request priority number                      */
#define STM0_CMP0_IR_SRPN       (50U)

#endif /* CDD_CONFIG_H_ */

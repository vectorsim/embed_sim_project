/**********************************************************************************************************************
 * \file        cdd_config.h
 * \brief       System-wide configuration constants and interrupt SRPN table
 *              for AURIX TC3xx bare-metal CDD layer.
 *
 * \details     Provides:
 *              - Mathematical and unit-conversion constants
 *              - Compile-time clock frequency constants
 *              - Complete SRPN (Service Request Priority Number) table for all
 *                CDD interrupts — one authoritative location, no duplication
 *
 *              IFX_INTERRUPT() vector registrations live in the .c file that
 *              owns each ISR body — NOT here.  Including IFX_INTERRUPT() in a
 *              header produces multiple-definition linker errors when the header
 *              is included by more than one translation unit.
 *
 *              SRPN assignment policy (TC3xx, 8-bit SRPN, 0=lowest, 255=highest):
 *
 *                  Priority band    SRPN    TOS   ISR / owner
 *                  ───────────────  ──────  ────  ─────────────────────────────
 *                  GPT12 encoder    20      CPU0  IfxGpt12_IncrEnc_init (iLLD)
 *                  STM0 tick        50      CPU0  Stm_00_Cmp_00_Isr (stm_app.c)
 *                  QSPI4 TX/RX/ERR  55-57   CPU0  IfxQspi_SpiMaster (iLLD)
 *                  QSPI2 TX/RX/ERR  60-62   CPU0  IfxQspi_SpiMaster (iLLD)
 *                  GTM ATOM0 CH0    80      CPU1  GTM_Atom_00_Ch_00_Isr (gtm_app.c)
 *                  EVADC G0 Ph-U    90      CPU1  EVADC_G0_Isr (evadc_app.c)
 *                  EVADC G1 Ph-V    91      CPU1  EVADC_G1_Isr (evadc_app.c)
 *                  EVADC G2 Ph-W    92      CPU1  EVADC_G2_Isr (evadc_app.c)
 *                  EVADC G8 DC-lnk  95      CPU1  EVADC_G8_Isr (evadc_app.c)
 *
 * \note        MISRA C:2012 Rule 8.5: IFX_INTERRUPT() must NOT appear in this
 *              header.  Place each ISR registration in its owning .c file.
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
#define PI                      (3.141592653589793)
#define RAD_360                 (6.283185307179586)
#define RAD_270                 (4.712388980384690)
#define RAD_240                 (4.188790204786391)
#define RAD_120                 (2.094395102393195)
#define RAD_90                  (1.570796326794897)

/**********************************************************************************************************************
 * Numerical Utility
 *********************************************************************************************************************/
#define EPSILON_ZERO            (1.0e-10f)

/**********************************************************************************************************************
 * Clock Frequency Constants  [Hz]
 *********************************************************************************************************************/
#define MHZ_300                 (300000000.0f)
#define MHZ_200                 (200000000.0f)
#define MHZ_160                 (160000000.0f)
#define MHZ_100                 (100000000.0f)
#define MHZ_50                   (50000000.0f)
#define MHZ_20                   (20000000.0f)
#define MHZ_5                     (5000000.0f)
#define MHZ_1                     (1000000.0f)

#define EVR_OSC_FREQUENCY       MHZ_100     /**< \brief EVR oscillator  [Hz]        */
#define XTAL_OSC_FREQUENCY      MHZ_20      /**< \brief External crystal [Hz]        */
#define SYSCLK_OSC_FREQUENCY    MHZ_20      /**< \brief System clock    [Hz]        */
#define GTM_CMU_CLK0_FREQUENCY  MHZ_200     /**< \brief GTM CMU CLK0   [Hz]        */

/**********************************************************************************************************************
 * SRPN Table (Service Request Priority Numbers)
 *
 * All CDD interrupt priorities are defined here and nowhere else.
 * Use these constants in every SRC register write and IFX_INTERRUPT() call.
 *********************************************************************************************************************/

/**
 * Macro to register interrupt routine
 */

#define EMBED_SIM_INTERRUPT(Isr, VectabNum, Prio) void __interrupt(Prio) __vector_table(VectabNum) Isr(void)

/* GPT12 encoder zero-pulse  —  CPU0, registered by IfxGpt12_IncrEnc_initConfig */
#define CORE_00_GPT12_ENCODER_ZERO_SRPN         (20U)

/* STM0 1 ms tick  —  CPU0  (cdd_stm_app.c) */
#define STM0_CMP0_IR_SRPN                       (50U)

/* QSPI4 TLE9180D SPI  —  CPU0, registered by iLLD IfxQspi_SpiMaster */
#define CORE_00_QSPI4_TX_SRPN                   (55U)
#define CORE_00_QSPI4_RX_SRPN                   (56U)
#define CORE_00_QSPI4_ERR_SRPN                  (57U)

/* QSPI2 TLF35584 SPI  —  CPU0, registered by iLLD IfxQspi_SpiMaster */
#define CORE_00_QSPI2_TX_SRPN                   (60U)
#define CORE_00_QSPI2_RX_SRPN                   (61U)
#define CORE_00_QSPI2_ERR_SRPN                  (62U)

/* GTM ATOM0_CH0 FOC control loop 20 kHz  —  CPU1  (cdd_gtm_app.c) */
#define CORE_01_ATOM_00_CH_00_CL_SRPN           (80U)

/* EVADC phase current + DC-link  —  CPU1  (cdd_evadc_app.c) */
#define CORE_01_ADC_PHASE_U_SRPN                (90U)   /**< \brief G0 AN00 Phase U  */
#define CORE_01_ADC_PHASE_V_SRPN                (91U)   /**< \brief G1 AN08 Phase V  */
#define CORE_01_ADC_PHASE_W_SRPN                (92U)   /**< \brief G2 AN16 Phase W  */
#define CORE_01_ADC_G_08_CH_08_DC_LINK_SRPN     (95U)   /**< \brief G8 AN40 DC-link  */


/**********************************************************************************************************************
 * Control Loop Frequency
 *********************************************************************************************************************/

/** \brief  FOC control loop (PWM) frequency  [Hz]  — 20 kHz centre-aligned   */
#ifndef BMC_SWC3_ED_CONTROL_FREQUENCY
#define BMC_SWC3_ED_CONTROL_FREQUENCY   (20000U)
#endif

/**********************************************************************************************************************
 * EVADC Service Request Enable Flags
 *
 * Written to SRC_VDACxx_SRx.B.SRE.  Set to 1U to enable the interrupt,
 * 0U to configure the channel without enabling the ISR (e.g. during debug).
 *********************************************************************************************************************/
#define EVADC_ENABLE_PHASE_U_SR         (1U)    /**< \brief G0 Phase U SR enable  */
#define EVADC_ENABLE_PHASE_V_SR         (1U)    /**< \brief G1 Phase V SR enable  */
#define EVADC_ENABLE_PHASE_W_SR         (1U)    /**< \brief G2 Phase W SR enable  */
#define EVADC_ENABLE_DC_LINK_SR         (1U)    /**< \brief G8 DC-link  SR enable  */



#endif /* CDD_CONFIG_H_ */

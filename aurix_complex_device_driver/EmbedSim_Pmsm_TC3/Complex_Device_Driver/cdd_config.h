/**********************************************************************************************************************
 * \file        cdd_config.h
 * \brief       System-wide configuration constants and interrupt SRPN table
 *              for AURIX TC3xx bare-metal CDD layer.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_CONFIG_H_
#define CDD_CONFIG_H_

#include "embed_sim_sys_types.h"

/**********************************************************************************************************************
 * Mathematical Constants
 *********************************************************************************************************************/
#define PI          (3.141592653589793)
#define RAD_360     (6.283185307179586)
#define RAD_270     (4.712388980384690)
#define RAD_240     (4.188790204786391)
#define RAD_120     (2.094395102393195)
#define RAD_90      (1.570796326794897)
#define EPSILON_ZERO (1.0e-10f)

/**********************************************************************************************************************
 * Clock Frequencies [Hz]
 *********************************************************************************************************************/
#define MHZ_300                 (300000000.0f)
#define MHZ_200                 (200000000.0f)
#define MHZ_160                 (160000000.0f)
#define MHZ_100                 (100000000.0f)
#define MHZ_50                   (50000000.0f)
#define MHZ_20                   (20000000.0f)
#define MHZ_5                     (5000000.0f)
#define MHZ_1                     (1000000.0f)

#define EVR_OSC_FREQUENCY       MHZ_100
#define XTAL_OSC_FREQUENCY      MHZ_20
#define SYSCLK_OSC_FREQUENCY    MHZ_20
#define GTM_CMU_CLK0_FREQUENCY  MHZ_200

/**********************************************************************************************************************
 * ISR Macro + SRPN Table
 *********************************************************************************************************************/
#define EMBED_SIM_INTERRUPT(Isr, VectabNum, Prio) \
    void __interrupt(Prio) __vector_table(VectabNum) Isr(void)

#define CORE_00_GPT12_ENCODER_ZERO_SRPN         (20U)
#define STM0_CMP0_IR_SRPN                       (50U)
#define CORE_00_QSPI4_TX_SRPN                   (55U)
#define CORE_00_QSPI4_RX_SRPN                   (56U)
#define CORE_00_QSPI4_ERR_SRPN                  (57U)
#define CORE_00_QSPI2_TX_SRPN                   (60U)
#define CORE_00_QSPI2_RX_SRPN                   (61U)
#define CORE_00_QSPI2_ERR_SRPN                  (62U)
#define CORE_01_ATOM_00_CH_00_CL_SRPN           (80U)
#define CORE_01_ADC_PHASE_U_SRPN                (90U)
#define CORE_01_ADC_PHASE_V_SRPN                (91U)
#define CORE_01_ADC_PHASE_W_SRPN                (92U)
#define CORE_01_ADC_G_08_CH_08_DC_LINK_SRPN     (95U)

/**********************************************************************************************************************
 * GTM Software Dead-Time
 *
 * At 200 MHz CMU CLK0 (5 ns/tick):  28 ticks = 140 ns.
 *
 * The software dead-time creates a CONTROLLED OVERLAP at each switching edge:
 *   LS turns OFF:  sr1_ls = sr1_hs + DT  (DT ticks after HS turns ON)
 *   LS turns ON:   sr0_ls = sr0_hs - DT  (DT ticks before HS turns OFF)
 *
 * The TLE9180D resolves this 140 ns overlap with its own 107 ns internal
 * dead-time (Dt_hs = Dt_ls = 0x00 in SPI config), preventing actual MOSFET
 * shoot-through.  Configure fm_in_diag = WARNING so the IC does not fault
 * during the brief overlap.
 *********************************************************************************************************************/
#ifndef CDD_GTM_SW_DEAD_TIME_TICKS
#define CDD_GTM_SW_DEAD_TIME_TICKS      (28U)   /* 140 ns */
#endif

/**
 * \brief  ADC valley trigger offset from CCU0 reset [CLK0 ticks].
 *         ATOM0_CH7 SR1.  CCU1 fires this many ticks after carrier valley.
 *         40 ticks = 200 ns.  Must be > CDD_GTM_SW_DEAD_TIME_TICKS (140 ns).
 */
#ifndef CDD_GTM_ADC_VALLEY_OFFSET_TICKS
#define CDD_GTM_ADC_VALLEY_OFFSET_TICKS (40U)   /* 200 ns */
#endif

/**********************************************************************************************************************
 * GTM Gate-Driver HS Polarity Flag
 *
 * Selects the ATOM SL bit value for ALL three HS channels (CH2, CH4, CH6).
 *
 * In TC38x ATOM SOMC, the output state machine is:
 *     ~SL   immediately after AGC-master reset and after CM0  (= gate OFF)
 *      SL   from CM1 to CM0                                   (= gate ON)
 *
 * CDD_GTM_HS_ACTIVE_LOW = 1U  (DEFAULT — TLE9180D AP32541, /IHx active LOW)
 *     Gate OFF = pin HIGH = ~SL  →  SL = 0   (ATOM_HS_CH_SL = 0x0U)
 *     Gate ON  = pin LOW  =  SL
 *     Initial state: ~SL = 1 → /IH = HIGH → HS gate OFF ✓ (safe at startup)
 *
 * CDD_GTM_HS_ACTIVE_LOW = 0U  (standard gate driver, IHx active HIGH)
 *     Gate OFF = pin LOW  = ~SL  →  SL = 1   (ATOM_HS_CH_SL = 0x1U)
 *     Gate ON  = pin HIGH =  SL
 *     Initial state: ~SL = 0 → IH = LOW → HS gate OFF ✓ (safe at startup)
 *
 * LS channels (ILx active HIGH) always use SL = 0x0U regardless of this flag.
 *     Initial state: ~SL = ~0 = 1 → IL = HIGH → LS gate ON (active freewheeling
 *     while HS is still OFF at startup — no shoot-through risk).
 *********************************************************************************************************************/
#ifndef CDD_GTM_HS_ACTIVE_LOW
#define CDD_GTM_HS_ACTIVE_LOW   (1U)   /* TLE9180D /IHx active LOW (default) */
#endif

/**********************************************************************************************************************
 * Control Loop Frequency
 *********************************************************************************************************************/
#ifndef CDD_CONTROL_LOOP_FREQUENCY
#define CDD_CONTROL_LOOP_FREQUENCY   (20000U)   /* 20 kHz */
#endif

/**********************************************************************************************************************
 * EVADC SR Enable
 *********************************************************************************************************************/
#define EVADC_ENABLE_PHASE_U_SR     (1U)
#define EVADC_ENABLE_PHASE_V_SR     (1U)
#define EVADC_ENABLE_PHASE_W_SR     (1U)
#define EVADC_ENABLE_DC_LINK_SR     (1U)

#endif /* CDD_CONFIG_H_ */

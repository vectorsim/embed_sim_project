/**********************************************************************************************************************
 * \file        cdd_sys_utility.h
 * \brief       System-level utility interfaces: clock tree, watchdog, spinlock,
 *              and CPU interrupt control for AURIX TC3xx.
 *
 * \details     Covers:
 *              - CPU core identification
 *              - CPU / Safety watchdog EndInit sequence helpers
 *              - Clock-tree frequency interrogation (SCU / PLL / GTM / STM / SPB)
 *              - GTM CMU CLK0 frequency configuration
 *              - CPU interrupt enable / disable / restore
 *              - Spinlock acquire / release using CddAsm_CmpAndSwap
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier, matching .c definition
 *              - Rule  8.6 : All definitions reside in cdd_sys_utility.c
 *
 * \copyright   Copyright (C) SEPL UG 2024
 *********************************************************************************************************************/

#ifndef CDD_SYS_UTILITY_H_
#define CDD_SYS_UTILITY_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"   /* embed_sim_sys_types.h + embed_sim_compiler.h */

/**********************************************************************************************************************
 * Function Prototypes — Core Identification
 *********************************************************************************************************************/

/** \brief  Returns the hardware CPU core ID of the executing core.
 *  \return Core ID  (0 = CPU0, 1 = CPU1, ...)  [dimensionless]
 */
extern uint32_T CddSys_GetCoreId(void);

/**********************************************************************************************************************
 * Function Prototypes — Busy-Wait
 *********************************************************************************************************************/

/** \brief  Software NOP delay loop.
 *  \param[in]  InnerLoop   Inner iteration count  [dimensionless]
 *  \param[in]  OuterLoop   Outer iteration count  [dimensionless]
 */
extern void CddSys_NopDelay(uint32_T InnerLoop, uint32_T OuterLoop);

/**********************************************************************************************************************
 * Function Prototypes — Watchdog Password Retrieval
 *********************************************************************************************************************/

/** \brief  Returns the current password for the CPU0 watchdog register.   \return CPU0 WDT password */
extern uint32_T CddSys_GetWdt00Pwd(void);

/** \brief  Returns the current password for the CPU1 watchdog register.   \return CPU1 WDT password */
extern uint32_T CddSys_GetWdt01Pwd(void);

/** \brief  Returns the current password for the Safety watchdog register. \return Safety WDT password */
extern uint32_T CddSys_GetSafetyWdtPwd(void);

/**********************************************************************************************************************
 * Function Prototypes — CPU Watchdog EndInit Control
 *********************************************************************************************************************/

/** \brief  Clears ENDINIT for CPU0 watchdog (unlocks ENDINIT-protected registers). */
extern void CddSys_ClearWdt00EndInit(void);

/** \brief  Clears ENDINIT for CPU1 watchdog. */
extern void CddSys_ClearWdt01EndInit(void);

/** \brief  Clears ENDINIT for the currently executing CPU (dispatches to CPU0/CPU1). */
extern void CddSys_ClearWdtEndInit(void);

/** \brief  Sets ENDINIT for CPU0 watchdog (re-locks ENDINIT-protected registers). */
extern void CddSys_SetWdt00EndInit(void);

/** \brief  Sets ENDINIT for CPU1 watchdog. */
extern void CddSys_SetWdt01EndInit(void);

/** \brief  Sets ENDINIT for the currently executing CPU. */
extern void CddSys_SetWdtEndInit(void);

/** \brief  Disables the CPU0 watchdog (DR bit, ds1 P.980). */
extern void CddSys_DisableWdt00(void);

/** \brief  Disables the CPU1 watchdog. */
extern void CddSys_DisableWdt01(void);

/**********************************************************************************************************************
 * Function Prototypes — Safety Watchdog EndInit Control
 *********************************************************************************************************************/

/** \brief  Clears ENDINIT for the Safety watchdog. */
extern void CddSys_ClearSafetyWdtEndInit(void);

/** \brief  Sets ENDINIT for the Safety watchdog. */
extern void CddSys_SetSafetyWdtEndInit(void);

/** \brief  Disables the Safety watchdog (WDTSCON1.DR, ds1 P.977). */
extern void CddSys_DisableSafetyWdt(void);

/**********************************************************************************************************************
 * Function Prototypes — Clock Tree Frequency Interrogation
 *********************************************************************************************************************/

/** \brief  Returns EVR (internal backup) oscillator frequency.     \return [Hz] */
extern real64_T CddSys_GetEvrFreq(void);

/** \brief  Returns system clock oscillator frequency.              \return [Hz] */
extern real64_T CddSys_GetSysClkFreq(void);

/** \brief  Returns external crystal oscillator frequency.          \return [Hz] */
extern real64_T CddSys_GetExtOscFreq(void);

/** \brief  Returns the PLL input clock frequency.                  \return [Hz] */
extern real64_T CddSys_GetPrimaryOscFreq(void);

/** \brief  Returns system PLL0 output frequency.                   \return [Hz] */
extern real64_T CddSys_GetPll00Freq(void);

/** \brief  Returns system PLL1 output frequency.                   \return [Hz] */
extern real64_T CddSys_GetPll01Freq(void);

/** \brief  Returns clock source 0 frequency.                       \return [Hz] */
extern real64_T CddSys_GetSrc00Freq(void);

/** \brief  Returns clock source 1 frequency.                       \return [Hz] */
extern real64_T CddSys_GetSrc01Freq(void);

/** \brief  Returns SRI bus frequency.                              \return [Hz] */
extern real64_T CddSys_GetSriFreq(void);

/** \brief  Returns CPU0 core frequency.                            \return [Hz] */
extern real64_T CddSys_GetCpuFreq(void);

/** \brief  Returns STM clock frequency.                            \return [Hz] */
extern real64_T CddSys_GetStmFreq(void);

/** \brief  Returns SPB (System Peripheral Bus) frequency.          \return [Hz] */
extern real64_T CddSys_GetSpbFreq(void);

/** \brief  Returns GTM clock source frequency.                     \return [Hz] */
extern real64_T CddSys_GetGtmSrcFreq(void);

/** \brief  Returns GTM module frequency.                           \return [Hz] */
extern real64_T CddSys_GetGtmFreq(void);

/** \brief  Returns GTM CMU cluster 0 frequency.                    \return [Hz] */
extern real64_T CddSys_GetGtmClusterFreq(void);

/** \brief  Returns GTM CMU global clock frequency.                 \return [Hz] */
extern real64_T CddSys_GetGtmCmuGlobalFreq(void);

/** \brief  Returns ADC clock frequency.                            \return [Hz] */
extern real64_T CddSys_GetAdcFreq(void);

/** \brief  Returns QSPI peripheral clock frequency.                \return [Hz] */
extern real64_T CddSys_GetQspiFreq(void);

/**********************************************************************************************************************
 * Function Prototypes — GTM CMU CLK0 Configuration
 *********************************************************************************************************************/

/**
 * \brief   Configures GTM CMU CLK0 to the requested frequency.
 * \param[in]  CmuClk00Freq   Target frequency  [Hz]
 */
extern void CddSys_SetGtmCmuClk00Freq(real64_T CmuClk00Freq);

/** \brief  Returns the currently configured GTM CMU CLK0 frequency.  \return [Hz] */
extern real64_T CddSys_GetGtmCmuClk00Freq(void);

/**********************************************************************************************************************
 * Function Prototypes — CPU Interrupt Control
 *********************************************************************************************************************/

/**
 * \brief   Returns 1 if CPU interrupts are currently enabled, 0 otherwise.
 * \return  Interrupt-enable state (1 = enabled, 0 = disabled)  [dimensionless]
 */
extern uint32_T CddSys_IsIrqEnabled(void);

/**
 * \brief   Disables CPU interrupts and returns the previous enable state.
 * \return  Previous interrupt state (pass to CddSys_RestoreIrq)  [dimensionless]
 */
extern uint32_T CddSys_DisableIrq(void);

/**
 * \brief   Restores CPU interrupt state saved by CddSys_DisableIrq.
 * \param[in]  PrevState  Value returned by CddSys_DisableIrq()
 */
extern void CddSys_RestoreIrq(uint32_T PrevState);

/**********************************************************************************************************************
 * Function Prototypes — Spinlock
 *********************************************************************************************************************/

/**
 * \brief   Attempts to acquire a spinlock using CMPSWAP.W (non-blocking).
 * \param[in,out]  LockPtr  Pointer to the lock variable (0 = free, 1 = held)
 * \return  1 if lock was acquired, 0 if already held by another core  [dimensionless]
 */
extern uint32_T CddSys_AcquireSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr);

/**
 * \brief   Releases a previously acquired spinlock.
 * \param[in,out]  LockPtr  Pointer to the lock variable
 */
extern void CddSys_ReleaseSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr);

/**
 * \brief   Tests whether two single-precision floats are equal within a tolerance.
 *
 * \param[in]  Lhs      First operand.
 * \param[in]  Rhs      Second operand.
 * \param[in]  Epsilon  Maximum permitted absolute difference  (>= 0.0f).
 * \return  1U if |Lhs - Rhs| <= Epsilon, 0U otherwise.  [dimensionless]
 */
extern uint32_T CddSys_AreEqual(real32_T Lhs, real32_T Rhs, real32_T Epsilon);

#endif /* CDD_SYS_UTILITY_H_ */

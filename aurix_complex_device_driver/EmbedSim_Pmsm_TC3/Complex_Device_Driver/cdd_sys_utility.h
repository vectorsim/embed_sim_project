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
 *              - Spinlock acquire / release using ASM_Cmp_And_Swap
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
#include "cdd_config.h"

/**********************************************************************************************************************
 * Function Prototypes — Core Identification
 *********************************************************************************************************************/

/** \brief  Returns the hardware CPU core ID of the executing core.
 *  \return Core ID  (0 = CPU0, 1 = CPU1, ...)
 */
extern uint32_T Get_Core_Id(void);

/**********************************************************************************************************************
 * Function Prototypes — Busy-Wait
 *********************************************************************************************************************/

/** \brief  Software NOP delay loop.
 *  \param  InnerLoop   Inner iteration count
 *  \param  OuterLoop   Outer iteration count
 */
extern void Nop_Delay(uint32_T InnerLoop, uint32_T OuterLoop);

/**********************************************************************************************************************
 * Function Prototypes — Watchdog Password Retrieval
 *********************************************************************************************************************/

/** \brief  Returns the current password for the CPU0 watchdog register.   \return CPU0 WDT password */
extern uint32_T Get_CPU_00_WDT_Pwd(void);

/** \brief  Returns the current password for the CPU1 watchdog register.   \return CPU1 WDT password */
extern uint32_T Get_CPU_01_WDT_Pwd(void);

/** \brief  Returns the current password for the Safety watchdog register. \return Safety WDT password */
extern uint32_T Get_Safety_WDT_Pwd(void);

/**********************************************************************************************************************
 * Function Prototypes — CPU Watchdog EndInit Control
 *********************************************************************************************************************/

/** \brief  Clears ENDINIT for CPU0 watchdog (unlocks ENDINIT-protected registers). */
extern void Clear_CPU_00_WDT_EndInit(void);

/** \brief  Clears ENDINIT for CPU1 watchdog. */
extern void Clear_CPU_01_WDT_EndInit(void);

/** \brief  Clears ENDINIT for the currently executing CPU (dispatches to CPU0/CPU1). */
extern void Clear_CPU_WDT_EndInit(void);

/** \brief  Sets ENDINIT for CPU0 watchdog (re-locks ENDINIT-protected registers). */
extern void Set_CPU_00_WDT_EndInit(void);

/** \brief  Sets ENDINIT for CPU1 watchdog. */
extern void Set_CPU_01_WDT_EndInit(void);

/** \brief  Sets ENDINIT for the currently executing CPU. */
extern void Set_CPU_WDT_EndInit(void);

/** \brief  Disables the CPU0 watchdog (DR bit, ds1 P.980). */
extern void Disable_CPU_00_WDT(void);

/** \brief  Disables the CPU1 watchdog. */
extern void Disable_CPU_01_WDT(void);

/**********************************************************************************************************************
 * Function Prototypes — Safety Watchdog EndInit Control
 *********************************************************************************************************************/

/** \brief  Clears ENDINIT for the Safety watchdog. */
extern void Clear_Safety_WDT_EndInit(void);

/** \brief  Sets ENDINIT for the Safety watchdog. */
extern void Set_Safety_WDT_EndInit(void);

/** \brief  Disables the Safety watchdog (WDTSCON1.DR, ds1 P.977). */
extern void Disable_Safety_WDT(void);

/**********************************************************************************************************************
 * Function Prototypes — Clock Tree Frequency Interrogation
 *********************************************************************************************************************/

/** \brief  Returns EVR (internal backup) oscillator frequency.     \return [Hz] */
extern real64_T Get_EVR_Frequency(void);

/** \brief  Returns system clock oscillator frequency.              \return [Hz] */
extern real64_T Get_SysClk_Frequency(void);

/** \brief  Returns external crystal oscillator frequency.          \return [Hz] */
extern real64_T Get_External_OSC_Frequency(void);

/** \brief  Returns the PLL input clock frequency (selected by SCU_SYSPLLCON0.INSEL). \return [Hz] */
extern real64_T Get_Primary_OSC_Frequency(void);

/** \brief  Returns system PLL0 output frequency.                   \return [Hz] */
extern real64_T Get_SYS_PLL_00_Frequency(void);

/** \brief  Returns system PLL1 output frequency.                   \return [Hz] */
extern real64_T Get_SYS_PLL_01_Frequency(void);

/** \brief  Returns clock source 0 frequency (drives SRI, STM, ...). \return [Hz] */
extern real64_T Get_Source_00_Frequency(void);

/** \brief  Returns clock source 1 frequency (drives ADC, ...).     \return [Hz] */
extern real64_T Get_Source_01_Frequency(void);

/** \brief  Returns SRI bus frequency.                              \return [Hz] */
extern real64_T Get_SRI_Frequency(void);

/** \brief  Returns CPU0 core frequency.                            \return [Hz] */
extern real64_T Get_CPU_Frequency(void);

/** \brief  Returns STM clock frequency.                            \return [Hz] */
extern real32_T Get_STM_Frequency(void);

/** \brief  Returns SPB (System Peripheral Bus) frequency.          \return [Hz] */
extern real64_T Get_SPB_Frequency(void);

/** \brief  Returns GTM clock source frequency.                     \return [Hz] */
extern real64_T Get_GTM_Source_Frequency(void);

/** \brief  Returns GTM module frequency.                           \return [Hz] */
extern real64_T Get_GTM_Frequency(void);

/** \brief  Returns GTM CMU cluster 0 frequency.                    \return [Hz] */
extern real64_T Get_GTM_Cluster_Frequency(void);

/** \brief  Returns GTM CMU global clock frequency.                 \return [Hz] */
extern real64_T Get_GTM_CMU_Global_Frequency(void);

/** \brief  Returns ADC clock frequency.                            \return [Hz] */
extern real64_T Get_ADC_Frequency(void);

/**********************************************************************************************************************
 * Function Prototypes — GTM CMU CLK0 Configuration
 *********************************************************************************************************************/

/**
 * \brief   Configures GTM CMU CLK0 to the requested frequency.
 * \param   CMU_CLK_00_Frequency   Target frequency  [Hz]
 */
extern void Set_GTM_CMU_CLK_00_Frequency(real64_T CMU_CLK_00_Frequency);

/** \brief  Returns the currently configured GTM CMU CLK0 frequency. \return [Hz] */
extern real64_T Get_GTM_CMU_CLK_00_Frequency(void);

/**********************************************************************************************************************
 * Function Prototypes — CPU Interrupt Control
 *********************************************************************************************************************/

/**
 * \brief   Returns 1 if CPU interrupts are currently enabled, 0 otherwise.
 * \return  Interrupt-enable state (1 = enabled, 0 = disabled)
 */
extern uint32_T Is_CPU_Interrupt_Enabled(void);

/**
 * \brief   Disables CPU interrupts and returns the previous enable state.
 * \return  Previous interrupt state (pass to Restore_CPU_Interrupt)
 */
extern uint32_T Disable_CPU_Interrupt(void);

/**
 * \brief   Restores CPU interrupt state saved by Disable_CPU_Interrupt.
 * \param   Previous_State   Value returned by Disable_CPU_Interrupt()
 */
extern void Restore_CPU_Interrupt(uint32_T Previous_State);

/**********************************************************************************************************************
 * Function Prototypes — Spinlock
 *********************************************************************************************************************/

/**
 * \brief   Attempts to acquire a spinlock using CMPSWAP.W (non-blocking).
 * \param   Lock_Ptr   Pointer to the lock variable (0 = free, 1 = held)
 * \return  1 if lock was acquired, 0 if already held by another core
 */
extern uint32_T Acquire_Spin_Lock(uint32_T * const Lock_Ptr);

/**
 * \brief   Releases a previously acquired spinlock.
 * \param   Lock_Ptr   Pointer to the lock variable
 */
extern void Release_Spin_Lock(uint32_T * const Lock_Ptr);

#endif /* CDD_SYS_UTILITY_H_ */

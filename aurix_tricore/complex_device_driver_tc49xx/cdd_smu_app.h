/**********************************************************************************************************************
 * \file      cdd_smu.h
 * \brief     Public interface for the CDD Safety Management Unit (SMU) driver.
 *
 * \details   Owns every SMU register access in the CDD layer:
 *              - SAFE instance alarm-action configuration (4-bit codes).
 *              - SAFE state-machine control (START -> RUN, FAULT -> RUN).
 *              - SAFE recovery-timer configuration and alarm binding.
 *              - SAFE alarm-status read/clear (W1C).
 *              - SAFE fault-signaling protocol (FSP) and port-emergency-stop
 *                (PES) configuration.
 *              - Register-key management (temporary and permanent lock).
 *
 *            The module deliberately does NOT depend on the iLLD. It talks
 *            directly to MODULE_SMU via the register definitions in
 *            IfxSmu_regdef.h.
 *
 * \note      Logical results are returned as uint32_T: 0U = false, 1U = true.
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : CddSmu_Pascal_Snake_Case
 *              - Parameters     : PascalCase
 *              - Local variables: lowerPascalCase
 *              - Macros         : CDD_SMU_UPPER_SNAKE_CASE
 *              - Typedefs       : Cdd_Smu_Pascal_Snake_Case_T
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.1 : explicit return types
 *              - Rule  8.4 : compatible declaration visible at every definition
 *              - Rule 14.4 : explicit comparisons in every controlling expression
 *              - Rule 15.5 : single exit point
 *              - Rule 17.2 : no recursion
 *
 * \version   1.0.0
 * \date      2026-10-22
 * \author    EmbedSim / EV Light Vehicle Foundation
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_SMU_H_
#define CDD_SMU_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** Magic key that unlocks SAFE / CS / GCC SMU configuration registers. */
#define CDD_SMU_KEY_UNLOCK              (0x000000BCU)

/** Value written to KEYS.PERLCK to lock configuration permanently. */
#define CDD_SMU_KEY_PERM_LOCK           (0xFFU)

/** SAFE instance selectors used throughout this module. */
#define CDD_SMU_SAFE_0                  (0U)
#define CDD_SMU_SAFE_1                  (1U)

/** SAFE state-machine selector (SSM0 / SSM1). */
#define CDD_SMU_SSM_0                   (0U)
#define CDD_SMU_SSM_1                   (1U)

/** FSP index selectors. */
#define CDD_SMU_FSP_0                   (0U)
#define CDD_SMU_FSP_1                   (1U)

/** Recovery-timer selectors. */
#define CDD_SMU_RT_0                    (0U)
#define CDD_SMU_RT_1                    (1U)

/** Command codes accepted by SMU_SAFE.CMD. */
#define CDD_SMU_CMD_START               (0U)
#define CDD_SMU_CMD_ACTIVATE_FSP        (1U)
#define CDD_SMU_CMD_RELEASE_FSP         (2U)
#define CDD_SMU_CMD_ACTIVATE_PES        (3U)
#define CDD_SMU_CMD_STOP_RT             (4U)
#define CDD_SMU_CMD_ALARM_CLR           (5U)
#define CDD_SMU_CMD_TRIGGER_ALARM       (6U)
#define CDD_SMU_CMD_ALIVE_TEST          (7U)

/** SAFE state-machine states (SMU_SAFE.DBG.SSMx). */
#define CDD_SMU_STATE_START             (0U)
#define CDD_SMU_STATE_RUN               (1U)
#define CDD_SMU_STATE_FAULT             (2U)
#define CDD_SMU_STATE_UNSPECIFIED       (3U)

/** SAFE internal alarm actions (CON[0..3] combine to a 4-bit code). */
#define CDD_SMU_ACTION_DISABLED         (0U)
#define CDD_SMU_ACTION_IGCS0_REQ        (2U)
#define CDD_SMU_ACTION_IGCS1_REQ        (3U)
#define CDD_SMU_ACTION_IGCS2_REQ        (4U)
#define CDD_SMU_ACTION_NMI0_REQ         (5U)
#define CDD_SMU_ACTION_RESET0_REQ       (6U)
#define CDD_SMU_ACTION_CPU_RESET0_REQ   (7U)
#define CDD_SMU_ACTION_PPU_RESET        (9U)
#define CDD_SMU_ACTION_NMI1_REQ         (13U)
#define CDD_SMU_ACTION_RESET1_REQ       (14U)
#define CDD_SMU_ACTION_CPU_RESET1_REQ   (15U)

/** PES action bits (SMU_SAFE.PES). */
#define CDD_SMU_PES_IGCS0               (0x00000001U)
#define CDD_SMU_PES_IGCS1               (0x00000002U)
#define CDD_SMU_PES_IGCS2               (0x00000004U)
#define CDD_SMU_PES_NMI0                (0x00000008U)
#define CDD_SMU_PES_CPURST0             (0x00000010U)
#define CDD_SMU_PES_PPURST0             (0x00000020U)
#define CDD_SMU_PES_NMI1                (0x00000040U)
#define CDD_SMU_PES_CPURST1             (0x00000080U)
#define CDD_SMU_PES_FSP0                (0x00010000U)
#define CDD_SMU_PES_FSP1                (0x00020000U)

/** FSP operating modes (SMU_SAFE.FSP[x].CON.MODE). */
#define CDD_SMU_FSP_MODE_BISTABLE       (0U)
#define CDD_SMU_FSP_MODE_DUAL_RAIL      (1U)
#define CDD_SMU_FSP_MODE_TIME_SWITCH    (2U)

/** Logical values. */
#define CDD_SMU_FALSE                   (0U)
#define CDD_SMU_TRUE                    (1U)

/** Sentinel group value that marks an unused recovery-timer slot. */
#define CDD_SMU_ALARM_SLOT_UNUSED       (0xFFU)

/*********************************************************************************************************************/
/*-------------------------------------------------Type Definitions--------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * @brief   One alarm reference used when binding alarms to a recovery timer.
 */
typedef struct
{
    uint8_T Group;      /**< SAFE alarm group (0..16), or CDD_SMU_ALARM_SLOT_UNUSED. */
    uint8_T Position;   /**< Bit position inside the group (0..31).                   */
} Cdd_Smu_AlarmRef_T;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/*--- Initialisation ----------------------------------------------------------*/

/**
 * @brief   Clears all SAFE alarm status flags and moves SSM0 to the RUN state.
 *
 * @param[in] SafeIndex  CDD_SMU_SAFE_0 or CDD_SMU_SAFE_1.
 *
 * @return  CDD_SMU_TRUE if the START command was accepted, CDD_SMU_FALSE otherwise.
 */
extern uint32_T CddSmu_InitialiseSafety(uint32_T SafeIndex);

/*--- Alarm-action configuration ---------------------------------------------*/

/**
 * @brief   Programs the 4-bit internal action code for one SAFE alarm.
 */
extern void CddSmu_SetAlarmAction(uint8_T AlarmGroup,
                                  uint8_T AlarmPos,
                                  uint8_T Action,
                                  uint32_T SafeIndex);

/**
 * @brief   Reads back the 4-bit action code for one SAFE alarm.
 */
extern uint8_T CddSmu_GetAlarmAction(uint8_T AlarmGroup,
                                     uint8_T AlarmPos,
                                     uint32_T SafeIndex);

/*--- State-machine control --------------------------------------------------*/

/**
 * @brief   Issues a START command to move the selected SAFE SSM to RUN.
 *
 * @return  CDD_SMU_TRUE on acceptance (STS.RES == 0), CDD_SMU_FALSE otherwise.
 */
extern uint32_T CddSmu_StartSafetySsm(uint32_T SafeIndex, uint32_T SsmId);

/**
 * @brief   Returns the current state of the selected SAFE SSM.
 */
extern uint8_T CddSmu_GetSafetySsmState(uint32_T SafeIndex, uint32_T SsmId);

/**
 * @brief   Enables the FAULT-to-RUN transition for the selected SSM.
 */
extern void CddSmu_EnableFaultToRunTransition(uint32_T SafeIndex, uint32_T SsmId);

/*--- Recovery timer ---------------------------------------------------------*/

/**
 * @brief   Configures prescaler and duration of the SAFE recovery timer.
 *
 * @param[in] SafeIndex    SAFE instance.
 * @param[in] Prescaler    RTC.PRE (0..7).
 * @param[in] Duration     RTC.RTD (24-bit).
 * @param[in] EnableRt0    CDD_SMU_TRUE to enable RT0.
 * @param[in] EnableRt1    CDD_SMU_TRUE to enable RT1.
 */
extern void CddSmu_ConfigureRecoveryTimer(uint32_T SafeIndex,
                                          uint8_T Prescaler,
                                          uint32_T Duration,
                                          uint32_T EnableRt0,
                                          uint32_T EnableRt1);

/**
 * @brief   Binds up to four alarms to one SAFE recovery timer.
 *
 * @param[in] SafeIndex  SAFE instance.
 * @param[in] TimerNum   CDD_SMU_RT_0 or CDD_SMU_RT_1.
 * @param[in] Alarms     Array of exactly four alarm references
 *                       (unused slots set Group = CDD_SMU_ALARM_SLOT_UNUSED).
 */
extern void CddSmu_BindRecoveryTimerAlarms(uint32_T SafeIndex,
                                           uint32_T TimerNum,
                                           const Cdd_Smu_AlarmRef_T Alarms[4]);

/**
 * @brief   Stops the requested SAFE recovery timer.
 *
 * @return  CDD_SMU_TRUE if the command was accepted, CDD_SMU_FALSE otherwise.
 */
extern uint32_T CddSmu_StopRecoveryTimer(uint32_T SafeIndex, uint8_T TimerNum);

/*--- Alarm status -----------------------------------------------------------*/

/**
 * @brief   Returns CDD_SMU_TRUE if the requested SAFE alarm is currently set.
 */
extern uint32_T CddSmu_GetAlarmStatus(uint8_T AlarmGroup,
                                      uint8_T AlarmPos,
                                      uint32_T SafeIndex);

/**
 * @brief   Clears the requested SAFE alarm status flag (W1C + ASCE command).
 */
extern void CddSmu_ClearAlarmStatus(uint8_T AlarmGroup,
                                    uint8_T AlarmPos,
                                    uint32_T SafeIndex);

/*--- FSP / PES --------------------------------------------------------------*/

/**
 * @brief   Configures the Fault Signalling Protocol (FSP) timing and mode.
 */
extern void CddSmu_ConfigureFsp(uint32_T SafeIndex,
                                uint32_T FspIndex,
                                uint8_T Mode,
                                uint8_T Prescalar1,
                                uint8_T Prescalar2,
                                uint16_T FaultStateDurationHigh,
                                uint16_T FaultStateDurationLow);

/**
 * @brief   OR-merges the requested PES action bits into SMU_SAFE.PES.
 */
extern void CddSmu_ConfigurePes(uint32_T SafeIndex, uint32_T PesBits);

/*--- Key management ---------------------------------------------------------*/

/**
 * @brief   Temporarily locks the SAFE configuration registers.
 */
extern void CddSmu_LockSafetyConfig(uint32_T SafeIndex);

/**
 * @brief   Permanently locks the SAFE configuration registers.
 *
 * @note    Irreversible until reset; use only in the final safety init.
 */
extern void CddSmu_PermanentlyLockSafetyConfig(uint32_T SafeIndex);

/*--- High-level convenience -------------------------------------------------*/

/**
 * @brief   Forwards a CPU-watchdog timeout on the calling core to the SMU.
 *
 * @details Configures the CPU-WDT alarm action, binds it to RT0, enables
 *          RT0 with the given duration, and starts SSM0.
 *
 * @param[in] Action      CDD_SMU_ACTION_* code.
 * @param[in] SafeIndex   SAFE instance.
 * @param[in] RtDuration  Recovery-timer timeout in SMU clock cycles.
 *
 * @return  CDD_SMU_TRUE on success, CDD_SMU_FALSE otherwise.
 */
extern uint32_T CddSmu_ForwardCpuWdt(uint8_T Action,
                                     uint32_T SafeIndex,
                                     uint32_T RtDuration);

#endif /* CDD_SMU_H_ */

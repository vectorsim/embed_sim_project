/**********************************************************************************************************************
 * \file      cdd_smu.c
 * \brief     Implementation of the CDD Safety Management Unit (SMU) driver.
 *
 * \details   Direct register-level access to MODULE_SMU. Does not depend on
 *            the iLLD. All writes are guarded by the SMU key sequence.
 *
 * \note      Logical results are returned as uint32_T: 0U = false, 1U = true.
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "cdd_smu_app.h"
#include "cdd_sys_util.h"
#include "IfxSmu_reg.h"

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/* CPU-WDT alarm mapping: coreId -> {group, bit}.
 * NOTE: device-specific; replace with values from the TC4x SMU alarm map. */
static const uint8_T CddSmu_CpuWdtGroup[6] = { 3U, 3U, 3U, 3U, 3U, 3U };
static const uint8_T CddSmu_CpuWdtBit  [6] = { 18U, 19U, 20U, 21U, 22U, 23U };

/*********************************************************************************************************************/
/*---------------------------------------------Private Helper Functions----------------------------------------------*/
/*********************************************************************************************************************/

/* Unlock SAFE configuration registers. */
static void CddSmu_UnlockSafe(uint32_T SafeIndex)
{
    MODULE_SMU.SAFE[SafeIndex].KEYS.U = CDD_SMU_KEY_UNLOCK;
}


/* Temporarily lock SAFE configuration registers (not permanent). */
static void CddSmu_TempLockSafe(uint32_T SafeIndex)
{
    MODULE_SMU.SAFE[SafeIndex].KEYS.U = 0U;
}

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

uint32_T CddSmu_InitialiseSafety(uint32_T SafeIndex)
{
    Ifx_SMU_SAFE_CMD cmd;
    uint32_T         grp;
    uint32_T         result;

    /* 1. Move SSM0 from START to RUN FIRST. */
    result = CddSmu_StartSafetySsm(SafeIndex, CDD_SMU_SSM_0);

    if (CDD_SMU_TRUE == result)
    {
        /* 2. NOW issue the alarm-status-clear command. */
        cmd.U     = MODULE_SMU.SAFE[SafeIndex].CMD.U;
        cmd.B.CMD = CDD_SMU_CMD_ALARM_CLR;
        cmd.B.ARG = 0U;
        MODULE_SMU.SAFE[SafeIndex].CMD.U = cmd.U;

        /* 3. NOW write 1-to-clear every status flag (W1C in RUN state). */
        for (grp = 0U; grp < 17U; grp++)
        {
            MODULE_SMU.SAFE[SafeIndex].AGSF[grp].STS.U = 0xFFFFFFFFU;
        }
    }

    return result;
}

void CddSmu_SetAlarmAction(uint8_T AlarmGroup,
                           uint8_T AlarmPos,
                           uint8_T Action,
                           uint32_T SafeIndex)
{
    uint32_T mask;
    uint32_T act;

    mask = (uint32_T)(1UL << AlarmPos);
    act  = (uint32_T)Action;

    CddSmu_UnlockSafe(SafeIndex);

    MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[0].U =
        (MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[0].U & ~mask)
      | (((act >> 0U) & 0x1U) ? mask : 0U);

    MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[1].U =
        (MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[1].U & ~mask)
      | (((act >> 1U) & 0x1U) ? mask : 0U);

    MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[2].U =
        (MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[2].U & ~mask)
      | (((act >> 2U) & 0x1U) ? mask : 0U);

    MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[3].U =
        (MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[3].U & ~mask)
      | (((act >> 3U) & 0x1U) ? mask : 0U);

    CddSmu_TempLockSafe(SafeIndex);
}


uint8_T CddSmu_GetAlarmAction(uint8_T AlarmGroup,
                              uint8_T AlarmPos,
                              uint32_T SafeIndex)
{
    uint32_T bit = (uint32_T)(1UL << AlarmPos);
    uint8_T  act = 0U;

    if (0U != (MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[0].U & bit)) { act = (uint8_T)(act | 0x1U); }
    if (0U != (MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[1].U & bit)) { act = (uint8_T)(act | 0x2U); }
    if (0U != (MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[2].U & bit)) { act = (uint8_T)(act | 0x4U); }
    if (0U != (MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].CON[3].U & bit)) { act = (uint8_T)(act | 0x8U); }

    return act;
}

uint32_T CddSmu_StartSafetySsm(uint32_T SafeIndex, uint32_T SsmId)
{
    Ifx_SMU_SAFE_CMD cmd;
    uint32_T          state;

    /* 1. Make sure Endinit is released for the SAFE instance.
     *    (If you already cleared it globally, delete this call.) */
    /* IfxScuWdt_clearSafetyEndinit(IfxScuWdt_getSafetyWatchdogPassword()); */

    /* 2. If already running, do nothing and report success. */
    state = (uint32_T)CddSmu_GetSafetySsmState(SafeIndex, SsmId);

    if (CDD_SMU_STATE_START == state)
    {
        /* 3. Issue START. ARG = which SSM (0 or 1). */
        cmd.U     = 0U;
        cmd.B.CMD = CDD_SMU_CMD_START;
        cmd.B.ARG = SsmId;
        MODULE_SMU.SAFE[SafeIndex].CMD.U = cmd.U;

        /* 4. Read the state back — this is the real success indicator. */
        state = (uint32_T)CddSmu_GetSafetySsmState(SafeIndex, SsmId);
    }

    return (CDD_SMU_STATE_RUN == state) ? CDD_SMU_TRUE : CDD_SMU_FALSE;
}


uint8_T CddSmu_GetSafetySsmState(uint32_T SafeIndex, uint32_T SsmId)
{
    uint8_T state;

    state = (uint8_T)((CDD_SMU_SSM_0 == SsmId)
                    ? MODULE_SMU.SAFE[SafeIndex].DBG.B.SSM0
                    : MODULE_SMU.SAFE[SafeIndex].DBG.B.SSM1);

    return state;
}


void CddSmu_EnableFaultToRunTransition(uint32_T SafeIndex, uint32_T SsmId)
{
    CddSmu_UnlockSafe(SafeIndex);

    if (CDD_SMU_SSM_0 == SsmId)
    {
        MODULE_SMU.SAFE[SafeIndex].AGC.B.EFRST0 = 1U;
    }
    else
    {
        MODULE_SMU.SAFE[SafeIndex].AGC.B.EFRST1 = 1U;
    }

    CddSmu_TempLockSafe(SafeIndex);
}


void CddSmu_ConfigureRecoveryTimer(uint32_T SafeIndex,
                                   uint8_T Prescaler,
                                   uint32_T Duration,
                                   uint32_T EnableRt0,
                                   uint32_T EnableRt1)
{
    Ifx_SMU_SAFE_RTC rtc;

    rtc.U      = MODULE_SMU.SAFE[SafeIndex].RTC.U;
    rtc.B.PRE  = (uint32_T)Prescaler & 0x7U;
    rtc.B.RTD  = Duration & 0xFFFFFFU;
    rtc.B.RT0E = (CDD_SMU_TRUE == EnableRt0) ? 1U : 0U;
    rtc.B.RT1E = (CDD_SMU_TRUE == EnableRt1) ? 1U : 0U;

    CddSmu_UnlockSafe(SafeIndex);
    MODULE_SMU.SAFE[SafeIndex].RTC.U = rtc.U;
    CddSmu_TempLockSafe(SafeIndex);
}


void CddSmu_BindRecoveryTimerAlarms(uint32_T SafeIndex,
                                    uint32_T TimerNum,
                                    const Cdd_Smu_AlarmRef_T Alarms[4])
{
    uint32_T aca;
    uint32_T acb;
    uint32_T i;

    aca = 0U;
    acb = 0U;

    for (i = 0U; i < 4U; i++)
    {
        uint32_T g;
        uint32_T p;

        if (CDD_SMU_ALARM_SLOT_UNUSED == Alarms[i].Group)
        {
            /* Slot intentionally left empty. */
        }
        else
        {
            g = (uint32_T)Alarms[i].Group;
            p = (uint32_T)Alarms[i].Position;

            if (0U == i)
            {
                aca = aca | (g & 0x1FU) | ((p & 0x1FU) << 5);
            }
            else if (1U == i)
            {
                aca = aca | ((g & 0x1FU) << 16) | ((p & 0x1FU) << 21);
            }
            else if (2U == i)
            {
                acb = acb | (g & 0x1FU) | ((p & 0x1FU) << 5);
            }
            else
            {
                acb = acb | ((g & 0x1FU) << 16) | ((p & 0x1FU) << 21);
            }
        }
    }

    CddSmu_UnlockSafe(SafeIndex);

    if (CDD_SMU_RT_0 == TimerNum)
    {
        MODULE_SMU.SAFE[SafeIndex].RT0ACA.U = aca;
        MODULE_SMU.SAFE[SafeIndex].RT0ACB.U = acb;
    }
    else
    {
        MODULE_SMU.SAFE[SafeIndex].RT1ACA.U = aca;
        MODULE_SMU.SAFE[SafeIndex].RT1ACB.U = acb;
    }

    CddSmu_TempLockSafe(SafeIndex);
}


uint32_T CddSmu_StopRecoveryTimer(uint32_T SafeIndex, uint8_T TimerNum)
{
    Ifx_SMU_SAFE_CMD cmd;
    uint32_T          result;

    cmd.U     = MODULE_SMU.SAFE[SafeIndex].CMD.U;
    cmd.B.CMD = CDD_SMU_CMD_STOP_RT;
    cmd.B.ARG = (uint32_T)TimerNum;
    MODULE_SMU.SAFE[SafeIndex].CMD.U = cmd.U;

    result = (0U == MODULE_SMU.SAFE[SafeIndex].STS.B.RES)
           ? CDD_SMU_TRUE
           : CDD_SMU_FALSE;

    return result;
}


uint32_T CddSmu_GetAlarmStatus(uint8_T AlarmGroup,
                               uint8_T AlarmPos,
                               uint32_T SafeIndex)
{
    uint32_T sts;
    uint32_T result;

    sts = MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].STS.U;

    result = (0U != (sts & ((uint32_T)1UL << AlarmPos)))
           ? CDD_SMU_TRUE
           : CDD_SMU_FALSE;

    return result;
}


void CddSmu_ClearAlarmStatus(uint8_T AlarmGroup,
                             uint8_T AlarmPos,
                             uint32_T SafeIndex)
{
    Ifx_SMU_SAFE_CMD cmd;

    cmd.U     = MODULE_SMU.SAFE[SafeIndex].CMD.U;
    cmd.B.CMD = CDD_SMU_CMD_ALARM_CLR;
    cmd.B.ARG = 0U;
    MODULE_SMU.SAFE[SafeIndex].CMD.U = cmd.U;

    /* W1C: writing 1 clears the corresponding status flag. */
    MODULE_SMU.SAFE[SafeIndex].AGSF[AlarmGroup].STS.U =
        (uint32_T)((uint32_T)1UL << AlarmPos);
}


void CddSmu_ConfigureFsp(uint32_T SafeIndex,
                         uint32_T FspIndex,
                         uint8_T Mode,
                         uint8_T Prescalar1,
                         uint8_T Prescalar2,
                         uint16_T FaultStateDurationHigh,
                         uint16_T FaultStateDurationLow)
{
    Ifx_SMU_SAFE_FSP_CON con;

    con.U           = MODULE_SMU.SAFE[SafeIndex].FSP[FspIndex].CON.U;
    con.B.MODE      = (uint32_T)Mode & 0x3U;
    con.B.PRE1      = (uint32_T)Prescalar1 & 0x7U;
    con.B.PRE2      = (uint32_T)Prescalar2 & 0x7U;
    con.B.TFSP_HIGH = (uint32_T)FaultStateDurationHigh & 0x3FFU;
    con.B.TFSP_LOW  = (uint32_T)FaultStateDurationLow  & 0x3FFFU;

    CddSmu_UnlockSafe(SafeIndex);
    MODULE_SMU.SAFE[SafeIndex].FSP[FspIndex].CON.U = con.U;
    CddSmu_TempLockSafe(SafeIndex);
}


void CddSmu_ConfigurePes(uint32_T SafeIndex, uint32_T PesBits)
{
    CddSmu_UnlockSafe(SafeIndex);
    MODULE_SMU.SAFE[SafeIndex].PES.U |= PesBits;
    CddSmu_TempLockSafe(SafeIndex);
}


void CddSmu_LockSafetyConfig(uint32_T SafeIndex)
{
    MODULE_SMU.SAFE[SafeIndex].KEYS.U = 0U;
}


void CddSmu_PermanentlyLockSafetyConfig(uint32_T SafeIndex)
{
    MODULE_SMU.SAFE[SafeIndex].KEYS.B.PERLCK = CDD_SMU_KEY_PERM_LOCK;
}


uint32_T CddSmu_ForwardCpuWdt(uint8_T Action,
                              uint32_T SafeIndex,
                              uint32_T RtDuration)
{
    uint32_T           coreId;
    uint8_T            grp;
    uint8_T            pos;
    Cdd_Smu_AlarmRef_T bind[4];
    uint32_T           result;

    result = CDD_SMU_FALSE;

    coreId = CddSys_CpuCoreId();

    if (coreId <= 5U)
    {
        grp = CddSmu_CpuWdtGroup[coreId];
        pos = CddSmu_CpuWdtBit  [coreId];

        /* 1. Program the alarm action. */
        CddSmu_SetAlarmAction(grp, pos, Action, SafeIndex);

        /* 2. Bind the alarm to RT0, slot 0. */
        bind[0].Group    = grp;
        bind[0].Position = pos;
        bind[1].Group    = CDD_SMU_ALARM_SLOT_UNUSED;
        bind[1].Position = 0U;
        bind[2].Group    = CDD_SMU_ALARM_SLOT_UNUSED;
        bind[2].Position = 0U;
        bind[3].Group    = CDD_SMU_ALARM_SLOT_UNUSED;
        bind[3].Position = 0U;
        CddSmu_BindRecoveryTimerAlarms(SafeIndex, CDD_SMU_RT_0, bind);

        /* 3. Enable RT0. */
        CddSmu_ConfigureRecoveryTimer(SafeIndex,
                                      0U,
                                      RtDuration,
                                      CDD_SMU_TRUE,
                                      CDD_SMU_FALSE);

        /* 4. Move SSM0 to RUN. */
        result = CddSmu_StartSafetySsm(SafeIndex, CDD_SMU_SSM_0);
    }

    return result;
}

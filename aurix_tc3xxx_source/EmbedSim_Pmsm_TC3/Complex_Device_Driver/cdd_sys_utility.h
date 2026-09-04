/**********************************************************************************************************************
 * \file        cdd_sys_utility.h
 * \brief       System-level utility interfaces for AURIX TC3xx.
 *
 * \details     Provides portable, MISRA C:2012-compliant wrappers for:
 *                - CPU core identification via CORE_ID SFR
 *                - CPU and Safety watchdog EndInit unlock / re-lock sequences (ds1 §P.974)
 *                - Clock-tree frequency interrogation: SCU, PLL0/1, GTM, STM, SPB, ADC, QSPI
 *                - GTM CMU CLK0 run-time frequency configuration
 *                - CPU interrupt enable / disable / save-restore pattern
 *                - Multicore spinlock acquire / release via CMPSWAP.W
 *                - Scalar single-precision near-equality predicate
 *
 * \note        MISRA C:2012 compliance:
 *              Rule  8.5 — One declaration per identifier; matches .c definition exactly.
 *              Rule  8.6 — All object and function definitions reside in cdd_sys_utility.c.
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

/**
 * \brief   Returns the hardware CPU core ID of the currently executing core.
 *
 * \details Reads the TriCore CORE_ID special-function register via MFCR.
 *          Safe to call from any core at any privilege level with no side effects.
 *
 * \return  Core index: 0 = CPU0, 1 = CPU1, …   [dimensionless]
 */
extern uint32_T CddSys_GetCoreId(void);

/**********************************************************************************************************************
 * Function Prototypes — Busy-Wait
 *********************************************************************************************************************/

/**
 * \brief   Software NOP delay loop (busy-wait, no timer dependency).
 *
 * \details Executes OuterLoop × InnerLoop NOP instructions inline.
 *          Exact wall-clock duration is CPU-frequency-dependent and subject to
 *          pipeline effects.  For calibrated delays use the STM module.
 *
 * \param[in]  InnerLoop   NOP count per outer iteration   [dimensionless]
 * \param[in]  OuterLoop   Number of outer iterations      [dimensionless]
 *
 * \return  void
 */
extern void CddSys_NopDelay(uint32_T InnerLoop, uint32_T OuterLoop);

/**********************************************************************************************************************
 * Function Prototypes — Watchdog Password Retrieval
 *********************************************************************************************************************/

/**
 * \brief   Returns the current rolling password from the CPU0 watchdog control register.
 *
 * \details Reads SCU_WDTCPU0CON0.B.PW and XORs bits [7:2] with 0x3F to undo the
 *          hardware inversion applied by TC3xx (ds1 P.975).  The result is the password
 *          value to embed in the subsequent ENDINIT access word.
 *
 * \return  Corrected WDTCPU0CON0.PW field value   [dimensionless]
 */
extern uint32_T CddSys_GetCpuWdt00Pwd(void);

/**
 * \brief   Returns the current rolling password from the CPU1 watchdog control register.
 *
 * \details Identical inversion logic applied to SCU_WDTCPU1CON0.B.PW (ds1 P.975).
 *
 * \return  Corrected WDTCPU1CON0.PW field value   [dimensionless]
 */
extern uint32_T CddSys_GetCpuWdt01Pwd(void);

/**
 * \brief   Returns the current rolling password from the Safety watchdog control register.
 *
 * \details Identical inversion logic applied to SCU_WDTSCON0.B.PW (ds1 P.975).
 *
 * \return  Corrected WDTSCON0.PW field value   [dimensionless]
 */
extern uint32_T CddSys_GetSafetyWdtPwd(void);

/**********************************************************************************************************************
 * Function Prototypes — CPU Watchdog EndInit Control
 *********************************************************************************************************************/

/**
 * \brief   Clears ENDINIT for the CPU0 watchdog, unlocking ENDINIT-protected registers.
 *
 * \details Performs the three-step TriCore ENDINIT clear sequence on SCU_WDTCPU0CON0
 *          (ds1 P.974): password access → unlock → clear ENDINIT + lock.
 *          Polls ENDINIT bit until hardware confirms the cleared state.
 *          The caller must disable interrupts before this call and bracket the
 *          protected register writes with a matching CddSys_SetCpuWdt00EndInit().
 *
 * \pre     SCU_WDTCPU0CON0 accessible; no concurrent access from another core.
 * \post    SCU_WDTCPU0CON0.ENDINIT == 0; ENDINIT-protected SFRs are writable.
 *
 * \return  void
 */
extern void CddSys_ClearCpuWdt00EndInit(void);

/**
 * \brief   Clears ENDINIT for the CPU1 watchdog, unlocking ENDINIT-protected registers.
 *
 * \details Identical sequence to CddSys_ClearCpuWdt00EndInit() applied to
 *          SCU_WDTCPU1CON0 (ds1 P.974).
 *
 * \pre     SCU_WDTCPU1CON0 accessible; no concurrent access from another core.
 * \post    SCU_WDTCPU1CON0.ENDINIT == 0; ENDINIT-protected SFRs are writable.
 *
 * \return  void
 */
extern void CddSys_ClearCpuWdt01EndInit(void);

/**
 * \brief   Clears ENDINIT for the watchdog of the currently executing CPU.
 *
 * \details Dispatches to CddSys_ClearCpuWdt00EndInit() or CddSys_ClearCpuWdt01EndInit()
 *          based on the CORE_ID SFR.  Use this in core-agnostic initialisation paths
 *          where the calling core is not statically known.
 *
 * \return  void
 */
extern void CddSys_ClearCpuWdtEndInit(void);

/**
 * \brief   Sets ENDINIT for the CPU0 watchdog, re-locking ENDINIT-protected registers.
 *
 * \details Performs the ENDINIT set sequence on SCU_WDTCPU0CON0 (ds1 P.974):
 *          password access → unlock → set ENDINIT + lock.
 *          Polls ENDINIT bit until hardware confirms the set state.
 *
 * \pre     CddSys_ClearCpuWdt00EndInit() previously called on this core.
 * \post    SCU_WDTCPU0CON0.ENDINIT == 1; ENDINIT-protected SFRs are write-locked.
 *
 * \return  void
 */
extern void CddSys_SetCpuWdt00EndInit(void);   /* renamed from SetWdt00EndInit — Cpu prefix restored */

/**
 * \brief   Sets ENDINIT for the CPU1 watchdog, re-locking ENDINIT-protected registers.
 *
 * \details Identical sequence to CddSys_SetCpuWdt00EndInit() applied to
 *          SCU_WDTCPU1CON0 (ds1 P.974).
 *
 * \pre     CddSys_ClearCpuWdt01EndInit() previously called on this core.
 * \post    SCU_WDTCPU1CON0.ENDINIT == 1; ENDINIT-protected SFRs are write-locked.
 *
 * \return  void
 */
extern void CddSys_SetCpuWdt01EndInit(void);

/**
 * \brief   Sets ENDINIT for the watchdog of the currently executing CPU.
 *
 * \details Dispatches to CddSys_SetCpuWdt00EndInit() or CddSys_SetCpuWdt01EndInit()
 *          based on the CORE_ID SFR.  Must be paired with CddSys_ClearCpuWdtEndInit().
 *
 * \return  void
 */
extern void CddSys_SetCpuWdtEndInit(void);

/**
 * \brief   Disables the CPU0 watchdog by setting the DR (Disable Request) bit.
 *
 * \details Issues the ENDINIT sequence internally:
 *            CddSys_ClearCpuWdt00EndInit() → SCU_WDTCPU0CON1.DR = 1 → CddSys_SetCpuWdt00EndInit().
 *          DR bit reference: TC3xx Safety Manual §P.980.
 *
 * \post    SCU_WDTCPU0CON1.DR == 1; CPU0 watchdog timer is halted.
 *
 * \return  void
 */
extern void CddSys_DisableCpuWdt00(void);

/**
 * \brief   Disables the CPU1 watchdog by setting the DR (Disable Request) bit.
 *
 * \details Issues the ENDINIT sequence internally:
 *            CddSys_ClearCpuWdt01EndInit() → SCU_WDTCPU1CON1.DR = 1 → CddSys_SetCpuWdt01EndInit().
 *
 * \post    SCU_WDTCPU1CON1.DR == 1; CPU1 watchdog timer is halted.
 *
 * \return  void
 */
extern void CddSys_DisableCpuWdt01(void);

/**********************************************************************************************************************
 * Function Prototypes — Safety Watchdog EndInit Control
 *********************************************************************************************************************/

/**
 * \brief   Clears ENDINIT for the Safety watchdog, unlocking Safety-ENDINIT-protected registers.
 *
 * \details Performs the ENDINIT clear sequence on SCU_WDTSCON0 (ds1 P.974).
 *          Must be paired with CddSys_SetSafetyWdtEndInit() as soon as the
 *          protected configuration is complete.  Holding Safety ENDINIT open
 *          indefinitely will trigger the Safety watchdog.
 *
 * \pre     SCU_WDTSCON0 accessible from the calling core.
 * \post    SCU_WDTSCON0.ENDINIT == 0; Safety-ENDINIT-protected SFRs are writable.
 *
 * \return  void
 */
extern void CddSys_ClearSafetyWdtEndInit(void);

/**
 * \brief   Sets ENDINIT for the Safety watchdog, re-locking Safety-ENDINIT-protected registers.
 *
 * \details Performs the ENDINIT set sequence on SCU_WDTSCON0 (ds1 P.974).
 *          Must always be called after CddSys_ClearSafetyWdtEndInit().
 *
 * \pre     CddSys_ClearSafetyWdtEndInit() previously called on this core.
 * \post    SCU_WDTSCON0.ENDINIT == 1; Safety-ENDINIT-protected SFRs are write-locked.
 *
 * \return  void
 */
extern void CddSys_SetSafetyWdtEndInit(void);

/**
 * \brief   Disables the Safety watchdog by setting the DR bit (SCU_WDTSCON1.DR).
 *
 * \details Issues the Safety ENDINIT sequence internally:
 *            CddSys_ClearSafetyWdtEndInit() → SCU_WDTSCON1.DR = 1 → CddSys_SetSafetyWdtEndInit().
 *          DR bit reference: TC3xx Safety Manual §P.977.
 *
 * \post    SCU_WDTSCON1.DR == 1; Safety watchdog timer is halted.
 *
 * \return  void
 */
extern void CddSys_DisableSafetyWdt(void);

/**********************************************************************************************************************
 * Function Prototypes — Clock Tree Frequency Interrogation
 *********************************************************************************************************************/

/**
 * \brief   Returns the EVR (internal 100 MHz backup oscillator) frequency.
 *
 * \details Returns the compile-time constant EVR_OSC_FREQUENCY.
 *          The EVR oscillator is the PLL input when SCU_SYSPLLCON0.INSEL == 0.
 *
 * \return  EVR oscillator frequency   [Hz]
 */
extern real64_T CddSys_GetEvrFreq(void);

/**
 * \brief   Returns the system clock oscillator frequency as configured in SCU_OSCCON.
 *
 * \details Returns the compile-time constant SYSCLK_OSC_FREQUENCY.
 *
 * \return  System oscillator frequency   [Hz]
 */
extern real64_T CddSys_GetSysClkFreq(void);

/**
 * \brief   Returns the external crystal oscillator frequency.
 *
 * \details Returns the compile-time constant XTAL_OSC_FREQUENCY.
 *          The external crystal is the PLL input when SCU_SYSPLLCON0.INSEL == 1.
 *
 * \return  External oscillator frequency   [Hz]
 */
extern real64_T CddSys_GetExtOscFreq(void);

/**
 * \brief   Returns the System PLL input (primary oscillator) frequency.
 *
 * \details Reads SCU_SYSPLLCON0.INSEL:
 *            0 → EVR backup oscillator (CddSys_GetEvrFreq)
 *            1 → External crystal      (CddSys_GetExtOscFreq)
 *            2 → System clock OSC      (CddSys_GetSysClkFreq)
 *
 * \return  PLL input frequency   [Hz]
 */
extern real64_T CddSys_GetPrimaryOscFreq(void);

/**
 * \brief   Returns the System PLL0 (fPLL0) output frequency.
 *
 * \details Computes the integer PLL formula from ds1 §P.937:
 *              fPLL0 = (fOSC × (NDIV + 1)) / ((PDIV + 1) × (K2DIV + 1))
 *          Returns 0.0 if either denominator divider is zero (PLL not locked / misconfigured).
 *
 * \return  PLL0 output frequency   [Hz]
 */
extern real64_T CddSys_GetPll00Freq(void);

/**
 * \brief   Returns the PerPLL K2-path output (fSOURCE1) after the PLL1DIVDIS post-divider.
 *
 * \details Peripheral PLL K2 path formula (ds1 §P.938):
 *              fPerPLL_K2 = (fOSC × (PERPLL_NDIV+1)) / ((PERPLL_PDIV+1) × (PERPLL_K2DIV+1))
 *          SCU_CCUCON1.PLL1DIVDIS == 0: ÷2 post-divider is active → fSOURCE1 = fPerPLL_K2 / 2.
 *          PLL1DIVDIS == 1: ÷2 bypassed → fSOURCE1 = fPerPLL_K2.
 *          Uses SCU_PERPLLCON0/1 registers — independent of the System PLL (fPLL0).
 *          fSOURCE1 feeds fADC and fQSPI (when SCU_CCUCON1.CLKSELQSPI == 0x1).
 *
 * \return  fSOURCE1 (PerPLL K2) frequency   [Hz]
 */
extern real64_T CddSys_GetPll01Freq(void);

/**
 * \brief   Returns the clock source 0 (fSOURCE0) frequency after the SCU_CCUCON0 mux.
 *
 * \details SCU_CCUCON0.CLKSEL selects either fPLL0 (0x1) or EVR (default).
 *          fSOURCE0 feeds fSRI, fSTM, fSPB and the GTM source domain.
 *
 * \return  fSOURCE0 frequency   [Hz]
 */
extern real64_T CddSys_GetSrc00Freq(void);

/**
 * \brief   Returns the clock source 1 (fSOURCE1) frequency after the SCU_CCUCON0 mux.
 *
 * \details SCU_CCUCON0.CLKSEL selects either fPLL1 (0x1) or EVR (default).
 *          fSOURCE1 feeds fADC and fQSPI domains.  In low-power mode (LPDIV >= 2),
 *          an additional ÷2 factor is applied.
 *
 * \return  fSOURCE1 frequency   [Hz]
 */
extern real64_T CddSys_GetSrc01Freq(void);

/**
 * \brief   Returns the SRI (System Resource Interconnect) bus frequency.
 *
 * \details Derived from fSOURCE0 via SCU_CCUCON0.SRIDIV (normal mode) or
 *          the LPDIV low-power prescaler table (LPDIV 1–4 → ÷30 … ÷240).
 *          Returns 0.0 if SRIDIV == 0 (clock off) or LPDIV is out of range.
 *
 * \return  fSRI frequency   [Hz]
 */
extern real64_T CddSys_GetSriFreq(void);

/**
 * \brief   Returns the CPU0 core clock frequency.
 *
 * \details Applies SCU_CCUCON6.CPU0DIV fractional divider to fSRI:
 *              fCPU0 = fSRI × (64 − CPU0DIV) / 64
 *          Returns fSRI unchanged when CPU0DIV == 0 (÷1 pass-through).
 *
 * \return  fCPU0 frequency   [Hz]
 */
extern real64_T CddSys_GetCpuFreq(void);

/**
 * \brief   Returns the STM (System Timer Module) clock frequency.
 *
 * \details fSTM = fSOURCE0 / STMDIV   (SCU_CCUCON0.STMDIV).
 *          Returns 0.0 if STMDIV == 0 (STM clock gated off).
 *
 * \return  fSTM frequency   [Hz]
 */
extern real64_T CddSys_GetStmFreq(void);

/**
 * \brief   Returns the SPB (System Peripheral Bus) clock frequency.
 *
 * \details Normal mode: fSPB = fSOURCE0 / SPBDIV (SPBDIV must be >= 2).
 *          Low-power mode (LPDIV 1–4): fSPB = fSOURCE0 / (30 × 2^(LPDIV-1)).
 *          SPBDIV < 2 or unknown LPDIV returns 0.0 (clock not valid).
 *          Governs ASCLIN, I²C, and all other SPB-clocked peripherals.
 *
 * \return  fSPB frequency   [Hz]
 */
extern real64_T CddSys_GetSpbFreq(void);


/**
 * \brief   Returns the Gpt12 clock frequency.
 *
 * \details
 *
 * \return  gpt12 frequency   [Hz]
 */
extern real64_T CddSys_GetGpt12Freq(void);


/**
 * @brief   Calculate the current input clock frequency of timer T5 in the GPT12 module.
 *
 * @details Reads the T5CON, T6CON registers directly via the provided macros.
 * @return  T5 timer input frequency in Hz as real64_T.
 */
extern real64_T CddSys_GetGpt12_T5Freq(void);


/**
 * \brief   Returns the GTM clock source frequency (fGTMSRC) before the GTM module divider.
 *
 * \details SCU_CCUCON0.GTMDIV decoding:
 *            1 → fGTMSRC = 2 × fSPB  (dedicated 2×SPB bypass mode)
 *            _ → fGTMSRC = fSOURCE0  (all other divider values use fSOURCE0)
 *          Note: GTMDIV == 0 (GTM off) is not filtered here; fSOURCE0 is returned
 *          as the dormant source.  CddSys_GetGtmFreq() guards the zero-divide case.
 *
 * \return  GTM source frequency   [Hz]
 */
extern real64_T CddSys_GetGtmSrcFreq(void);

/**
 * \brief   Returns the GTM module input frequency (fGTM) after the CCU divider.
 *
 * \details fGTM = fGTMSRC / GTMDIV   (SCU_CCUCON0.GTMDIV).
 *          Returns 0.0 if GTMDIV == 0 (GTM clock gated off).
 *
 * \return  GTM module frequency   [Hz]
 */
extern real64_T CddSys_GetGtmFreq(void);

/**
 * \brief   Returns the GTM CMU cluster 0 (CLS0) fixed-divider output frequency.
 *
 * \details fCLS0 = fGTM / CLS0_CLK_DIV   (GTM_CLS_CLK_CFG.CLS0_CLK_DIV).
 *          CLS0 is the reference from which CMU CLK0–CLK7 are sub-divided.
 *          Returns 0.0 if CLS0_CLK_DIV == 0.
 *
 * \return  GTM CMU cluster 0 frequency   [Hz]
 */
extern real64_T CddSys_GetGtmClusterFreq(void);

/**
 * \brief   Returns the GTM CMU global clock (GCLK) frequency.
 *
 * \details fGCLK = (GCLK_DEN / GCLK_NUM) × fCLS0
 *          (GTM_CMU_GCLK_NUM / GTM_CMU_GCLK_DEN, ds2 §P.188).
 *          GCLK drives TBU time-bases TB0/TB1/TB2.
 *          Returns 0.0 if GCLK_NUM == 0 (divider not configured).
 *
 * \return  GTM CMU global clock frequency   [Hz]
 */
extern real64_T CddSys_GetGtmCmuGlobalFreq(void);

/**
 * \brief   Returns the EVADC (Versatile ADC) module clock frequency.
 *
 * \details fADC == fSOURCE1 on TC3xx; delegates to CddSys_GetSrc01Freq().
 *
 * \return  fADC frequency   [Hz]
 */
extern real64_T CddSys_GetAdcFreq(void);

/**
 * \brief   Returns the PerPLL K3-path output (fSOURCE2) after the DIVBY fractional factor.
 *
 * \details PerPLL K3 path formula (ds1 §P.938):
 *              fPerPLL_K3 = (fOSC × (NDIV+1)) / ((PDIV+1) × (K3DIV+1) × factor)
 *          SCU_PERPLLCON0.DIVBY: 0 → factor=1.6 (fractional), 1 → factor=2.0 (integer).
 *          fSOURCE2 feeds fQSPI when SCU_CCUCON1.CLKSELQSPI == 0x2.
 *
 * \return  fSOURCE2 (PerPLL K3) frequency   [Hz]
 */
extern real64_T CddSys_GetPerPllK3Freq(void);

/**
 * \brief   Returns the QSPI peripheral clock frequency.
 *
 * \details QSPI source is from the Peripheral PLL domain (ds1 §P.962).
 *          SCU_CCUCON1.CLKSELQSPI selects:
 *            0x1 → fSOURCE1 = PerPLL K2 path (CddSys_GetPll01Freq)
 *            0x2 → fSOURCE2 = PerPLL K3 path (CddSys_GetPerPllK3Freq)
 *            other → 0 Hz (clock off).
 *          Then divided by SCU_CCUCON1.QSPIDIV.  Returns 0.0 if QSPIDIV == 0.
 *
 * \return  fQSPI frequency   [Hz]
 */
extern real64_T CddSys_GetQspiFreq(void);

/**********************************************************************************************************************
 * Function Prototypes — GTM CMU CLK0 Configuration
 *********************************************************************************************************************/

/**
 * \brief   Configures GTM CMU CLK0 to the nearest achievable frequency.
 *
 * \details Computation (ds2 §P.186):
 *              CLK_CNT = round(fGCLK / CmuClk00Freq) − 1
 *          Write sequence (CMU CLK registers require EN_CLK0 == 0 during write):
 *            1. ClearCpuWdtEndInit
 *            2. GTM_CMU_CLK_EN.EN_CLK0 = 0x1U  (disable)
 *            3. GTM_CMU_CLK_0_CTRL.CLK_CNT = clk_cnt
 *            4. SetCpuWdtEndInit
 *            5. GTM_CMU_CLK_EN.EN_CLK0 = 0x2U  (re-enable)
 *          The caller must ensure no ATOM/TOM channel actively consumes CLK0
 *          during this call, or mask those channels first.
 *
 * \pre     GTM module clock (fGTM) running; fGCLK > 0.
 * \post    CMU CLK0 running at the nearest achievable frequency to CmuClk00Freq.
 *
 * \param[in]  CmuClk00Freq   Target CLK0 frequency   [Hz]   (must be > 0.0)
 *
 * \return  void
 */
extern void CddSys_SetGtmCmuClk00Freq(real64_T CmuClk00Freq);

/**
 * \brief   Returns the currently configured GTM CMU CLK0 frequency.
 *
 * \details Back-computes from GTM_CMU_CLK_0_CTRL.CLK_CNT (ds2 §P.186):
 *              fCLK0 = fGCLK / (CLK_CNT + 1)
 *          Returns 0.0 if CLK0 is not enabled (GTM_CMU_CLK_EN.EN_CLK0 != 0x3).
 *
 * \return  Current CMU CLK0 frequency   [Hz]
 */
extern real64_T CddSys_GetGtmCmuClk00Freq(void);

/**********************************************************************************************************************
 * Function Prototypes — CPU Interrupt Control
 *********************************************************************************************************************/

/**
 * \brief   Returns 1 if CPU global interrupts are currently enabled, 0 otherwise.
 *
 * \details Reads ICR.IE (Interrupt Control Register — Interrupt Enable bit) of the
 *          currently executing core via MFCR(CPU_ICR).
 *
 * \return  1 = interrupts enabled, 0 = interrupts disabled   [dimensionless]
 */
extern uint32_T CddSys_IsIrqEnabled(void);

/**
 * \brief   Disables CPU global interrupts and returns the prior enable state.
 *
 * \details Issues a DISABLE instruction on TriCore.  The returned state must be
 *          passed verbatim to CddSys_RestoreIrq() to restore the original ICR.IE
 *          without unconditionally re-enabling interrupts.  This save-restore
 *          pattern is safe for nested critical sections.
 *
 * \post    ICR.IE == 0 on the calling core.
 *
 * \return  Prior interrupt-enable state: 1 = was enabled, 0 = was already disabled
 */
extern uint32_T CddSys_DisableIrq(void);

/**
 * \brief   Restores the CPU interrupt enable state saved by CddSys_DisableIrq().
 *
 * \details If PrevState == 1 issues ENABLE; otherwise leaves ICR.IE == 0.
 *          Permits correct nesting: an outer critical section that was already
 *          in disabled state is not accidentally re-enabled by an inner section.
 *
 * \param[in]  PrevState   Value returned by a prior call to CddSys_DisableIrq()
 *
 * \return  void
 */
extern void CddSys_RestoreIrq(uint32_T PrevState);

/**********************************************************************************************************************
 * Function Prototypes — Spinlock
 *********************************************************************************************************************/

/**
 * \brief   Attempts a single non-blocking acquire of a multicore spinlock via CMPSWAP.W.
 *
 * \details Atomically tests *LockPtr == 0 (free) and, if free, writes 1 (held).
 *          The CMPSWAP.W instruction is atomic across all TriCore cores sharing the
 *          same LMU / DLMU bus segment.
 *          The caller must ensure LockPtr points to a shared, cache-coherent, 4-byte-aligned
 *          memory location (e.g. placed in .lmudata section).
 *          This is a try-once call; spin-looping is the caller's responsibility.
 *
 * \param[in,out]  LockPtr   Pointer to the lock word (0 = free, 1 = held).
 *                            Must be 4-byte aligned.
 *
 * \return  1 = lock acquired,   0 = lock already held by another core   [dimensionless]
 */
extern uint32_T CddSys_AcquireSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr);

/**
 * \brief   Releases a previously acquired multicore spinlock.
 *
 * \details Writes 0 to *LockPtr, making the lock available to other cores.
 *          Calling this without a prior successful CddSys_AcquireSpinLock() on the
 *          same core produces undefined behaviour — no ownership check is performed.
 *
 * \pre     CddSys_AcquireSpinLock(LockPtr) previously returned 1 on this core.
 * \post    *LockPtr == 0; lock is free for acquisition by any core.
 *
 * \param[in,out]  LockPtr   Pointer to the lock word (same pointer passed to Acquire).
 *
 * \return  void
 */
extern void CddSys_ReleaseSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr);

/**
 * \brief   Tests whether two single-precision floats are equal within an absolute tolerance.
 *
 * \details Evaluates: |Lhs − Rhs| ≤ Epsilon.
 *          No NaN or infinity handling is performed; inputs must be finite.
 *          Intended for CDD post-condition checks and self-test assertions, not
 *          for floating-point comparison inside control-loop paths.
 *
 * \param[in]  Lhs      First operand                                     [application units]
 * \param[in]  Rhs      Second operand                                    [application units]
 * \param[in]  Epsilon  Maximum permitted absolute difference (>= 0.0f)   [application units]
 *
 * \return  1 = |Lhs − Rhs| ≤ Epsilon,   0 = values differ beyond tolerance   [dimensionless]
 */
extern uint32_T CddSys_AreEqual(real32_T Lhs, real32_T Rhs, real32_T Epsilon);

/**
 * \brief   Tests whether two single-precision floats are equal within an absolute tolerance.
 *
 * \details Evaluates: |Lhs − Rhs| ≤ Epsilon.
 *          Explicit 32-bit variant; prefer this over CddSys_AreEqual for new call sites.
 *          No NaN or infinity handling is performed; inputs must be finite.
 *
 * \param[in]  Lhs      First operand                                     [application units]
 * \param[in]  Rhs      Second operand                                    [application units]
 * \param[in]  Epsilon  Maximum permitted absolute difference (>= 0.0f)   [application units]
 *
 * \return  1 = |Lhs − Rhs| ≤ Epsilon,   0 = values differ beyond tolerance   [dimensionless]
 */
extern uint32_T CddSys_AreEqual32(real32_T Lhs, real32_T Rhs, real32_T Epsilon);

/**
 * \brief   Tests whether two double-precision floats are equal within an absolute tolerance.
 *
 * \details Evaluates: |Lhs − Rhs| ≤ Epsilon.
 *          64-bit variant for clock-frequency and high-resolution comparisons.
 *          No NaN or infinity handling is performed; inputs must be finite.
 *
 * \param[in]  Lhs      First operand                                      [application units]
 * \param[in]  Rhs      Second operand                                     [application units]
 * \param[in]  Epsilon  Maximum permitted absolute difference (>= 0.0)    [application units]
 *
 * \return  1 = |Lhs − Rhs| ≤ Epsilon,   0 = values differ beyond tolerance   [dimensionless]
 */
extern uint32_T CddSys_AreEqual64(real64_T Lhs, real64_T Rhs, real64_T Epsilon);

#endif /* CDD_SYS_UTILITY_H_ */

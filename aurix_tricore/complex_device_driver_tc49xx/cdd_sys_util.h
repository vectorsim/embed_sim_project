/**********************************************************************************************************************
 * \file      cdd_sys_util.h
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
 * \date      2026-09-25
 * \author   Paul Abraham
 *             EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_SYS_UTIL_H_
#define CDD_SYS_UTIL_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
/**********************************************************************************************************************
 * Clock Frequencies [Hz]
 *********************************************************************************************************************/

/**
 * @brief Backup (EVR) oscillator frequency in Hz.
 * @note TC4x default: 100 MHz.
 */
#define CDD_SYS_EVR_OSC_FREQUENCY    MHZ_100

/** @brief External crystal oscillator frequency in Hz. */
#define CDD_XTAL_FREQUENCY           MHZ_25

/** @brief Configured oscillator F frequency in Hz. */
#define CDD_CFG_OSC_F_FREQ           MHZ_20


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*------------------------------------------------ Inline Functions - -----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * @brief Reads a TriCore Core Special Function Register (CSFR).
 *
 * Uses the TriCore MFCR (Move From Core Register) instruction to read
 * the specified CPU core register.
 *
 * @param[in] RegAddr
 *     Compile-time address of the Core Special Function Register.
 *
 * @return
 *     32-bit value read from the specified core register.
 *
 * @note
 *     RegAddr must be a compile-time constant because the inline assembly
 *     uses the "i" immediate operand constraint.
 *
 * @note
 *     The function is always inlined and therefore introduces no
 *     function-call overhead.
 */
static __inline__ __attribute__((__always_inline__))
uint32_T CddSys_Mfcr(const uint32_T RegAddr)
{
    uint32_T res;

    __asm__ volatile (
        "mfcr %0, LO:%1"
        : "=d" (res)
        : "i" (RegAddr)
        : "memory"
    );

    return res;
}

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Atomic bit-field load-modify-store (TriCore LDMST instruction).
 *
 * \details Inserts the LDMST instruction with the given mask and value. The
 *          instruction reads the target word, replaces only the bits selected
 *          by Mask with the corresponding bits of Value, and writes the result
 *          back atomically with respect to interrupts on the same core.
 *
 *          All operands must be word-aligned. The 64-bit operand encoding
 *          requires the mask in the upper 32 bits and the value in the lower
 *          32 bits, which is why the two are packed before the instruction
 *          is emitted.
 *
 * \param[in]  AddressPtr  Word-aligned address of the target register.
 *                         Must not be NULL and must be 4-byte aligned.
 * \param[in]  Mask        Bit mask selecting the bits to write. Bits set to
 *                         1 in Mask are replaced; bits set to 0 are preserved.
 * \param[in]  Value       Value whose bits are written into the masked field.
 *                         Bits of Value outside Mask are ignored.
 *
 * \return  void
 *
 * \note    Typical usage:
 *            CddSys_Ldmst((volatile uint32_T *)&MODULE_P00->IOCR0,
 *                         0x0000FF00U, 0x00001200U);
 *
 * \note    The instruction is atomic on the same core but does not protect
 *          against accesses from other cores. For cross-core mutual exclusion
 *          use CddSys_CmpAndSwap.
 *
 * \see     CddSys_CmpAndSwap
 */
extern void CddSys_Ldmst(P2VAR(volatile uint32_T, AUTOMATIC, CDD_APPL_DATA) AddressPtr,
                         uint32_T Mask,
                         uint32_T Value);

/**
 * \brief   Insert a single NOP instruction.
 *
 * \details Emits one TriCore NOP. Used for short, deterministic pipeline
 *          alignment or as a placeholder when a delay of exactly one
 *          instruction slot is required. A memory clobber is attached so
 *          the compiler does not reorder memory accesses across the NOP.
 *
 * \return  void
 */
extern void CddSys_Nop(void);


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

/**
 * \brief   Binary semaphore acquire/release using CMPSWAP.W instruction.
 *
 * \details Implements the standard atomic compare-and-swap pattern using the
 *          TriCore CMPSWAP.W instruction. The function reads the lock word,
 *          compares it against Condition, and - only if they match - writes
 *          Value into the lock word. It always returns the value the lock
 *          word held before the operation.
 *
 *          A binary semaphore is built on top of this as follows:
 *            acquire: if (CddSys_CmpAndSwap(&Lock, 1U, 0U) == 0U) { ... }
 *            release: (void)CddSys_CmpAndSwap(&Lock, 0U, 1U);
 *
 * \param[in,out] AddressPtr  Word-aligned pointer to the lock word.
 *                            Must not be NULL and must be 4-byte aligned.
 *                            On success the lock word is updated in place.
 * \param[in]     Value       New value to install if the compare succeeds.
 * \param[in]     Condition   Value that *AddressPtr must equal for the swap
 *                            to occur.
 *
 * \return  The value *AddressPtr held before the operation:
 *            - 0 if the acquire succeeded (or if the lock was already 0),
 *            - non-zero if the compare failed and no write took place.
 */
extern uint32_T CddSys_CmpAndSwap(P2VAR(volatile uint32_T, AUTOMATIC, CDD_APPL_DATA) AddressPtr, uint32_T Value, uint32_T Condition);

/**
 * \brief   Enable interrupts immediately at function entry.
 *
 * \details Emits the TriCore `enable` instruction, which clears the interrupt
 *          disable bit in the program status word. The effect is immediate:
 *          pending interrupts are serviced before the next instruction of
 *          the caller executes. A memory clobber is attached so the compiler
 *          does not move memory accesses across the enable.
 *
 *          This helper is intended for use in the entry sequence of an
 *          initialisation function or an ISR that was entered with
 *          interrupts globally disabled and needs them re-enabled early.
 *
 * \return  void
 *
 */
extern void CddSys_IsrEnable(void);

/**
 * \brief   Disables interrupts immediately at function entry.
 *
 * \details Emits the TriCore `enable` instruction, which clears the interrupt
 *          disable bit in the program status word. The effect is immediate:
 *          pending interrupts are serviced before the next instruction of
 *          the caller executes. A memory clobber is attached so the compiler
 *          does not move memory accesses across the enable.
 *
 *          This helper is intended for use in the entry sequence of an
 *          initialisation function or an ISR that was entered with
 *          interrupts globally disabled and needs them re-enabled early.
 *
 * \return  void
 */
extern void CddSys_IsrDisable(void);

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

/**
 * @brief Returns the index of the currently executing CPU core.
 *
 * Reads the TriCore CPU_CORE_ID Core Special Function Register (CSFR)
 * using CddSys_Mfcr() and extracts the CORE_ID field.
 *
 * @return
 *     Zero-based index identifying the CPU core on which the calling
 *     code is currently executing.
 *
 * @note
 *     This function reads the CPU core's own CPU_CORE_ID register and
 *     therefore does not require any input parameter.
 *
 * @note
 *     The returned value corresponds to the hardware core ID assigned
 *     by the AURIX TriCore CPU.
 */
extern uint32_T CddSys_CpuCoreId(void);


/**
 * @brief  Gets the CPU Watchdog Timer password for the current CPU core.
 *
 * The function accesses the CPU Watchdog Timer Control Register A
 * (CTRLA) of the currently executing CPU core and reads the PW field.
 * The password is transformed by XORing it with 0x007F.
 *
 * @return uint16_T
 *         Transformed CPU Watchdog Timer password.
 */
extern uint16_T CddSys_GetCpuWdtPwd(void);


/**
 * @brief  Disables the CPU Watchdog Timer for the current CPU core.
 *
 * The function accesses the CPU Watchdog Timer corresponding to the
 * currently executing CPU core. If the Watchdog Timer Control Register A
 * (CTRLA) is locked, the CPU Watchdog password is retrieved and used to
 * unlock the register.
 *
 * The Disable Request (DR) bit of the Watchdog Timer Control Register B
 * (CTRLB) is then set to disable the CPU Watchdog Timer. Finally, the
 * LCK bit of CTRLA is set again to protect the register.
 *
 */
extern void CddSys_DisableCpuWdt(void);


/**
 * @brief  Enables the CPU Watchdog Timer for the current CPU core.
 *
 * The function accesses the CPU Watchdog Timer corresponding to the
 * currently executing CPU core. If the Watchdog Timer Control Register A
 * (CTRLA) is locked, the CPU Watchdog password is retrieved and used to
 * unlock the register.
 *
 */
extern void CddSys_EnableCpuWdt(void);


/**
 * @brief  Services (reloads) the CPU Watchdog Timer for the current CPU core.
 *
 * The function accesses the CPU Watchdog Timer corresponding to the
 * currently executing CPU core. If the Watchdog Timer Control Register A
 * (CTRLA) is locked, the CPU Watchdog password is retrieved and used to
 * unlock the register.
 *
 * If the timer is counting (TC bit in STAT set), the inverted Timeout
 * Interval (TIM) field of the STAT register is written to the TCVI field
 * of CTRLA, reloading the watchdog counter. The LCK bit of CTRLA is then
 * set again to protect the register.

 */
extern void CddSys_ServiceCpuWdt(void);

/**
 * @brief  Gets the System Watchdog Timer password.
 *
 * Reads the password field (PW) from the System Watchdog Timer
 * Control Register A (CTRLA).
 *
 * The lower seven bits of the password are inverted when read from
 * the register. These bits are therefore toggled before returning
 * the password.
 *
 * @return uint16_T
 *         System Watchdog Timer password.
 */
extern uint16_T CddSys_GetSystemWdtPwd(void);


/**
* @brief  Disables the System Watchdog Timer.
*
* The function reads the current System Watchdog Timer configuration
* and, if the Control Register A (CTRLA) is locked, uses the
* System Watchdog Timer password to unlock the register.
*
* The Disable Request (DR) bit of Control Register B (CTRLB) is then
* set to disable the System Watchdog Timer. Finally, the LCK bit of
* CTRLA is set again to protect the register.
*
* @note   The System Watchdog Timer password is obtained using
*         CddSys_GetSystemWdtPwd().
*/
extern void CddSys_DisableSystemWdt(void);

/**
 * @brief Gets the EVR backup clock frequency.
 *
 * Returns the configured frequency of the EVR backup oscillator.
 *
 * @return EVR backup clock frequency in Hz.
 */
extern real32_T CddSys_GetEvrFrequency(void);

/**
 * @brief Get the selected oscillator input frequency.
 *
 * Reads the oscillator input selection from CLOCK_OSCCON.B.INSEL
 * and returns the corresponding clock frequency in Hz.
 *
 * @return Oscillator input frequency in Hz.
 */
extern real32_T CddSys_GetOscFrequency(void);

/**
 * @brief Get the system PLL output frequency.
 *
 * Calculates the PLL output frequency from the selected oscillator
 * frequency and the SYSPLL configuration registers.
 *
 * The calculated frequency is additionally gated by the PLL lock
 * and power status bits.
 *
 * @return PLL output frequency in Hz.
 */
extern real32_T CddSys_GetPllFrequency(void);

/**
 * @brief Gets the current ramp frequency.
 *
 * Determines the ramp frequency based on the ramp sequence status
 * and frequency status registers.
 *
 * @return Current ramp frequency in Hz.
 *         - At top: UFL multiplied by 1,000,000.
 *         - At base: 100,000,000 Hz.
 *         - Otherwise: 0 Hz.
 *
 * @note The frequency is returned as zero if the ramp sequence is
 *       not idle or the frequency is between the base and top values.
 *
 * @note The function follows a single-entry, single-exit structure.
 */
extern real32_T CddSys_GetRampFrequency(void);

/**
 * @brief Gets the system clock source frequency.
 *
 * Determines the active system clock source from the CLKSELS field
 * of the CCUSTAT register and retrieves its frequency.
 *
 * @return System clock source frequency in Hz.
 *         - 0x0U: PLL frequency.
 *         - 0x1U: EVR backup frequency (fBACK).
 *         - 0x2U: Ramp frequency.
 *         - Other values: 0 Hz.
 */
extern real32_T CddSys_GetSysSourceFrequency(void);

/**
 * @brief Gets the eGTM clock frequency.
 *
 * Calculates the eGTM clock frequency based on the system clock
 * source frequency and the configured eGTM and low-power dividers.
 *
 * If LPDIV is zero, the system clock source frequency is divided
 * by the configured eGTM divider value. If EGTMDIV is zero,
 * the eGTM frequency is returned as zero.
 *
 * If LPDIV is non-zero, the system clock source frequency is
 * divided by 120.
 *
 * @return eGTM clock frequency in Hz.
 *
 * @note EGTMDIV is used as an index into the clock divider lookup
 *       table. The register value must be within the valid range
 *       of 0 to 15.
 */
extern real32_T CddSys_GetEgtmFrequency(void);


/**
 * @brief Gets the eGTM Cluster 0 clock frequency.
 *
 * Reads the Cluster 0 clock divider configuration and calculates
 * the cluster clock frequency from the eGTM clock frequency.
 *
 * If the cluster divider is zero, Cluster 0 is considered disabled
 * and the function returns zero.
 *
 * @return Cluster 0 clock frequency in Hz, or 0.0F if the cluster
 *         is disabled.
 */
extern real32_T CddSys_GetEgtmCls0Frequency(void);

/**
 * @brief Gets the eGTM CMU clock frequency.
 *
 * Calculates the CMU clock frequency for eGTM Cluster 0 using
 * the cluster clock frequency and the configured GCLK numerator
 * and denominator values.
 *
 * The frequency is calculated as:
 *
 * cmuFrequency = (GCLK_DEN / GCLK_NUM) * moduleFrequency
 *
 * @return CMU clock frequency in Hz.
 *
 */
extern real32_T CddSys_GetEgtmCmuFrequency(void);

/**
 * \brief   Configures the eGTM Cluster 0 CMU GCLK frequency.
 *
 * \details Searches for the (numerator, denominator) pair that produces
 *          a CMU clock frequency closest to the requested value, using
 *          the current eGTM Cluster 0 module frequency as the reference.
 *
 *          The search iterates the denominator from 1 up to 0xFFFFFE and,
 *          for each denominator, iterates the numerator from the current
 *          denominator down to 1. The pair with the smallest absolute
 *          frequency error is programmed into the CMU GCLK_NUM and
 *          GCLK_DEN registers. The search terminates early once the
 *          frequency error falls below 0.1 Hz.
 *
 *          The resulting CMU frequency follows the same relation used
 *          by CddSys_GetEgtmCmuFrequency():
 *
 *            cmuFrequency = (GCLK_DEN / GCLK_NUM) * moduleFrequency
 *
 * \param[in]  Frequency   Desired CMU clock frequency in Hz.
 *
 * \return  void
 */
extern void CddSys_SetEgtmCmuFrequency(real32_T Frequency);


/**
 * @brief   Sets the EGTM Cluster 0 CMU clock 0 to the requested frequency.
 *
 * @details Computes the counter reload value as round((F_source / Frequency) - 1)
 *          and writes it to EGTM_CLS0_CMU_CLK0_CTRL.B.CLK_CNT.
 *
 * @param[in] Frequency Desired output frequency in Hz.
 *                      Must be > 0 and < CddSys_GetEgtmCmuFrequency().
 *
 * @return  None.
 */
extern void CddSys_SetEgtmCmu0Frequency(real32_T Frequency);

/**
 * \brief   Compare two real32_T values for approximate equality.
 *
 * \details Returns TRUE when |Lhs - Rhs| <= Epsilon, FALSE otherwise.
 *          Avoids fabsf() to keep the module math-library independent.
 *
 * \param[in]  Lhs       Left-hand side value.                  [dimensionless]
 * \param[in]  Rhs       Right-hand side value.                 [dimensionless]
 * \param[in]  Epsilon   Maximum allowed absolute difference.
 *                       Must be >= 0.0F.                       [dimensionless]
 *
 * \return  TRUE (1U) if |Lhs - Rhs| <= Epsilon, FALSE (0U) otherwise.
 */
extern uint32_T CddSys_AreEqual32(real32_T Lhs, real32_T Rhs, real32_T Epsilon);

/**
 * \brief   Gets the current eGTM Cluster 0 CMU clock 0 output frequency.
 *
 * \details Returns 0.0F if CMU clock 0 is disabled (EN_CLK0 != 0x3U).
 *          Otherwise computes CddSys_GetEgtmCmuFrequency() / (CLK_CNT + 1).
 *
 * \return  CMU clock 0 output frequency in Hz, or 0.0F if disabled.
 */
extern real32_T CddSys_GetEgtmCmu0Frequency(void);


/**
 * \brief   Enables eGTM Cluster 0 CMU clock 0.
 *
 * \details Sets the EN_CLK0 field of EGTM_CLS0_CMU_CLK_EN to 0x2U, which
 *          enables CMU clock 0. Once enabled, the CMU global frequency is
 *          divided by (CLK_CNT + 1) to produce the CMU clock 0 output, as
 *          reported by CddSys_GetEgtmCmu0Frequency().
 *
 *          Typical sequence:
 *            1. CddSys_EnableEgtmClock()          - release module and
 *                                                   cluster protection,
 *            2. CddSys_SetEgtmCmuFrequency()      - configure GCLK_NUM /
 *                                                   GCLK_DEN,
 *            3. CddSys_EnableEgtmCmu0Clock()      - enable clock 0 output,
 *            4. CddSys_SetEgtmCmu0Frequency()     - set CLK_CNT divider.
 *
 * \return  void
 *
 * \note    The eGTM module clock and the Cluster 0 protection must already
 *          be released (see CddSys_EnableEgtmClock) before the CMU
 *          registers are writable.
 *
 * \see     CddSys_EnableEgtmClock, CddSys_SetEgtmCmuFrequency,
 *          CddSys_SetEgtmCmu0Frequency, CddSys_GetEgtmCmu0Frequency
 */
extern void CddSys_EnableEgtmCmu0Clock(void);

/**
 * \brief   Enables eGTM Cluster 0 CMU clock 0.
 *
 *
 * \return  void
 *
 */
extern void CddSys_EnableEgtmCmu0Clock(void);


/**
 * \brief   Releases eGTM module and Cluster 0 protection and enables the
 *          eGTM clock.
 *
 * \return  void
 */
extern void CddSys_EnableEgtmClock(void);

/**
 * @brief Gets the SRI bus clock frequency.
 *
 * Reads the SRIDIV field of CLOCK_SYSCCUCON0 and divides the current
 * system source frequency accordingly. If the low-power divider is
 * active (LPDIV != 0), the source is divided by 120.
 *
 * @return SRI clock frequency in Hz, or 0.0F if the source is not valid.
 */
extern real32_T CddSys_GetSriFrequency(void);

/**
 * @brief Gets the CPU clock frequency.
 *
 * On TC4x the TriCore CPU core is clocked from the SRI bus, so this
 * returns the same value as CddSys_GetSriFrequency().
 *
 * @return CPU clock frequency in Hz.
 */
extern real32_T CddSys_GetCpuFrequency(void);

/**
 * @brief Gets the STM (System Timer Module) clock frequency.
 *
 * Reads the STMDIV field of CLOCK_SYSCCUCON0 and divides the current
 * system source frequency accordingly. If STMDIV = 0, the STM is
 * considered disabled and 0.0F is returned. If the low-power divider
 * is active (LPDIV != 0), the source is divided by 120.
 *
 * @return STM clock frequency in Hz, or 0.0F if STM is disabled.
 */
extern real32_T CddSys_GetStmFrequency(void);

/**
 * @brief Gets the Peripheral PLL1 (K2 divider) output frequency.
 *
 * Computes fPLL1 = N * fIN / (P * K2 * K2PREDIV), gated by the PLL
 * lock and power status bits of PERPLLSTAT. fIN is the currently
 * selected oscillator frequency.
 *
 * @return Peripheral PLL1 frequency in Hz.
 */
extern real32_T CddSys_GetPerPll1Frequency(void);

/**
 * @brief Gets the ADC (EVADC / EDSADC) clock frequency.
 *
 * Returns 0.0F if the ADC peripheral clock is disabled
 * (CLOCK_PERCCUCON1.ADCPERON = 0). Otherwise returns the Peripheral
 * PLL1 frequency (Fsource_1), which is the ADC clock source.
 *
 * @return ADC clock frequency in Hz, or 0.0F if the ADC clock is
 *         disabled.
 */
extern real32_T CddSys_GetAdcFrequency(void);

/**
 * \brief   Performs a single CMPSWAP.W atomic try-acquire on *LockPtr.
 *
 * \details alleged_free == 0 is the expected value (lock free).
 *          CddAsm_CmpAndSwap returns the *previous* value of *LockPtr:
 *            prev == 0 → swap succeeded → lock is now held by this core → return 1.
 *            prev != 0 → lock was held by another core → no swap → return 0.
 */
extern uint32_T CddSys_AcquireSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr);


/**
 * \brief   Releases the spinlock by writing 0 to *LockPtr atomically.
 *
 * \details A plain store is sufficient here because only the owning core writes 0,
 *          and the TriCore memory model guarantees store visibility across cores
 *          sharing LMU.  No CMPSWAP needed for the release path.
 */
extern void CddSys_ReleaseSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr);


#endif /* CDD_SYS_UTIL_H_ */

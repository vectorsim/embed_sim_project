/**********************************************************************************************************************
 * \file        cdd_asm_functions.c
 * \brief       Implementation of cdd_asm_functions.h
 *
 * \details     Implements CMPSWAP.W and LDMST atomic primitives using TASKING
 *              inline assembler for AURIX TC3xx.
 *
 *              Register allocation convention:
 *                E8  = extended register pair (D8 = low word, D9 = high word)
 *                CMPSWAP.W [An]off, En  :  D(n)   = new value
 *                                          D(n+1) = compare value (current)
 *                                          returns previous *ptr in D(n)
 *                LDMST     [An]off, En  :  D(n)   = data
 *                                          D(n+1) = mask
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  1.1  : All code is standard C99 with TASKING extensions
 *              - Rule  2.2  : No dead code
 *              - Directive 4.3: Assembly language isolated in this file only
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_asm_functions.h"

/**********************************************************************************************************************
 * Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CddAsm_CmpAndSwap
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddAsm_CmpAndSwap(P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) RegPtr,
                            uint32_T NewRegVal,
                            uint32_T CurrentRegVal)
{
    volatile uint32_T reg_value;

    __asm
    (
        /* Load operands into extended register pair E8:
         *   D8 = new value     (lower word of CMPSWAP pair)
         *   D9 = compare value (upper word of CMPSWAP pair) */
        "mov        d8,  %2          \n\t"
        "mov        d9,  %3          \n\t"
        /* Atomic compare-and-swap: if [%1] == D9, write D8 -> [%1]
         * Previous value of [%1] is returned in D8 */
        "cmpswap.w  [%1]0x0, e8      \n\t"
        /* Capture result */
        "mov        %0,  d8          \n\t"

        /* Outputs */
        : "=d" (reg_value)              /* %0 : previous register value     */
        /* Inputs */
        : "a"  (RegPtr),                /* %1 : target address register      */
          "d"  (NewRegVal),             /* %2 : new value to write           */
          "d"  (CurrentRegVal)          /* %3 : expected current value       */
        /* Clobbers — E8 is modified by the instruction */
        :
    );

    return reg_value;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddAsm_LdmSt
 *------------------------------------------------------------------------------------------------------------------*/
void CddAsm_LdmSt(P2VAR(volatile uint32_T, AUTOMATIC, CDD_APPL_DATA) RegPtr,
                  uint32_T Mask,
                  uint32_T Data)
{
    __asm
    (
        /* Load operands into extended register pair E8:
         *   D8 = data  (lower word of LDMST pair)
         *   D9 = mask  (upper word of LDMST pair) */
        "mov        d8,  %2          \n\t"
        "mov        d9,  %1          \n\t"
        /* Atomic load-modify-store:
         * *RegPtr = (*RegPtr & ~D9) | (D8 & D9) */
        "ldmst      [%0]0x0, e8      \n\t"

        /* No output operands */
        :
        /* Inputs */
        : "a"  (RegPtr),                /* %0 : target address register      */
          "d"  (Mask),                  /* %1 : bitmask                      */
          "d"  (Data)                   /* %2 : data for masked bits         */
        /* Clobbers */
        :
    );
}

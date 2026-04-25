/**********************************************************************************************************************
 * \file        cdd_asm_functions.c
 * \brief       Implementation of cdd_asm_functions.h
 *
 * \details     Implements CMPSWAP.W and LDMST atomic primitives using TASKING
 *              inline assembler for AURIX TC3xx.
 *
 *              Register allocation convention used here:
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
 * ASM_Cmp_And_Swap
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T ASM_Cmp_And_Swap(uint32_T *Reg_Ptr,
                          uint32_T  New_Reg_Value,
                          uint32_T  Current_Reg_Value)
{
    volatile uint32_T reg_value;

    __asm
    (
        /* Load operands into extended register pair E8:
         *   D8 = new value    (lower word of CMPSWAP pair)
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
        : "a"  (Reg_Ptr),               /* %1 : target address register      */
          "d"  (New_Reg_Value),         /* %2 : new value to write           */
          "d"  (Current_Reg_Value)      /* %3 : expected current value       */
        /* Clobbers — E8 is modified by the instruction */
        :
    );

    return reg_value;
}

/*--------------------------------------------------------------------------------------------------------------------
 * ASM_Load_Mode_Store
 *------------------------------------------------------------------------------------------------------------------*/
void ASM_Load_Mode_Store(volatile uint32_T *Reg_Ptr,
                         uint32_T           Mask,
                         uint32_T           Data)
{
    __asm
    (
        /* Load operands into extended register pair E8:
         *   D8 = data  (lower word of LDMST pair)
         *   D9 = mask  (upper word of LDMST pair) */
        "mov        d8,  %2          \n\t"
        "mov        d9,  %1          \n\t"
        /* Atomic load-modify-store:
         * *Reg_Ptr = (*Reg_Ptr & ~D9) | (D8 & D9) */
        "ldmst      [%0]0x0, e8      \n\t"

        /* No output operands */
        :
        /* Inputs */
        : "a"  (Reg_Ptr),               /* %0 : target address register      */
          "d"  (Mask),                  /* %1 : bitmask                      */
          "d"  (Data)                   /* %2 : data for masked bits         */
        /* Clobbers */
        :
    );
}

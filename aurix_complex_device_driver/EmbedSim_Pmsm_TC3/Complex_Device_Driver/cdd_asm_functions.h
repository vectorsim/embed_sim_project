/**********************************************************************************************************************
 * \file        cdd_asm_functions.h
 * \brief       Inline-assembler wrappers for TriCore atomic memory operations.
 *
 * \details     Exposes two hardware atomic primitives available on AURIX TC3xx:
 *              - CMPSWAP.W  : Compare-and-swap word (lock-free synchronisation)
 *              - LDMST      : Load-modify-store word (atomic bit-field update)
 *
 *              Both operations complete in a single bus cycle and are safe across
 *              cores sharing LMU RAM without requiring interrupt disabling.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5  : Declarations match definitions in cdd_asm_functions.c
 *              - Rule 17.2  : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_ASM_FUNCTIONS_H_
#define CDD_ASM_FUNCTIONS_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Atomic compare-and-swap on a 32-bit memory location.
 *
 * \details If *Reg_Ptr == Current_Reg_Value, atomically writes New_Reg_Value
 *          to *Reg_Ptr.  In either case the original value of *Reg_Ptr before
 *          the operation is returned (CMPSWAP.W semantics, TC3xx ISA P.3-45).
 *
 * \param   Reg_Ptr             Pointer to the target 32-bit location
 * \param   New_Reg_Value       Value to write if the compare succeeds
 * \param   Current_Reg_Value   Expected current value of *Reg_Ptr
 * \return  Value of *Reg_Ptr before the operation
 */
extern uint32_T ASM_Cmp_And_Swap(uint32_T *Reg_Ptr,
                                  uint32_T  New_Reg_Value,
                                  uint32_T  Current_Reg_Value);

/**
 * \brief   Atomic load-modify-store on a 32-bit memory location.
 *
 * \details Performs the operation:
 *              *Reg_Ptr = (*Reg_Ptr & ~Mask) | (Data & Mask)
 *          atomically using the LDMST instruction (TC3xx ISA P.3-88).
 *          Only bits set in Mask are affected.
 *
 * \param   Reg_Ptr   Pointer to the target 32-bit location
 * \param   Mask      Bitmask selecting the bits to modify (1 = modifiable)
 * \param   Data      New data for the masked bits
 * \return  None
 */
extern void ASM_Load_Mode_Store(volatile uint32_T *Reg_Ptr,
                                uint32_T           Mask,
                                uint32_T           Data);

#endif /* CDD_ASM_FUNCTIONS_H_ */

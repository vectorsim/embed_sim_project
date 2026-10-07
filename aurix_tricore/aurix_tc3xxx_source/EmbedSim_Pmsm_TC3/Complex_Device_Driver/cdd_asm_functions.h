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
#include "cdd_config.h"   /* embed_sim_sys_types.h + embed_sim_compiler.h */

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Atomic compare-and-swap on a 32-bit memory location.
 *
 * \details If *RegPtr == CurrentRegVal, atomically writes NewRegVal to *RegPtr.
 *          In either case the original value of *RegPtr before the operation is
 *          returned (CMPSWAP.W semantics, TC3xx ISA P.3-45).
 *
 * \param[in,out] RegPtr       Pointer to the target 32-bit location
 * \param[in]     NewRegVal    Value to write if the compare succeeds
 * \param[in]     CurrentRegVal Expected current value of *RegPtr
 * \return  Value of *RegPtr before the operation  [dimensionless]
 */
extern uint32_T CddAsm_CmpAndSwap(P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) RegPtr,
                                   uint32_T NewRegVal,
                                   uint32_T CurrentRegVal);

/**
 * \brief   Atomic load-modify-store on a 32-bit memory location.
 *
 * \details Performs:  *RegPtr = (*RegPtr & ~Mask) | (Data & Mask)
 *          atomically using the LDMST instruction (TC3xx ISA P.3-88).
 *          Only bits set in Mask are affected.
 *
 * \param[in,out] RegPtr  Pointer to the target 32-bit location (volatile)
 * \param[in]     Mask    Bitmask selecting bits to modify (1 = modifiable)
 * \param[in]     Data    New data for the masked bits
 * \return  void
 */
extern void CddAsm_LdmSt(P2VAR(volatile uint32_T, AUTOMATIC, CDD_APPL_DATA) RegPtr,
                          uint32_T Mask,
                          uint32_T Data);

#endif /* CDD_ASM_FUNCTIONS_H_ */

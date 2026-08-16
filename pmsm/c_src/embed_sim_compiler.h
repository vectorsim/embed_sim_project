/**********************************************************************************************************************
 * \file        embed_sim_compiler.h
 * \brief       AUTOSAR compiler abstraction macros for EmbedSim CDD layer.
 *
 * \details     Provides:
 *                - Storage class macros : STATIC, INLINE, LOCAL_INLINE
 *                - Pointer class macros : P2VAR, P2CONST, CONSTP2VAR, CONSTP2CONST, P2FUNC
 *                - Memory class tokens  : AUTOMATIC, CDD_APPL_DATA, CDD_APPL_CODE
 *
 *              Replaces bare C keywords in all CDD translation units.
 *              Compatible with TASKING ctc v6.x and GCC arm-none-eabi.
 *
 *              AUTOSAR references:
 *                  [AUTOSAR_SWS_CompilerAbstraction] AUTOSAR_SWS_CompilerAbstraction.pdf
 *                  [AUTOSAR_SWS_PlatformTypes]       AUTOSAR_SWS_PlatformTypes.pdf
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule 20.10 : ## and # operators not used in these macros.
 *              - Rule  4.9  : Function-like macros used for portability only;
 *                             no function equivalent exists for storage class
 *                             specifiers or type qualifiers.
 *
 * \version     1.0.0
 * \date        2025-05-24
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *              Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBED_SIM_COMPILER_H
#define EMBED_SIM_COMPILER_H

/* ─────────────────────────────────────────────────────────────────────────────
 * Storage class macros
 * ───────────────────────────────────────────────────────────────────────────*/

/** \brief  Internal linkage — file-scope functions and variables.
 *          Replaces bare \c static keyword across all CDD translation units.  */
#define STATIC              static

/** \brief  Inlining hint for non-static utility functions.                   */
#define INLINE              inline

/** \brief  Static inline for performance-critical helpers.                    */
#define LOCAL_INLINE        static inline

/* ─────────────────────────────────────────────────────────────────────────────
 * Memory class tokens
 *
 * Kept as empty expansions on TriCore flat-memory targets.
 * Replace with linker-section attributes for banked / AUTOSAR OS targets.
 * ───────────────────────────────────────────────────────────────────────────*/

/** \brief  Automatic storage duration (local variables, formal parameters).  */
#define AUTOMATIC

/** \brief  Application data section (RAM, no const).                         */
#define CDD_APPL_DATA

/** \brief  Application code section (flash / ROM).                           */
#define CDD_APPL_CODE

/** \brief  Application const section (flash / ROM, read-only data).          */
#define CDD_APPL_CONST

/* ─────────────────────────────────────────────────────────────────────────────
 * Pointer class macros  [AUTOSAR_SWS_CompilerAbstraction §8.3]
 *
 * Usage examples:
 *
 *   void  CddQspi4_Exchange(P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA) txBuf,
 *                           P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA) rxBuf,
 *                           uint32_T count);
 *
 *   void  CddSys_AcquireSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) lockPtr);
 * ───────────────────────────────────────────────────────────────────────────*/

/**
 * \brief  Pointer to variable data.  The pointer itself may be re-seated.
 *         \p ptrtype  pointed-at type  \p memclass  memory section of pointed data
 *         \p ptrclass memory section of the pointer variable itself
 */
#define P2VAR(ptrtype, memclass, ptrclass)          ptrtype *               /* PRQA S 3453 */

/**
 * \brief  Pointer to constant data.  The pointed-at value is read-only.
 */
#define P2CONST(ptrtype, memclass, ptrclass)        const ptrtype *         /* PRQA S 3453 */

/**
 * \brief  Constant pointer to variable data.  The pointer address is fixed
 *         (e.g. a pointer passed by address and never re-seated by callee).
 */
#define CONSTP2VAR(ptrtype, memclass, ptrclass)     ptrtype * const         /* PRQA S 3453 */

/**
 * \brief  Constant pointer to constant data.
 */
#define CONSTP2CONST(ptrtype, memclass, ptrclass)   const ptrtype * const   /* PRQA S 3453 */

/**
 * \brief  Pointer to a function.
 *         \p rettype  return type  \p ptrclass  memory class  \p fctname  pointer name
 */
#define P2FUNC(rettype, ptrclass, fctname)          rettype (* fctname)     /* PRQA S 3453 */

#endif /* EMBED_SIM_COMPILER_H */

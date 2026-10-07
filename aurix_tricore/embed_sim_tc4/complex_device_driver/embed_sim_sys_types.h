/**********************************************************************************************************************
 * \file      embed_sim_sys_types.h
 * \brief     EmbedSim system-wide scalar types, math constants, and inline helpers.
 *
 * \details   Single source of truth for the fixed-width integer / float names
 *            (`int8_T` .. `uint64_T`, `real32_T`, `real64_T`), project math
 *            constants (`ES_MATH_*`), unit conversions (`ES_CON_*`),
 *            frequency constants (`MHZ_*`), the AUTOSAR compiler-abstraction
 *            macros (`STATIC`, `AUTOMATIC`, `P2VAR`, `P2CONST`, ...), and two
 *            tiny inline helpers (`EmbedSim_WrapAngleTwoPi`,
 *            `EmbedSim_ClampValue`).
 *
 *            Targets 32-bit MCUs (Infineon AURIX TriCore, ARM Cortex-M4).
 *            Compiles under hosted (unit tests, simulation) and freestanding
 *            (target) toolchains.
 *
 * \note      MISRA C:2012 : 8.5, 8.6 (documented deviation for the two
 *            `inline` helpers), 10.4 (`ES_MATH_*` macros carry an explicit
 *            `(real32_T)` cast), 17.2, 21.1.
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
 * \version   2.0.0
 * \date      2026-08-12
 * \author    EmbedSim / EV Light Vehicle Foundation
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/
#ifndef SYS_TYPES_H
#define SYS_TYPES_H

#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

/*═════════════════════════════════════════════════════════════════════════════
 *  1. Boolean constants
 *═══════════════════════════════════════════════════════════════════════════*/
#ifndef FALSE
#define FALSE   (0U)   /**< Boolean false. */
#endif
#ifndef TRUE
#define TRUE    (1U)   /**< Boolean true.  */
#endif

#if (!defined(__cplusplus)) && (!defined(__bool_true_false_are_defined))
#  ifndef false
#    define false   (0U)   /**< C99 lowercase false. */
#  endif
#  ifndef true
#    define true    (1U)   /**< C99 lowercase true.  */
#  endif
#endif

/*═════════════════════════════════════════════════════════════════════════════
 *  2. Fixed-width integer types
 *═══════════════════════════════════════════════════════════════════════════*/
typedef signed char        int8_T;    /**< Signed   8-bit. */
typedef unsigned char      uint8_T;   /**< Unsigned 8-bit. */
typedef short              int16_T;   /**< Signed  16-bit. */
typedef unsigned short     uint16_T;  /**< Unsigned 16-bit.*/
typedef int                int32_T;   /**< Signed  32-bit. */
typedef unsigned int       uint32_T;  /**< Unsigned 32-bit.*/
typedef long long          int64_T;   /**< Signed  64-bit. */
typedef unsigned long long uint64_T;  /**< Unsigned 64-bit.*/

/*═════════════════════════════════════════════════════════════════════════════
 *  3. Floating-point types
 *═══════════════════════════════════════════════════════════════════════════*/
typedef float              real32_T;  /**< IEEE-754 single precision. */
typedef double             real64_T;  /**< IEEE-754 double precision. */

/*═════════════════════════════════════════════════════════════════════════════
 *  4. Generic / legacy types
 *═══════════════════════════════════════════════════════════════════════════*/
typedef double             real_T;       /**< Generic real (double).      */
typedef double             time_T;       /**< Time in seconds (double).   */
typedef unsigned char      boolean_T;    /**< Boolean byte (0 or 1).      */
typedef int                int_T;        /**< Generic signed int.         */
typedef unsigned int       uint_T;       /**< Generic unsigned int.       */
typedef unsigned long      ulong_T;      /**< Generic unsigned long.      */
typedef unsigned long long ulonglong_T;  /**< Generic unsigned long long. */
typedef char               char_T;       /**< Generic char.               */
typedef unsigned char      uchar_T;      /**< Generic unsigned char.      */
typedef char_T             byte_T;       /**< Generic byte.               */
typedef void *             pointer_T;    /**< Opaque generic pointer.     */

/*═════════════════════════════════════════════════════════════════════════════
 *  5. Integer limits (companion to §2)
 *═══════════════════════════════════════════════════════════════════════════*/
#define MAX_int8_T    ((int8_T)(127))
#define MIN_int8_T    ((int8_T)(-128))
#define MAX_uint8_T   ((uint8_T)(255U))

#define MAX_int16_T   ((int16_T)(32767))
#define MIN_int16_T   ((int16_T)(-32768))
#define MAX_uint16_T  ((uint16_T)(65535U))

#define MAX_int32_T   ((int32_T)(2147483647))
#define MIN_int32_T   ((int32_T)(-2147483647-1))
#define MAX_uint32_T  ((uint32_T)(0xFFFFFFFFU))

#define MAX_int64_T   ((int64_T)(9223372036854775807LL))
#define MIN_int64_T   ((int64_T)(-9223372036854775807LL-1LL))
#define MAX_uint64_T  ((uint64_T)(0xFFFFFFFFFFFFFFFFULL))

/*═════════════════════════════════════════════════════════════════════════════
 *  6. Storage class macros
 *═══════════════════════════════════════════════════════════════════════════*/
#define STATIC        static         /**< Internal linkage.               */
#define INLINE        inline         /**< Inlining hint.                  */
#define LOCAL_INLINE  static inline  /**< Static inline for hot helpers.  */

/*═════════════════════════════════════════════════════════════════════════════
 *  7. Memory class tokens (empty on flat-memory targets)
 *═══════════════════════════════════════════════════════════════════════════*/
#define AUTOMATIC       /**< Automatic storage duration. */
#define CDD_APPL_DATA   /**< RAM, no const.              */
#define CDD_APPL_CODE   /**< Flash / ROM code.           */
#define CDD_APPL_CONST  /**< Flash / ROM const data.     */

/*═════════════════════════════════════════════════════════════════════════════
 *  8. Pointer class macros  [AUTOSAR_SWS_CompilerAbstraction §8.3]
 *═══════════════════════════════════════════════════════════════════════════*/
#define P2VAR(PtrType, MemClass, PtrClass)          PtrType *              /* PRQA S 3453 */
#define P2CONST(PtrType, MemClass, PtrClass)        const PtrType *        /* PRQA S 3453 */
#define CONSTP2VAR(PtrType, MemClass, PtrClass)     PtrType * const        /* PRQA S 3453 */
#define CONSTP2CONST(PtrType, MemClass, PtrClass)   const PtrType * const  /* PRQA S 3453 */
#define P2FUNC(RetType, PtrClass, FctName)          RetType (* FctName)    /* PRQA S 3453 */

/*═════════════════════════════════════════════════════════════════════════════
 *  9. Frequency constants [Hz]
 *═══════════════════════════════════════════════════════════════════════════*/
#define MHZ_400   (400000000.0F)
#define MHZ_300   (300000000.0F)
#define MHZ_200   (200000000.0F)
#define MHZ_160   (160000000.0F)
#define MHZ_100   (100000000.0F)
#define MHZ_50    ( 50000000.0F)
#define MHZ_30    ( 30000000.0F)
#define MHZ_25    ( 25000000.0F)
#define MHZ_20    ( 20000000.0F)
#define MHZ_5     (  5000000.0F)
#define MHZ_1     (  1000000.0F)

#define KHZ_30    ( 30000.0F)
#define KHZ_25    ( 25000.0F)
#define KHZ_20    ( 20000.0F)
#define KHZ_10    ( 10000.0F)

/*═════════════════════════════════════════════════════════════════════════════
 * 10. Mathematical constants (real32_T, MISRA 10.4-cast)
 *═══════════════════════════════════════════════════════════════════════════*/
#define ES_MATH_HALF_F             ((real32_T)0.50000000000f)   /**< 0.5.  */
#define ES_MATH_ONE_F              ((real32_T)1.00000000000f)   /**< 1.0.  */
#define ES_MATH_TWO_F              ((real32_T)2.00000000000f)   /**< 2.0.  */

#define ES_MATH_ONE_THIRD_F        ((real32_T)0.33333333333f)   /**< 1/3.  */
#define ES_MATH_TWO_THIRDS_F       ((real32_T)0.66666666667f)   /**< 2/3.  */

#define ES_MATH_SQRT3_F            ((real32_T)1.73205080757f)   /**< √3.   */
#define ES_MATH_HALF_SQRT3_F       ((real32_T)0.86602540378f)   /**< √3/2. */
#define ES_MATH_INV_SQRT3_F        ((real32_T)0.57735026919f)   /**< 1/√3. */
#define ES_MATH_TWO_INV_SQRT3_F    ((real32_T)1.15470053838f)   /**< 2/√3. */

#define ES_MATH_PI_OVER_6_F        ((real32_T)0.52359877559f)   /**< π/6  [rad]. */
#define ES_MATH_PI_OVER_3_F        ((real32_T)1.04719755120f)   /**< π/3  [rad]. */
#define ES_MATH_PI_OVER_2_F        ((real32_T)1.57079632679f)   /**< π/2  [rad]. */
#define ES_MATH_2PI_OVER_3_F       ((real32_T)2.09439510239f)   /**< 2π/3 [rad]. */
#define ES_MATH_PI_F               ((real32_T)3.14159265359f)   /**< π    [rad]. */
#define ES_MATH_4PI_OVER_3_F       ((real32_T)4.18879020479f)   /**< 4π/3 [rad]. */
#define ES_MATH_5PI_OVER_3_F       ((real32_T)5.23598775598f)   /**< 5π/3 [rad]. */
#define ES_MATH_2PI_F              ((real32_T)6.28318530718f)   /**< 2π   [rad]. */

/*═════════════════════════════════════════════════════════════════════════════
 * 11. Unit conversions
 *═══════════════════════════════════════════════════════════════════════════*/
/** \brief RPM -> rad/s.  Note: argument is evaluated more than once. */
#define ES_CON_RPM_TO_RAD(RPM)   ((RPM * ES_MATH_2PI_F) / 60.0F)

/** \brief rad/s -> RPM.  Note: argument is evaluated more than once. */
#define ES_CON_RAD_TO_RPM(RAD)   ((RAD * 60.0F) / ES_MATH_2PI_F)

/*═════════════════════════════════════════════════════════════════════════════
 * 12. Inline helpers
 *═══════════════════════════════════════════════════════════════════════════*/

/**
 * \brief   Wrap an angle in radians into [0.0, 2π).
 *
 * \param[in,out] AnglePtr  Angle to wrap, modified in place. Must not be NULL.
 *
 * \details Applies `fmodf` then adds 2π when the result is negative.
 *          NaN / Inf propagate unchanged. Large inputs lose low-order bits.
 *
 * \note    Rule 8.6 deviation: defined inline in this header by design.
 *
 * \see     ES_MATH_2PI_F
 */
inline void EmbedSim_WrapAngleTwoPi(real32_T* AnglePtr)
{
    *AnglePtr = fmodf(*AnglePtr, ES_MATH_2PI_F);
    if (*AnglePtr < 0.0F)
    {
        *AnglePtr += ES_MATH_2PI_F;
    }
}

/**
 * \brief   Clamp a value to the closed interval [MinVal, MaxVal].
 *
 * \param[in] Val     Value to clamp.
 * \param[in] MinVal  Lower bound (inclusive).
 * \param[in] MaxVal  Upper bound (inclusive), must be >= MinVal.
 *
 * \return  MinVal if Val < MinVal, MaxVal if Val > MaxVal, else Val.
 *          NaN is returned unchanged. MinVal > MaxVal is a caller error.
 *
 * \note    Rule 8.6 deviation: defined inline in this header by design.
 */
inline real32_T EmbedSim_ClampValue(real32_T Val, real32_T MinVal, real32_T MaxVal)
{
    real32_T result;

    if (Val < MinVal)
    {
        result = MinVal;
    }
    else if (Val > MaxVal)
    {
        result = MaxVal;
    }
    else
    {
        result = Val;
    }

    return result;
}

#endif /* SYS_TYPES_H */

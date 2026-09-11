/**********************************************************************************************************************
 * \file      embed_sim_cython_interface.h
 * \brief     Top-level PMSM control module with DFC controller.
 *
 * \details   Defines the main control structures and functions for permanent magnet
 *            synchronous motors (PMSM). Supports open-loop and DFC control modes
 *            with smooth reference trajectory generation.
 *            Targets 32-bit MCUs (Infineon AURIX TriCore, ARM Cortex-M4).
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier
 *              - Rule  8.6 : No definitions in header files
 *              - Rule 17.2 : No recursion
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter → Uppercase)
 *              - Output pointers: PascalCasePtr
 *              - Local variables: Lower camelCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   2.0.0
 * \date      2026-08-12
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/
#ifndef SYS_TYPES_H
#define SYS_TYPES_H


#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * Boolean constants
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef FALSE
#define FALSE   (0U)
#endif

#ifndef TRUE
#define TRUE    (1U)
#endif

/* ─────────────────────────────────────────────────────────────────────────────
 * Logical types
 * ───────────────────────────────────────────────────────────────────────────*/
#if (!defined(__cplusplus)) && (!defined(__bool_true_false_are_defined))
#  ifndef false
#    define false                      (0U)
#  endif
#  ifndef true
#    define true                       (1U)
#  endif
#endif

/* ─────────────────────────────────────────────────────────────────────────────
 * Fixed-width integer types
 * ───────────────────────────────────────────────────────────────────────────*/
typedef signed char        int8_T;
typedef unsigned char      uint8_T;
typedef short              int16_T;
typedef unsigned short     uint16_T;
typedef int                int32_T;
typedef unsigned int       uint32_T;
typedef long long          int64_T;
typedef unsigned long long uint64_T;

/* ─────────────────────────────────────────────────────────────────────────────
 * Floating-point types
 * ───────────────────────────────────────────────────────────────────────────*/
typedef float              real32_T;
typedef double             real64_T;

/* ─────────────────────────────────────────────────────────────────────────────
 * Generic types
 * ───────────────────────────────────────────────────────────────────────────*/
typedef double             real_T;
typedef double             time_T;
typedef unsigned char      boolean_T;
typedef int                int_T;
typedef unsigned int       uint_T;
typedef unsigned long      ulong_T;
typedef unsigned long long ulonglong_T;
typedef char               char_T;
typedef unsigned char      uchar_T;
typedef char_T             byte_T;

/* ─────────────────────────────────────────────────────────────────────────────
 * Integer limits
 * ───────────────────────────────────────────────────────────────────────────*/
#define MAX_int8_T         ((int8_T)(127))
#define MIN_int8_T         ((int8_T)(-128))
#define MAX_uint8_T        ((uint8_T)(255U))

#define MAX_int16_T        ((int16_T)(32767))
#define MIN_int16_T        ((int16_T)(-32768))
#define MAX_uint16_T       ((uint16_T)(65535U))

#define MAX_int32_T        ((int32_T)(2147483647))
#define MIN_int32_T        ((int32_T)(-2147483647-1))
#define MAX_uint32_T       ((uint32_T)(0xFFFFFFFFU))

#define MAX_int64_T        ((int64_T)(9223372036854775807LL))
#define MIN_int64_T        ((int64_T)(-9223372036854775807LL-1LL))
#define MAX_uint64_T       ((uint64_T)(0xFFFFFFFFFFFFFFFFULL))


/* ─────────────────────────────────────────────────────────────────────────────
 * Pointer type (D-Work blocks)
 * ───────────────────────────────────────────────────────────────────────────*/
typedef void *             pointer_T;

/* ─────────────────────────────────────────────────────────────────────────────
 * Mathematical Constants  (real32_T, single-precision)
 *
 * Central definitions for all EmbedSim modules.
 * Cast to (real32_T) for MISRA C:2012 Rule 10.4 type consistency.
 * real32_T is defined above — no additional include required.
 *
 * Naming:  ES_MATH_  prefix is project-wide and avoids collision with
 *          non-standard glibc macros (M_SQRT3 etc.) and module-local prefixes.
 * ───────────────────────────────────────────────────────────────────────────*/

/* --- Integer-valued scalars ----------------------------------------------- */
/** \brief  0.5   = 1/2                              [dimensionless] */
#define ES_MATH_HALF_F              ((real32_T)0.50000000000f)

/** \brief  1.0                                      [dimensionless] */
#define ES_MATH_ONE_F               ((real32_T)1.00000000000f)

/** \brief  2.0                                      [dimensionless] */
#define ES_MATH_TWO_F               ((real32_T)2.00000000000f)

/* --- Rational fractions --------------------------------------------------- */
/** \brief  1/3  ≈ 0.33333333333                     [dimensionless] */
#define ES_MATH_ONE_THIRD_F         ((real32_T)0.33333333333f)

/** \brief  2/3  ≈ 0.66666666667                     [dimensionless] */
#define ES_MATH_TWO_THIRDS_F        ((real32_T)0.66666666667f)

/* --- Square-root family --------------------------------------------------- */
/** \brief  √3   ≈ 1.73205080757                     [dimensionless] */
#define ES_MATH_SQRT3_F             ((real32_T)1.73205080757f)

/** \brief  √3/2 ≈ 0.86602540378                     [dimensionless] */
#define ES_MATH_HALF_SQRT3_F        ((real32_T)0.86602540378f)

/** \brief  1/√3 ≈ 0.57735026919                     [dimensionless] */
#define ES_MATH_INV_SQRT3_F         ((real32_T)0.57735026919f)

/** \brief  2/√3 ≈ 1.15470053838                     [dimensionless] */
#define ES_MATH_TWO_INV_SQRT3_F     ((real32_T)1.15470053838f)

/* --- π and its common multiples  [rad] ------------------------------------ */
/** \brief  π/6  =  30°  ≈ 0.52359877559             [rad] */
#define ES_MATH_PI_OVER_6_F         ((real32_T)0.52359877559f)

/** \brief  π/3  =  60°  ≈ 1.04719755120             [rad] */
#define ES_MATH_PI_OVER_3_F         ((real32_T)1.04719755120f)

/** \brief  π/2  =  90°  ≈ 1.57079632679             [rad] */
#define ES_MATH_PI_OVER_2_F         ((real32_T)1.57079632679f)

/** \brief  2π/3 = 120°  ≈ 2.09439510239             [rad] */
#define ES_MATH_2PI_OVER_3_F        ((real32_T)2.09439510239f)

/** \brief  π    = 180°  ≈ 3.14159265359             [rad] */
#define ES_MATH_PI_F                ((real32_T)3.14159265359f)

/** \brief  4π/3 = 240°  ≈ 4.18879020479             [rad] */
#define ES_MATH_4PI_OVER_3_F        ((real32_T)4.18879020479f)

/** \brief  5π/3 = 300°  ≈ 5.23598775598             [rad] */
#define ES_MATH_5PI_OVER_3_F        ((real32_T)5.23598775598f)

/** \brief  2π   = 360°  ≈ 6.28318530718             [rad] */
#define ES_MATH_2PI_F               ((real32_T)6.28318530718f)

/**
 * \brief  Conversion: RPM to rad/s
 *
 * \param[in] RPM  Speed in revolutions per minute.
 *
 * \return  Speed in radians per second.
 */
#define ES_CON_RPM_TO_RAD(RPM)             ((RPM * ES_MATH_2PI_F) / 60.0F)

/**
 * \brief  Conversion: rad/s to RPM
 *
 * \param[in] RAD  Speed in radians per second.
 *
 * \return  Speed in revolutions per minute.
 */
#define ES_CON_RAD_TO_RPM(RAD)             ((RAD * 60.0F) / ES_MATH_2PI_F)



/*********************************************************************************************************************/
/*------------------------------------------------common inline function --------------------------------------------*/
/*********************************************************************************************************************/
inline void EmbedSim_WrapAngleTwoPi(real32_T* AnglePtr)
{
    *AnglePtr = fmodf(*AnglePtr, ES_MATH_2PI_F);
    if(*AnglePtr < 0.0F)
    {
        *AnglePtr += ES_MATH_2PI_F;
    }
}



/**
 * \brief   Clamp value to specified limits
 *
 * \details Limits a value to a range defined by minVal and maxVal.
 *          If value is below minVal, returns minVal.
 *          If value is above maxVal, returns maxVal.
 *          Otherwise returns the original value.
 *
 * \param[in] val     Value to clamp.
 * \param[in] minVal  Minimum allowed value.
 * \param[in] maxVal  Maximum allowed value.
 *
 * \return  Clamped value within [minVal, maxVal].
 */
inline  real32_T EmbedSim_ClampValue(real32_T Val, real32_T MinVal, real32_T MaxVal)
 {
     real32_T result;

     if(Val < MinVal)
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

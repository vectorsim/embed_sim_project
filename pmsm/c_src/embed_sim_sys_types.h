/**********************************************************************************************************************
 * \file      embed_sim_sys_types.h
 * \brief     EmbedSim system-wide fixed-width types, mathematical constants,
 *            and common inline helpers.
 *
 * \details   Single source of truth for the scalar type names used across the
 *            EmbedSim codebase (int8_T .. uint64_T, real32_T, real64_T, ...),
 *            the project-wide mathematical constants prefixed `ES_MATH_`, the
 *            unit-conversion macros `ES_CON_*`, and a small set of inline
 *            helpers (`EmbedSim_WrapAngleTwoPi`, `EmbedSim_ClampValue`) that
 *            are small enough to live in a header.
 *
 *            Targets 32-bit MCUs (Infineon AURIX TriCore, ARM Cortex-M4) and
 *            is written to compile cleanly under both a hosted toolchain
 *            (unit tests, simulation) and a freestanding one (target).
 *
 *            ## Why this file exists
 *            - The short ILLD-style aliases (`sint32`, `uint16`, ...) are NOT
 *              re-exported here. Only the `*_T` names are visible, so a header
 *              that includes `embed_sim_sys_types.h` alone never depends on an
 *              iLLD include being pulled in first.
 *            - `real32_T` is single-precision `float` and `real64_T` is
 *              `double`, chosen so that all motor-control math stays in
 *              single precision on the target unless a caller deliberately
 *              promotes.
 *            - `ES_MATH_*` constants are all `(real32_T)`-suffixed so that
 *              mixed arithmetic with `real32_T` operands does not trigger
 *              MISRA C:2012 Rule 10.4 (no implicit narrowing of a
 *              double-precision literal into a single-precision operation).
 *
 *            ## Boolean handling
 *            `TRUE` / `FALSE` and `true` / `false` are defined here only if
 *            the translation environment has not already provided them. This
 *            keeps the header usable from C90 (no `<stdbool.h>`), from C99+
 *            (where `<stdbool.h>` may have been included first), and from C++
 *            (where `true` / `false` are keywords and must not be redefined).
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier.
 *              - Rule  8.6 : No definitions in header files - the two inline
 *                            helpers below are the single, deliberate
 *                            exception; they are `inline` so the linker sees
 *                            one definition per translation unit that uses
 *                            them, not a duplicate symbol.
 *              - Rule 10.4 : `ES_MATH_*` macros carry an explicit
 *                            `(real32_T)` cast so expressions built from them
 *                            do not implicitly promote to `double`.
 *              - Rule 17.2 : No recursion (none present).
 *              - Rule 21.1 : No `#define` of reserved identifiers (`_X`).
 *
 *              Deviation, flagged here rather than inline:
 *              - Rule 8.6 : the two `inline` helpers are defined, not just
 *                           declared, in this header. This is intentional.
 *                           See the notes on each helper below.
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
 *
 * Guarded so a translation unit that has already included a header defining
 * TRUE / FALSE (rare, but happens with vendor SDKs) does not trip a
 * macro-redefinition warning.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef FALSE
/** \brief Boolean false, as an unsigned int literal. */
#define FALSE   (0U)
#endif

#ifndef TRUE
/** \brief Boolean true, as an unsigned int literal. */
#define TRUE    (1U)
#endif

/* ─────────────────────────────────────────────────────────────────────────────
 * Logical types
 *
 * Only define lowercase `true` / `false` when:
 *   - we are not compiling as C++ (where they are keywords), and
 *   - <stdbool.h> has not already been included (which would define
 *     `__bool_true_false_are_defined`), and
 *   - the name is not already taken by a vendor macro.
 * ───────────────────────────────────────────────────────────────────────────*/
#if (!defined(__cplusplus)) && (!defined(__bool_true_false_are_defined))
#  ifndef false
/** \brief Lowercase boolean false, C99-compatible. */
#    define false                      (0U)
#  endif
#  ifndef true
/** \brief Lowercase boolean true, C99-compatible. */
#    define true                       (1U)
#  endif
#endif

/* ─────────────────────────────────────────────────────────────────────────────
 * Fixed-width integer types
 *
 * Provided so the codebase does not depend on <stdint.h> being present, and
 * so every module spells the same name the same way. The widths are the
 * standard ones for the target ABIs (ILP32 on TriCore / Cortex-M).
 * ───────────────────────────────────────────────────────────────────────────*/
/** \brief Signed 8-bit integer.   */
typedef signed char        int8_T;
/** \brief Unsigned 8-bit integer. */
typedef unsigned char      uint8_T;
/** \brief Signed 16-bit integer.  */
typedef short              int16_T;
/** \brief Unsigned 16-bit integer.*/
typedef unsigned short     uint16_T;
/** \brief Signed 32-bit integer.  */
typedef int                int32_T;
/** \brief Unsigned 32-bit integer.*/
typedef unsigned int       uint32_T;
/** \brief Signed 64-bit integer.  */
typedef long long          int64_T;
/** \brief Unsigned 64-bit integer.*/
typedef unsigned long long uint64_T;

/* ─────────────────────────────────────────────────────────────────────────────
 * Floating-point types
 * ───────────────────────────────────────────────────────────────────────────*/
/** \brief Single-precision IEEE-754 float.  */
typedef float              real32_T;
/** \brief Double-precision IEEE-754 float.  */
typedef double             real64_T;

/* ─────────────────────────────────────────────────────────────────────────────
 * Generic types
 *
 * Legacy / Simulink-style aliases kept for generated-code compatibility.
 * New hand-written modules should prefer the fixed-width names above.
 * ───────────────────────────────────────────────────────────────────────────*/
/** \brief Generic real (double). */
typedef double             real_T;
/** \brief Time in seconds (double). */
typedef double             time_T;
/** \brief Boolean-as-byte (0 or 1). */
typedef unsigned char      boolean_T;
/** \brief Generic signed int. */
typedef int                int_T;
/** \brief Generic unsigned int. */
typedef unsigned int       uint_T;
/** \brief Generic unsigned long. */
typedef unsigned long      ulong_T;
/** \brief Generic unsigned long long. */
typedef unsigned long long ulonglong_T;
/** \brief Generic char. */
typedef char               char_T;
/** \brief Generic unsigned char. */
typedef unsigned char      uchar_T;
/** \brief Generic byte (char-sized). */
typedef char_T             byte_T;

/* ─────────────────────────────────────────────────────────────────────────────
 * Integer limits
 *
 * Companion macros to the fixed-width typedefs above. Values are written out
 * rather than taken from <limits.h> so the same names exist on every target.
 * ───────────────────────────────────────────────────────────────────────────*/
/** \brief Maximum value of int8_T.   */
#define MAX_int8_T         ((int8_T)(127))
/** \brief Minimum value of int8_T.   */
#define MIN_int8_T         ((int8_T)(-128))
/** \brief Maximum value of uint8_T.  */
#define MAX_uint8_T        ((uint8_T)(255U))

/** \brief Maximum value of int16_T.  */
#define MAX_int16_T        ((int16_T)(32767))
/** \brief Minimum value of int16_T.  */
#define MIN_int16_T        ((int16_T)(-32768))
/** \brief Maximum value of uint16_T. */
#define MAX_uint16_T       ((uint16_T)(65535U))

/** \brief Maximum value of int32_T.  */
#define MAX_int32_T        ((int32_T)(2147483647))
/** \brief Minimum value of int32_T (written as -2147483647-1 to avoid the
 *         "negation of an unsigned constant" trap). */
#define MIN_int32_T        ((int32_T)(-2147483647-1))
/** \brief Maximum value of uint32_T. */
#define MAX_uint32_T       ((uint32_T)(0xFFFFFFFFU))

/** \brief Maximum value of int64_T.  */
#define MAX_int64_T        ((int64_T)(9223372036854775807LL))
/** \brief Minimum value of int64_T (same -N-1 trick as int32_T). */
#define MIN_int64_T        ((int64_T)(-9223372036854775807LL-1LL))
/** \brief Maximum value of uint64_T. */
#define MAX_uint64_T       ((uint64_T)(0xFFFFFFFFFFFFFFFFULL))


/* ─────────────────────────────────────────────────────────────────────────────
 * Pointer type (D-Work blocks)
 *
 * Opaque generic pointer used by generated D-Work blocks. Not for use in
 * hand-written driver code - prefer a typed pointer.
 * ───────────────────────────────────────────────────────────────────────────*/
/** \brief Opaque generic pointer. */
typedef void *             pointer_T;

/* ─────────────────────────────────────────────────────────────────────────────
 * Mathematical Constants  (real32_T, single-precision)
 *
 * Central definitions for all EmbedSim modules.
 * Cast to (real32_T) for MISRA C:2012 Rule 10.4 type consistency.
 * real32_T is defined above - no additional include required.
 *
 * Naming:  ES_MATH_ prefix is project-wide and avoids collision with
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
 *
 * \note   Evaluates its argument more than once if the argument is an
 *         expression with side effects. Pass a plain variable.
 */
#define ES_CON_RPM_TO_RAD(RPM)             ((RPM * ES_MATH_2PI_F) / 60.0F)

/**
 * \brief  Conversion: rad/s to RPM
 *
 * \param[in] RAD  Speed in radians per second.
 *
 * \return  Speed in revolutions per minute.
 *
 * \note   Evaluates its argument more than once if the argument is an
 *         expression with side effects. Pass a plain variable.
 */
#define ES_CON_RAD_TO_RPM(RAD)             ((RAD * 60.0F) / ES_MATH_2PI_F)



/*********************************************************************************************************************/
/*------------------------------------------------common inline function --------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Wrap an angle in radians into the range [0.0, 2*pi).
 * \param   AnglePtr - Pointer to the angle value to wrap, in radians.
 *                     Modified in place. Must not be NULL.
 * \return  void
 *
 * \details Applies `fmodf(x, 2*pi)` to reduce the magnitude to (-2*pi, 2*pi),
 *          then adds `2*pi` when the result is negative. The output is always
 *          in the half-open interval [0.0, 2*pi), regardless of whether the
 *          input is positive, negative, or already inside the range.
 *
 *          Typical use: normalise a continuously growing or shrinking angle
 *          (e.g. an integrated rotor position, a commanded electrical angle,
 *          or a field-oriented-control angle) before it is fed to a sine /
 *          cosine lookup, a Park or Clarke transform, or a PWM modulator.
 *
 *          Behaviour examples (angles in radians):
 *          - Input  ` 0.0`          -> output ` 0.0`
 *          - Input  ` 3.0`          -> output ` 3.0`          (unchanged)
 *          - Input  ` 2*pi`         -> output ` 0.0`
 *          - Input  ` 2*pi + 0.5`   -> output ` 0.5`
 *          - Input  `-0.5`          -> output ` 2*pi - 0.5`
 *          - Input  `-2*pi - 0.5`   -> output ` 2*pi - 0.5`
 *          - Input  ` 100.0`        -> output ` 100.0 - 15*2*pi` ( ~ 5.75 )
 *
 * \note    Floating-point: uses `fmodf()` from `<math.h>`. For very large
 *          inputs, `fmodf` loses low-order bits as it truncates the quotient,
 *          so callers should keep the accumulated angle within a few
 *          revolutions of zero when possible (e.g. by subtracting a
 *          multiple of `2*pi` at each control-loop tick rather than letting
 *          it grow unbounded).
 *
 * \note    NaN / Inf: `fmodf(NaN, y)` and `fmodf(Inf, y)` both return NaN.
 *          The subsequent `NaN < 0.0F` comparison is false, so a NaN input
 *          propagates unchanged to the output. Callers that need to guard
 *          against non-finite inputs must do so before calling this helper.
 *
 * \note    MISRA C:2012 :
 *            - Rule  8.1 : explicit `void` return type.
 *            - Rule 14.4 : controlling expression (`*AnglePtr < 0.0F`) is
 *                          essentially Boolean.
 *            - Rule 15.5 : single exit point.
 *
 * \note    Rule 8.6 deviation: this helper is *defined* (not merely declared)
 *          in the header. It is `inline`, so each translation unit that uses
 *          it emits its own definition and the linker does not see a
 *          duplicate symbol. This is the standard C99 inline idiom and is
 *          accepted as a documented deviation for tiny leaf helpers.
 *
 * \warning `AnglePtr` must point to a valid, writable `real32_T`. A NULL
 *          pointer is undefined behaviour; this helper does not check for it
 *          (MISRA 18.4 keeps the signature free of non-constant pointer
 *          arithmetic, and defensive NULL checks are handled by the caller).
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
 * \param[in] MaxVal  Upper bound (inclusive). Must be >= MinVal.
 *
 * \return  Clamped value:
 *            - `MinVal` if `Val < MinVal`,
 *            - `MaxVal` if `Val > MaxVal`,
 *            - `Val`    otherwise.
 *
 * \details Saturating helper used wherever a physical quantity must be
 *          limited before it is written to a register, a reference input, or
 *          a state variable (e.g. current references, duty-cycle limits,
 *          speed commands). Preserves NaN by returning it unchanged, because
 *          both comparisons against NaN are false and the `else` branch is
 *          taken. Callers that must reject NaN should validate before
 *          calling.
 *
 *          Behaviour examples:
 *          - `EmbedSim_ClampValue( 5.0F,  0.0F, 10.0F)` -> ` 5.0F`
 *          - `EmbedSim_ClampValue(-3.0F,  0.0F, 10.0F)` -> ` 0.0F`
 *          - `EmbedSim_ClampValue(12.0F,  0.0F, 10.0F)` -> `10.0F`
 *          - `EmbedSim_ClampValue( 5.0F, 10.0F,  0.0F)` -> unspecified
 *            (MinVal > MaxVal is a caller error; this helper does not
 *            swap the bounds).
 *
 * \note    MISRA C:2012 :
 *            - Rule  8.1 : explicit `real32_T` return type.
 *            - Rule 10.4 : all operands and the return value are `real32_T`;
 *                          no implicit promotion to `double` occurs because
 *                          every literal in the caller is expected to be
 *                          `real32_T`-suffixed, matching the project
 *                          convention.
 *            - Rule 14.4 : controlling expressions are essentially Boolean.
 *            - Rule 15.5 : single exit point.
 *
 * \note    Rule 8.6 deviation: same as `EmbedSim_WrapAngleTwoPi` above - the
 *          function is `inline` and defined in the header by design.
 *
 * \see     EmbedSim_WrapAngleTwoPi
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

/*
 * sys_types.h
 * ===========
 *
 * Fixed-width type definitions for EmbedSim C code targeting Aurix TriCore.
 *
 * TARGET
 * ------
 *   Primary  : Infineon Aurix TriCore  (TASKING ctc compiler)
 *   Secondary: ARM Cortex-M4           (GCC / LLVM)
 *   Simulation: Windows / Linux        (via Cython wrapper)
 *
 * ORIGIN
 * ------
 *   Type definitions originally from Simulink Coder ERT target (R2020a)
 *   for Infineon TriCore. Adopted as the EmbedSim standard type header
 *   to guarantee identical type sizes across simulation and firmware.
 *
 * DEVICE CHARACTERISTICS
 * ----------------------
 *   Device     : Infineon TriCore
 *   Byte order : Little Endian
 *   char  :  8 bit     short :  16 bit
 *   int   : 32 bit     long  :  32 bit     long long : 64 bit
 *
 * Author : EmbedSim Framework
 * Version: 1.0.0
 */

#ifndef RTWTYPES_H
#define RTWTYPES_H

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
 *
 *   real32_T — 32-bit IEEE 754, native to TriCore FPU.
 *              Used exclusively in FOC inner loop (Clarke, Park, PI).
 *   real64_T — 64-bit IEEE 754, for offline analysis only.
 *              Never use in embedded control loops.
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

#endif /* RTWTYPES_H */

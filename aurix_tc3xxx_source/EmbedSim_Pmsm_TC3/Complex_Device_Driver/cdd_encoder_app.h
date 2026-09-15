/**********************************************************************************************************************
 * \file      cdd_encoder_app.h
 * \brief     Incremental Encoder Driver with 4x Decoding for Motor Control Applications
 *
 * \details   Provides a complete incremental encoder interface using the Infineon GPT12
 *            module in Incremental Interface Mode (Mode 6). Features:
 *
 *            - **4x Decoding**: Quadrature decoding on both edges of T3IN and T3EUD
 *            - **Zero-Index (Z) Pulse**: Hardware-cleared T3 for exact single-turn angle
 *            - **Direction Detection**: Hardware direction tracking via quadrature decode
 *            - **Pulse-count Speed**: signed per-tick diff, converted to rad/s in FLOAT
 *              and then smoothed by a first-order IIR low-pass
 *            - **Absolute Multi-turn Position**: TurnCount * N +/- T3, computed on demand
 *
 *            ## Design Note - T5 / time-diff mode removed
 *            The earlier dual-mode speed (T5 CAPREL / time-diff) has been removed.
 *            On this target the T5SC / T5CLR capture bits were not producing a
 *            CAPREL update on encoder edges, so any tick with diff <= 10 fell into
 *            the CAPREL branch, read 0, and held SpeedRad at 0. Pulse-count mode
 *            is now used unconditionally.
 *
 *            ## Design Note - Direct per-tick delta (no moving average)
 *            The raw signed count delta `diff` for the current 20 kHz tick is
 *            converted straight to rad/s:
 *
 *                rawSpeed = (real32_T)diff * T3SpeedConversionQuotient
 *
 *            where T3SpeedConversionQuotient = 2*pi / (EncoderResolution * UpdatePeriod).
 *            No integer division is performed anywhere on the speed path, so the
 *            historical "800 expected, 600 observed" truncation defect cannot
 *            occur: the conversion is a single float multiply of a signed integer.
 *
 *            The previous round-buffer moving-average on `diff` has been removed.
 *            It is no longer needed once the divide-by-N is gone.
 *
 *            ## Design Note - Speed IIR low-pass
 *            Per-tick speed is quantisation-noisy: at 800 RPM and 4000 CPR the
 *            delta dithers {3,3,2,...}. A first-order IIR is therefore applied
 *            in the SpeedRad domain before publication:
 *
 *                SpeedRad = alpha * rawSpeed + (1 - alpha) * SpeedRad
 *
 *            with alpha = SPEED_LPF_ALPHA (0.0589), which yields approximately
 *            a 600 Hz cutoff at the 20 kHz update rate. Downstream consumers
 *            receive this already-filtered value from CddEncoder_GetSpeedRad()
 *            and CddEncoder_GetSpeedRpm(); they are not expected to re-filter.
 *            RotorAngle is not filtered - it is derived directly from T3 and is
 *            exact to one count.
 *
 *            ## Design Note - T3 is a 16-bit counter
 *            T3 lives in the full 16-bit space even with CLRT3EN = 1. Over one
 *            mechanical revolution it visits:
 *                CW  : 0x0000 .. 0x0F9F   ( 0 .. +3999 )
 *                ACW : 0xF061 .. 0xFFFF   ( -3999 .. -1 as int16_T )
 *            The only safe way to compute a delta is to cast T3 to int16_T and do
 *            the subtraction in int32_T. "T3 % 4000" is NOT valid, because 65536
 *            is not a multiple of 4000: 65535 % 4000 == 1535, not 3999.
 *
 *            ## Design Note - Index-Reset Architecture
 *            The Z-index pulse hardware-clears T3 to zero once per revolution.
 *            T3CounterPrev is latched every update, including the Z-skip tick, so
 *            the tick after a Z-event computes its delta from the post-Z position.
 *
 *            ## Corner case - wrap-around at Z-boundary
 *            Because T3 is hardware-cleared by T4 on every Z-pulse, the raw
 *            delta between two consecutive ticks can appear huge even though
 *            the physical movement was small:
 *
 *                CW  : prev=3998, new=6    -> raw diff = 6 - 3998    = -3992
 *                ACW : prev=6,    new=3994 -> raw diff = 3994 - 6    = +3988
 *
 *            Both are corrected by comparing |diff| against half the encoder
 *            resolution and adding/subtracting ENCODER_COUNTS_PER_REV. The
 *            corrected values are +8 (CW) and -12 (ACW) respectively.
 *
 *            ## Concurrency
 *            Encoder_Index_ISR updates TurnCount asynchronously to the readers
 *            of EncoderState_G. No function in this driver reads TurnCount, so
 *            no internal race exists. Any external reader MUST protect its
 *            read with a critical section: 64-bit accesses are not atomic on
 *            TriCore.
 *
 *            ## Usage Example
 *            ```c
 *            CddEncoder_Init();
 *            // In 20 kHz control ISR:
 *            CddEncoder_Update();
 *            real32_T speed    = CddEncoder_GetSpeedRad();  // already IIR-filtered
 *            real32_T position = CddEncoder_GetRotorPosition();
 *            ```
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.1 : Functions have explicit return types.
 *              - Rule  8.5 : One declaration per identifier.
 *              - Rule  8.6 : No definitions in header files.
 *              - Rule  8.7 : Internal linkage for static helpers (in .c file).
 *              - Rule  8.9 : No file-scope object with internal linkage in the
 *                            .c file. EncoderState_G has external linkage by
 *                            design and is declared extern below.
 *              - Rule 13.1 : No initializer lists with side effects.
 *              - Rule 13.2 : No persistent side effects in assignment RHS.
 *              - Rule 14.4 : Controlling expressions are essentially Boolean.
 *              - Rule 15.5 : Single exit point per function.
 *              - Rule 17.2 : No recursion.
 *              - Rule 18.4 : No non-constant pointer arithmetic.
 *
 *              Deviations, flagged inline in the .c file:
 *              - Rule 10.3 : uint16_T -> int16_T and int32_T -> real32_T
 *                            narrowing casts on the delta / angle paths.
 *              - Rule 10.4 : mixed int32_T / real32_T arithmetic at the
 *                            speed conversion and IIR update.
 *              - Rule 10.5 : casts between signed and unsigned integer types.
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase
 *              - Output pointers: PascalCasePtr
 *              - Local variables: Lower camelCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   2.8.1
 * \date      2026-09-13
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim - EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/

#ifndef COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_

#include "embed_sim_sys_types.h"
#include "embed_sim_compiler.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Encoder resolution in physical lines per revolution. */
#define ENCODER_RESOLUTION                  (1000U)

/** \brief Quadrature decoding factor. */
#define ENCODER_DECODING_FACTOR             (4U)

/** \brief Effective counts per mechanical revolution. */
#define ENCODER_COUNTS_PER_REV              (ENCODER_RESOLUTION * ENCODER_DECODING_FACTOR)

/** \brief Radians per encoder count. */
#define ENCODER_COUNTS_TO_RAD               (ES_MATH_2PI_F / (real32_T)ENCODER_COUNTS_PER_REV)

/** \brief Encoder update frequency in Hz. */
#define ENCODER_UPDATE_FREQ_HZ              (20000.0F)

/** \brief Encoder update period in seconds. */
#define ENCODER_UPDATE_PERIOD               (1.0F / ENCODER_UPDATE_FREQ_HZ)

/** \brief rad/s to RPM conversion factor. */
#define ENCODER_RAD_PER_SEC_TO_RPM          (60.0F / ES_MATH_2PI_F)

/** \brief Speed filter coefficients.
 *         First-order IIR low-pass applied to SpeedRad inside CddEncoder_Update().
 *         Alpha = 0.0589 gives an approximately 600 Hz cutoff at the 20 kHz
 *         update rate. Downstream consumers receive the filtered value and are
 *         not expected to re-filter. */
#define SPEED_LPF_ALPHA                     (0.0589F)                    /**< New-data weight    */
#define SPEED_LPF_ONE_MINUS_ALPHA           (1.0F - SPEED_LPF_ALPHA)     /**< Old-data weight    */

/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/

/**
 * \brief Encoder direction enumeration.
 *
 * \details GPT12 T3RDIR is mapped directly:
 *          0 = CW, 1 = ACW.
 */
typedef enum
{
    ENC_DIR_CW  = 0x0U,
    ENC_DIR_ACW = 0x1U
} Encoder_Direction_T;

/**
 * \brief Encoder runtime state.
 *
 * \warning TurnCount is written from Encoder_Index_ISR without a critical
 *          section. It is not read inside this driver. Any external reader
 *          MUST protect the read against the ISR, because 64-bit accesses
 *          are not atomic on TriCore.
 */
typedef struct
{
    real32_T    SpeedRad;            /**< IIR-filtered signed speed, rad/s          */
    real32_T    SpeedRpm;            /**< IIR-filtered signed speed, RPM            */
    real32_T    RotorAngle;          /**< Mechanical angle in [0, 2*pi) rad         */

    /*
     * Raw 16-bit T3 snapshot. Always interpret as int16_T before calculating
     * a signed delta.
     */
    uint16_T    T3CounterPrev;

    /*
     * Set by the Z ISR. The next update synchronizes T3CounterPrev to the
     * post-Z counter and suppresses the reset-spanning speed delta.
     */
    uint32_T    ZEventPending;

    real32_T    T3SpeedConversionQuotient;
    real32_T    CountsToRadians;
    uint32_T    EncoderResolution;
    real32_T    UpdatePeriod;

    /*
     * Revolution count. Positive = CW, negative = ACW according to the
     * selected GPT12 direction convention. Written only from the Z ISR;
     * 64-bit reads from other contexts are not atomic.
     */
    int64_T     TurnCount;

    uint32_T    Direction;
    uint32_T    Initialized;

} CddEncoder_State_T;

/**********************************************************************************************************************
 * Global Variables
 *********************************************************************************************************************/

/** \brief Global encoder state instance. */
extern CddEncoder_State_T EncoderState_G;

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/** \brief Initialize encoder hardware and software state. */
extern uint32_T CddEncoder_Init(void);

/**
 * \brief Update encoder state.
 * \details Call at exactly 20 kHz for the configured speed conversion and
 *          IIR filter coefficient.
 */
extern void CddEncoder_Update(void);

/** \brief Reset encoder position and runtime state. */
extern void CddEncoder_Reset(void);

/** \brief Get mechanical rotor position in [0, 2*pi) rad. */
extern real32_T CddEncoder_GetRotorPosition(void);

/**
 * \brief Get signed angular velocity in rad/s.
 * \details Positive = CW, negative = ACW. The returned value has already been
 *          smoothed by the SPEED_LPF_ALPHA first-order IIR; callers do not
 *          need to re-filter.
 */
extern real32_T CddEncoder_GetSpeedRad(void);

/**
 * \brief Get signed angular velocity in RPM.
 * \details Positive = CW, negative = ACW. Shares the IIR-filtered SpeedRad
 *          source with CddEncoder_GetSpeedRad().
 */
extern real32_T CddEncoder_GetSpeedRpm(void);

/** \brief Get current direction: ENC_DIR_CW or ENC_DIR_ACW. */
extern uint32_T CddEncoder_GetDirection(void);

/** \brief Return 1 when initialized, otherwise 0. */
extern uint32_T CddEncoder_IsInitialized(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_ */

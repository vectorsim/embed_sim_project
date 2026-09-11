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
 *            - **Speed Calculation**: Time-based velocity estimate with IIR low-pass
 *            - **Absolute Position**: TurnCount * N +/- T3, exact at every instant
 *
 *            ## Hardware Configuration
 *            - **T3**: Core encoder counter in Incremental Interface Mode (T3M = 0x6).
 *                        Hardware-cleared at each Z-pulse (CLRT3EN = 1) so it directly
 *                        encodes the mechanical angle within one revolution.
 *            - **T4**: Index pulse capture on rising edge (Z-signal). Resets T3 and
 *                        fires the ISR that updates TurnCount.
 *
 *            ## Design Note - Index-Reset Architecture
 *            The Z-index pulse hardware-clears T3 to zero once per revolution. This
 *            makes the mechanical angle exactly:
 *                angle_rad = T3 * (2*pi / EncoderResolution)
 *            with no integrator, no IIR lag, and no float drift on the angle path.
 *            The only cost is that the speed delta across the reset tick must be
 *            discarded once per revolution; the Z-ISR sets ZEventPending so the next
 *            update handles this cleanly.
 *
 *            ## Usage Example
 *            ```c
 *            // Initialization (call once at system startup)
 *            CddEncoder_Init();
 *
 *            // Main control loop (call at 20 kHz)
 *            CddEncoder_Update();
 *            real32_T speed    = CddEncoder_GetSpeedRad();
 *            real32_T position = CddEncoder_GetRotorPosition();
 *            int64_T  absolute = CddEncoder_GetAbsolutePositionCounts();
 *            ```
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier.
 *              - Rule  8.6 : No definitions in header files.
 *              - Rule 17.2 : No recursion.
 *              - Rule 14.7 : Single return point.
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
 * \version   2.3.0
 * \date      2026-09-11
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim - EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/

#include "embed_sim_sys_types.h"
#include "embed_sim_compiler.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Encoder resolution in lines per revolution (PPR).
 *         Physical lines on the encoder disk. A 1000 PPR encoder produces
 *         1000 A/B cycles per mechanical revolution.
 *         MISRA 20.x : literal carries an explicit unsigned suffix. */
#define ENCODER_RESOLUTION                  (1000U)

/** \brief 4x decoding factor.
 *         Quadrature decoding multiplies resolution by 4 by detecting both
 *         rising and falling edges on both A and B channels. */
#define ENCODER_DECODING_FACTOR             (4U)

/** \brief Counts per revolution.
 *         For 1000 PPR with 4x decoding: 1000 x 4 = 4000 counts/rev. */
#define ENCODER_COUNTS_PER_REV              (ENCODER_RESOLUTION * ENCODER_DECODING_FACTOR)

/** \brief Radians per encoder count.
 *         Used to convert the raw T3 value directly into a mechanical angle. */
#define ENCODER_COUNTS_TO_RAD             (ES_MATH_2PI_F / (real32_T)ENCODER_COUNTS_PER_REV)

/** \brief Update frequency in Hz (20 kHz).
 *         The encoder state is updated at this rate. MISRA 20.x : float suffix. */
#define ENCODER_UPDATE_FREQ_HZ              (20000.0F)

/** \brief Update period in seconds (50 us).
 *         Time between consecutive encoder updates; matches the control loop. */
#define ENCODER_UPDATE_PERIOD               (1.0F / ENCODER_UPDATE_FREQ_HZ)

/** \brief Speed filter coefficients.
 *         First-order IIR low-pass for velocity estimation. Alpha = 0.0589
 *         gives a cutoff of approximately 600 Hz at 20 kHz update rate.
 *         The filter is sign-symmetric: it does not distinguish CW from ACW. */
#define SPEED_LPF_ALPHA                     (0.0589F)
#define SPEED_LPF_ONE_MINUS_ALPHA           (1.0F - SPEED_LPF_ALPHA)

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Encoder direction enumeration (matches ILLD convention).
 *          T3RDIR bit from GPT12_T3CON: 0 = CW, 1 = ACW.
 *          MISRA 8.5 : one declaration per identifier.
 */
typedef enum
{
    ENC_DIR_CW  = 0x0U,       /**< Clockwise direction (forward)        */
    ENC_DIR_ACW = 0x1U,       /**< Anti-clockwise direction (reverse)   */
} Encoder_Direction_T;

/**
 * \brief   Encoder state structure.
 *          Holds all runtime data for the encoder driver.
 */
typedef struct
{
    real32_T    SpeedRad;                    /**< Filtered angular velocity [rad/s], signed             */
    real32_T    SpeedRpm;                    /**< Filtered angular velocity [RPM], signed               */
    real32_T    RotorAngle;                  /**< Mechanical rotor position in [0.0, 2*pi) [rad],
                                                  computed directly from T3 (no integrator)              */

    uint16_T    T3CounterPrev;               /**< Snapshot of the 16-bit T3 counter from the previous
                                                  update. MUST be uint16_T: the delta is computed
                                                  as a 16-bit modular subtraction, which is what
                                                  makes the 0xFFFF <-> 0x0000 wrap transparent in
                                                  both directions.                                        */

    uint32_T    ZEventPending;               /**< Set by the Z-ISR, cleared by the update. Causes
                                                  the update to skip one delta computation because
                                                  T3 was hardware-cleared.                                */

    real32_T    T3SpeedConversionQuotient;   /**< Constant: (2*pi) / (EncoderResolution * UpdatePeriod),
                                                  i.e. radians per count per update                       */

    real32_T    CountsToRadians;             /**< Constant: 2*pi / EncoderResolution, used to convert
                                                  the raw T3 value to a mechanical angle directly         */

    uint32_T    EncoderResolution;           /**< Cached encoder resolution (4000 counts/rev)           */
    real32_T    UpdatePeriod;                /**< Cached update period (50 us)                          */

    int64_T     TurnCount;                   /**< Number of complete revolutions (positive = CW,
                                                  negative = ACW). Updated exclusively by the
                                                  Z-index ISR.                                            */

    uint32_T    Direction;                   /**< Latest direction (ENC_DIR_CW / ENC_DIR_ACW) from
                                                  T3RDIR. Written by both the Z-ISR and the
                                                  periodic update.                                        */

    uint32_T    Initialized;                 /**< Initialization flag (0 = uninit, 1 = initialized)     */
} CddEncoder_State_T;

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Global encoder state instance.
 *         Exposed for debug access and because the Z-index ISR updates the
 *         turn count directly.
 *         MISRA 8.4 : definition lives in cdd_encoder_app.c; only a declaration
 *         is placed here. */
extern CddEncoder_State_T EncoderState_G;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize the encoder hardware and state.
 * \return  uint32_T - 1 if successful, 0 if already initialized.
 *
 * \details Steps:
 *          1. Reset the software state structure.
 *          2. Enable the GPT12 module clock.
 *          3. Configure T3 for Incremental Interface Mode; T4 clears T3 on Z-pulse.
 *          4. Configure T4 for Z-index capture with CLRT3EN = 1.
 *          5. Set up the Z-index interrupt.
 *          6. Start T3.
 *
 * \note    Safe to call multiple times; subsequent calls are no-ops.
 */
extern uint32_T CddEncoder_Init(void);

/**
 * \brief   Update encoder state.
 * \return  void
 *
 * \details Must be called at the configured update rate (20 kHz). Reads the T3
 *          counter, computes the signed delta using 16-bit modular subtraction
 *          (skipping the tick immediately after a Z-event), applies the IIR
 *          low-pass to the raw speed, and computes the mechanical angle directly
 *          from T3.
 *
 * \warning Failure to call this function at the correct rate yields incorrect
 *          speed values (the angle path itself is instantaneous).
 */
extern void CddEncoder_Update(void);

/**
 * \brief   Reset encoder position and state.
 * \return  void
 *
 * \details Stops T3, clears the counter, resets all software state (including the
 *          ZEventPending flag), and restarts T3. Useful for homing sequences or
 *          error recovery.
 *
 * \note    The encoder must be initialized before calling. T3 is temporarily
 *          stopped during the reset to ensure atomic counter updates.
 */
extern void CddEncoder_Reset(void);

/**
 * \brief   Get mechanical rotor position.
 * \return  real32_T - Position in radians [0.0, 2*pi).
 *
 * \details Read directly from T3 in the last update; no filter lag, no drift.
 */
extern real32_T CddEncoder_GetRotorPosition(void);

/**
 * \brief   Get angular velocity in rad/s.
 * \return  real32_T - Filtered speed in rad/s. Sign indicates direction:
 *                     positive = CW, negative = ACW.
 */
extern real32_T CddEncoder_GetSpeedRad(void);

/**
 * \brief   Get angular velocity in RPM.
 * \return  real32_T - Filtered speed in RPM. Sign indicates direction:
 *                     positive = CW, negative = ACW.
 */
extern real32_T CddEncoder_GetSpeedRpm(void);

/**
 * \brief   Get direction.
 * \return  uint32_T - ENC_DIR_CW (0) or ENC_DIR_ACW (1).
 */
extern uint32_T CddEncoder_GetDirection(void);


/**
 * \brief   Check if encoder is initialized.
 * \return  uint32_T - 1 if initialized, 0 if not.
 */
extern uint32_T CddEncoder_IsInitialized(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_ENCODER_APP_H_ */

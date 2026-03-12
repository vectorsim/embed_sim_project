/*
 * pi_buck_controller.h
 * =====================
 *
 * PURPOSE
 * -------
 * This header file defines the data structures and function prototypes for the
 * PI (Proportional-Integral) voltage controller used in the Buck Converter.
 *
 * A HEADER FILE (.h) contains DECLARATIONS — it tells other C files what
 * exists and what shape everything has, but contains no executable code itself.
 * The actual code lives in pi_buck_controller.c.
 *
 * WHAT IS A BUCK CONVERTER?
 * -------------------------
 * A buck converter is a DC-DC power supply that steps a higher voltage DOWN
 * to a lower voltage. Example: 24V battery → 12V output rail.
 * It works by rapidly switching a transistor on/off (PWM) and using an
 * inductor + capacitor to filter the result into a smooth DC voltage.
 *
 * WHAT IS A PI CONTROLLER?
 * -------------------------
 * A PI controller is the most common feedback controller in power electronics.
 * It continuously measures the error between what we want (V_ref) and what
 * we have (V_meas), then adjusts the control output (duty cycle) to drive
 * that error to zero.
 *
 *   "P" (Proportional): react immediately to the current error.
 *                        Output ∝ error right now.
 *   "I" (Integral):      accumulate past errors over time.
 *                        Drives steady-state error to exactly zero.
 *
 * POSITION IN CONTROL CHAIN
 * -------------------------
 *
 *   V_ref  ──► ┌─────────┐           ┌──────────────────┐
 *              │ PI Buck │──► duty ──►│ Buck Converter   │──► V_out
 *   V_meas ──► └─────────┘           │ Plant (Modelica  │
 *      ▲                             │ FMU or hardware) │
 *      └─────────────────────────────┘  (feedback)
 *
 * ALGORITHM (discrete-time, forward Euler integration)
 * ---------
 *   e(k)      = V_ref(k) - V_meas(k)                          ← error this step
 *   integ(k)  = clamp( integ(k-1) + e(k)*dt,                  ← accumulated error
 *                       -duty_max/Ki,  +duty_max/Ki )          ← anti-windup clamp
 *   duty(k)   = clamp( Kp*e(k) + Ki*integ(k),                 ← PI output
 *                       duty_min,  duty_max )                  ← output clamp
 *
 * WHAT IS ANTI-WINDUP?
 * --------------------
 * Without a limit, the integrator can accumulate to enormous values when the
 * output is saturated (e.g., duty = 0.95 maximum but error is still large).
 * When the error eventually reverses, the integrator takes a long time to
 * "unwind" back to a useful range — causing a slow, sluggish response.
 * The clamp ±duty_max/Ki prevents this: it limits the integrator so that
 * Ki * integ can never exceed duty_max, keeping the controller responsive.
 *
 * TARGET PLATFORMS
 * ----------------
 *   Primary  : Infineon Aurix TriCore  (TASKING ctc compiler)
 *   Secondary: ARM Cortex-M4           (GCC / LLVM)
 *   Simulation: Windows / Linux        (called via Cython wrapper from Python)
 *
 * @author EmbedSim Framework
 * @version 1.0.0
 * @date 2025
 */

/*
 * INCLUDE GUARD
 * -------------
 * These three lines (#ifndef / #define / #endif) prevent the file from being
 * included MORE THAN ONCE in the same compilation. Without them, if two
 * different .c files both #include this header, the compiler would see the
 * struct definitions twice and throw an error.
 *
 * Convention: use the filename in ALL_CAPS with dots replaced by underscores.
 */
#ifndef PI_BUCK_CONTROLLER_H
#define PI_BUCK_CONTROLLER_H

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

/*
 * Sys_Types.h defines our custom fixed-width types:
 *   real32_T  → float  (32-bit IEEE 754 — native to TriCore FPU)
 *   uint8_T   → unsigned char
 *   int32_T   → int
 *   etc.
 *
 * WHY NOT JUST USE "float"?
 * On some compilers "float" might be 64-bit. real32_T is GUARANTEED to be
 * exactly 32 bits on every target — critical for hardware register mapping
 * and for matching the Simulink Coder ERT types used in production firmware.
 */
#include "Sys_Types.h"    /* real32_T, uint8_T */

/******************************************************************************/
/*----------------------------- Data Structures ------------------------------*/
/******************************************************************************/

/*
 * ─────────────────────────────────────────────────────────────────────────────
 * STRUCT PATTERN (EmbedSim standard)
 * ─────────────────────────────────────────────────────────────────────────────
 * Every EmbedSim C block uses THREE structs:
 *
 *   _Params_T  — tuning constants (written ONCE at startup, never change at runtime)
 *   _State_T   — internal memory (updated EVERY control step)
 *   _Block_T   — the complete block: contains a Params and a State nested inside
 *
 * Plus two more for data flow:
 *   _Input_T   — signals fed INTO the block this step
 *   _Output_T  — signals coming OUT of the block this step
 *
 * This pattern keeps the function signature clean: pass pointers to these
 * structs instead of a long list of individual arguments.
 * ─────────────────────────────────────────────────────────────────────────────
 */

/**
 * @brief  PI Buck controller parameters — TUNING CONSTANTS.
 *
 * These values are set once at startup (by PI_Buck_SetParams) and do NOT
 * change during normal operation. They are stored inside PI_Buck_Block_T.
 *
 * Memory: 5 × 4 bytes = 20 bytes total.
 */
typedef struct
{
    real32_T Kp;       /**< Proportional gain.
                         *  Units: duty_cycle_change / volt_of_error  [1/V]
                         *  Effect: larger Kp → faster response, but too large
                         *          causes oscillation or instability.
                         *  Typical: 0.1 – 0.3 for this buck circuit.         */

    real32_T Ki;       /**< Integral gain.
                         *  Units: duty_cycle / (volt × second)  [1/(V·s)]
                         *  Effect: drives steady-state error to zero.
                         *          Too large → integrator oscillates.
                         *  Typical: 5.0 – 20.0 for this buck circuit.        */

    real32_T duty_max; /**< Maximum allowed duty cycle  [dimensionless, 0–1].
                         *  0.95 means the switch is ON for 95% of each period.
                         *  Safety ceiling — prevents switch from being always ON
                         *  which would remove the buck action entirely.       */

    real32_T duty_min; /**< Minimum allowed duty cycle  [dimensionless, 0–1].
                         *  0.05 means the switch is ON for at least 5%.
                         *  Prevents the converter entering discontinuous
                         *  conduction mode (DCM) during light load.           */

    real32_T Ts;       /**< Default sample period  [seconds].
                         *  Used only if the caller passes dt ≤ 0 to Compute().
                         *  At runtime, EmbedSim always supplies dt directly,
                         *  so this is a fallback / initialisation value.
                         *  0.0001 = 100 µs = 10 kHz control loop.            */
} PI_Buck_Params_T;


/**
 * @brief  PI Buck controller run-time state — MEMORY BETWEEN STEPS.
 *
 * These values change every time PI_Buck_Compute() is called.
 * They represent the controller's "memory" from one step to the next.
 *
 * Memory: 3 × 4 bytes = 12 bytes total.
 */
typedef struct
{
    real32_T integrator;   /**< The accumulated integral of the error.
                             *  This is what makes the "I" in PI.
                             *  It grows as long as error ≠ 0, and shrinks
                             *  when the error reverses sign.
                             *  Clamped by anti-windup to ±duty_max/Ki.       */

    real32_T prev_error;   /**< The error value from the previous step.
                             *  Not used in the PI calculation itself —
                             *  stored here for debugging and oscilloscope
                             *  inspection only. Could be removed in a
                             *  minimal firmware build.                        */

    real32_T last_output;  /**< The duty cycle produced at the last step.
                             *  Stored for debugging and for the Cython
                             *  .last_output property. Not needed by the
                             *  PI algorithm itself.                           */
} PI_Buck_State_T;


/**
 * @brief  Complete PI Buck block — the "object" that represents one controller.
 *
 * This is the struct you declare ONE instance of:
 *     PI_Buck_Block_T  my_controller;
 *
 * You never access params or state directly from application code —
 * always use the provided functions (Init, SetParams, Compute, Reset).
 * This is the C equivalent of encapsulation in object-oriented languages.
 */
typedef struct
{
    PI_Buck_Params_T params;  /**< Tuning constants (Kp, Ki, duty limits, Ts) */
    PI_Buck_State_T  state;   /**< Run-time memory (integrator, prev_error)   */
} PI_Buck_Block_T;


/**
 * @brief  Input signals for ONE compute step.
 *
 * Bundle the two input signals into a struct so the function signature
 * stays clean. Pass a pointer to this struct to PI_Buck_Compute().
 */
typedef struct
{
    real32_T V_ref;   /**< Desired output voltage  [Volts].
                        *  This is the SETPOINT — what we want V_out to be.
                        *  Example: 12.0f for a 12V output.                   */

    real32_T V_meas;  /**< Measured output voltage  [Volts].
                        *  This is the FEEDBACK — what V_out actually is.
                        *  Comes from the FMU plant or a real ADC reading.    */
} PI_Buck_Input_T;


/**
 * @brief  Output signals for ONE compute step.
 *
 * The only output of the PI controller is the duty cycle command sent to
 * the buck converter's PWM generator.
 */
typedef struct
{
    real32_T duty;    /**< PWM duty cycle  [dimensionless, 0.0 – 1.0].
                        *  0.0 = switch always OFF → V_out → 0V
                        *  1.0 = switch always ON  → V_out → V_in (24V)
                        *  0.5 = 50% → V_out ≈ 12V (ideal, lossless)
                        *  In practice clamped to [duty_min, duty_max].       */
} PI_Buck_Output_T;


/******************************************************************************/
/*------------------------ Function Prototypes --------------------------------*/
/******************************************************************************/

/*
 * WHAT IS A FUNCTION PROTOTYPE?
 * ------------------------------
 * A prototype tells the compiler "this function exists, here are its
 * argument types and return type." The actual implementation is in the .c file.
 * Any .c file that #includes this header can then CALL these functions safely,
 * because the compiler knows what to expect.
 *
 * The "extern" keyword is redundant here (global functions are extern by
 * default) but makes the intent explicit: these functions are implemented
 * in ANOTHER translation unit (pi_buck_controller.c).
 */

/**
 * @brief  Initialise PI Buck block with safe default parameters.
 *
 * MUST be called before the first call to PI_Buck_Compute().
 * Sets default gains (Kp=0.1, Ki=5.0) and zeros the integrator.
 * You will normally follow this with PI_Buck_SetParams() to override
 * the defaults with your tuned values.
 *
 * @param[out] pPI  Pointer to the block to initialise.
 *                  "[out]" means this function WRITES to it.
 */
extern void PI_Buck_Init(PI_Buck_Block_T* pPI);


/**
 * @brief  Set PI Buck parameters after initialisation.
 *
 * Call this after PI_Buck_Init() to apply your tuned gains.
 * The function validates the inputs (guards against Ki=0, Ts=0,
 * duty_min > duty_max) so you do not need to pre-validate them.
 *
 * @param[out] pPI       Pointer to the block.
 * @param[in]  Kp        Proportional gain     [1/V]
 * @param[in]  Ki        Integral gain          [1/(V·s)]   must be > 0
 * @param[in]  duty_max  Maximum duty cycle     [0–1]
 * @param[in]  duty_min  Minimum duty cycle     [0–1]       must be ≤ duty_max
 * @param[in]  Ts        Default sample period  [s]         must be > 0
 */
extern void PI_Buck_SetParams(PI_Buck_Block_T* pPI,
                              real32_T         Kp,
                              real32_T         Ki,
                              real32_T         duty_max,
                              real32_T         duty_min,
                              real32_T         Ts);


/**
 * @brief  Reset integrator and state to zero.
 *
 * Call this when:
 *   • The converter is disabled and re-enabled (bumpless transfer).
 *   • A fault condition clears and normal operation resumes.
 *   • The reference changes so abruptly that old integral wind-up
 *     would cause an undesirable transient.
 *
 * Does NOT reset the parameters (Kp, Ki, etc.) — only the state
 * (integrator, prev_error, last_output).
 *
 * @param[out] pPI  Pointer to the block to reset.
 */
extern void PI_Buck_ResetState(PI_Buck_Block_T* pPI);


/**
 * @brief  Execute ONE step of the PI control algorithm.
 *
 * This is the function called from your real-time interrupt (ISR) or
 * from EmbedSim's simulation loop every control period.
 *
 * Call sequence (typical embedded firmware):
 *
 *   void PWM_ISR(void)              // fires every 100 µs
 *   {
 *       PI_Buck_Input_T  in;
 *       PI_Buck_Output_T out;
 *
 *       in.V_ref  = target_voltage;
 *       in.V_meas = ADC_ReadVoltage();
 *
 *       PI_Buck_Compute(&g_pi_block, &in, 0.0001f, &out);
 *
 *       PWM_SetDuty(out.duty);
 *   }
 *
 * @param[in,out] pPI   Block pointer. "[in,out]" = function reads AND writes it.
 *                      Reads: params (Kp, Ki, duty limits), state.integrator
 *                      Writes: state.integrator, state.prev_error, state.last_output
 * @param[in]     pIn   Input struct — V_ref and V_meas for THIS step.
 *                      Declared "const" because Compute() never modifies inputs.
 * @param[in]     dt    Time step for this call  [seconds].
 *                      If dt > 0, it overrides params.Ts.
 *                      If dt = 0 (or negative), params.Ts is used instead.
 *                      Passing dt explicitly allows EmbedSim's RK4 solver to
 *                      use variable step sizes without touching the stored Ts.
 * @param[out]    pOut  Output struct — filled with the duty cycle command.
 */
extern void PI_Buck_Compute(PI_Buck_Block_T*       pPI,
                            const PI_Buck_Input_T* pIn,
                            real32_T               dt,
                            PI_Buck_Output_T*      pOut);

#endif /* PI_BUCK_CONTROLLER_H */
/* ── End of pi_buck_controller.h ─────────────────────────────────────────── */
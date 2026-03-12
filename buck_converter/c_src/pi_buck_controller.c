/*
/*
 * pi_buck_controller.c
 * =====================
 *
 * PURPOSE
 * -------
 * This is the IMPLEMENTATION file for the PI Buck voltage controller.
 * It contains the actual executable C code for every function declared
 * in pi_buck_controller.h.
 *
 * RULE: This file #includes only its own header. It has no global variables,
 * no malloc(), no printf(), no floating-point library functions — it is
 * completely "bare metal" clean and can be compiled for Aurix TriCore,
 * ARM Cortex-M4, or x86 without any changes.
 *
 * ALGORITHM RECAP (see header for full explanation)
 * ---------
 *   e(k)      = V_ref(k) - V_meas(k)
 *   integ(k)  = clamp( integ(k-1) + e(k)*dt,  -duty_max/Ki,  +duty_max/Ki )
 *   duty(k)   = clamp( Kp*e(k) + Ki*integ(k),  duty_min,      duty_max    )
 *
 * @author EmbedSim Framework
 * @version 1.0.0
 * @date 2025
 */

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

/*
 * Only one include needed: our own header.
 * That header already pulls in Sys_Types.h (for real32_T).
 * NEVER include more than you need — every extra header increases
 * compile time and risks name collisions on embedded targets.
 */
#include "pi_buck_controller.h"

/******************************************************************************/
/*-----------------------------Static Helpers---------------------------------*/
/******************************************************************************/

/*
 * pi_clamp()
 * ----------
 * Clamps (clips) a value to the range [lo, hi].
 *
 * WHAT DOES "static" MEAN HERE?
 * "static" on a file-scope function means it is PRIVATE to this .c file.
 * No other .c file can see or call pi_clamp(). It is an internal helper.
 * This is good practice — it prevents name clashes and clarifies intent.
 *
 * WHAT DOES "inline" MEAN?
 * "inline" asks the compiler to insert the function body directly at each
 * call site instead of generating a real function call instruction. For a
 * tiny 3-line function called inside a real-time ISR, this avoids the
 * push/pop overhead of a function call. The compiler may ignore the hint
 * if it decides the trade-off isn't worth it.
 *
 * WHY NOT USE fminf() / fmaxf() FROM <math.h>?
 * On some embedded targets, math.h functions involve library calls that
 * pull in floating-point support routines — increasing code size. This
 * hand-written version uses only comparisons, which compile to a single
 * conditional move instruction on TriCore and Cortex-M4.
 *
 * Examples:
 *   pi_clamp(1.2f, 0.0f, 1.0f) → 1.0f   (too high, clamped to hi)
 *   pi_clamp(-0.1f, 0.0f, 1.0f) → 0.0f  (too low, clamped to lo)
 *   pi_clamp(0.5f, 0.0f, 1.0f) → 0.5f   (in range, unchanged)
 */
static inline real32_T pi_clamp(real32_T val, real32_T lo, real32_T hi)
{
    if (val > hi) return hi;   /* over the ceiling → return ceiling */
    if (val < lo) return lo;   /* under the floor  → return floor   */
    return val;                /* in range         → return as-is   */
}

/******************************************************************************/
/*-------------------------Function Implementations---------------------------*/
/******************************************************************************/

/*
 * PI_Buck_Init()
 * ==============
 * Sets up a fresh PI_Buck_Block_T with safe default values.
 *
 * WHY PROVIDE DEFAULTS?
 * A newly declared PI_Buck_Block_T in C has UNDEFINED (garbage) values.
 * In C there is no constructor like in C++. You must explicitly zero
 * or set every field before using it. PI_Buck_Init() does this for you.
 *
 * In practice, after calling Init you will call SetParams() to overwrite
 * these defaults with your tuned gains. The defaults exist so the block
 * is in a SAFE state even if SetParams is accidentally skipped.
 */
void PI_Buck_Init(PI_Buck_Block_T* pPI)
{
    /*
     * Default tuning values — chosen for a "typical" buck converter:
     *   L = 100 µH, C = 100 µF, Vin = 24V, Vout = 12V, fsw = 100 kHz
     *
     * These are conservative gains (slow but stable). Your application
     * will call PI_Buck_SetParams() to replace them with tuned values.
     */
    pPI->params.Kp        = 0.1f;     /* 0.1 duty cycle per 1V of error      */
    pPI->params.Ki        = 5.0f;     /* integrator winds up at 5 duty/V/s   */
    pPI->params.duty_max  = 0.95f;    /* never exceed 95% duty cycle          */
    pPI->params.duty_min  = 0.05f;    /* never go below 5% duty cycle         */
    pPI->params.Ts        = 0.0001f;  /* 100 µs = 10 kHz control rate         */

    /*
     * Zero all state fields explicitly.
     * "Integrator starts at zero" means: no prior history of errors.
     * This is correct for a cold start — the output voltage is 0V and
     * the PI starts building up duty cycle from scratch.
     */
    pPI->state.integrator  = 0.0f;   /* no accumulated error yet             */
    pPI->state.prev_error  = 0.0f;   /* no previous step error yet           */
    pPI->state.last_output = 0.0f;   /* no duty cycle produced yet           */
}

/*
 * PI_Buck_SetParams()
 * ====================
 * Overrides the default parameters with application-specific tuned values.
 *
 * Call this AFTER PI_Buck_Init() and BEFORE the first PI_Buck_Compute().
 * If you call SetParams mid-run (gain scheduling), the integrator state is
 * preserved — it is NOT reset here. Call PI_Buck_ResetState() first if you
 * want a clean integrator after a parameter change.
 *
 * PARAMETER VALIDATION
 * The function silently clamps/guards invalid inputs rather than returning
 * an error code. This is appropriate for embedded real-time code where
 * error handling must be instantaneous and simple.
 */
void PI_Buck_SetParams(PI_Buck_Block_T* pPI,
                       real32_T         Kp,
                       real32_T         Ki,
                       real32_T         duty_max,
                       real32_T         duty_min,
                       real32_T         Ts)
{
    /* Kp: no validation needed — any real value (even negative) is
     * mathematically valid, though negative Kp would make the loop
     * unstable. We trust the caller here. */
    pPI->params.Kp = Kp;

    /*
     * Ki must be > 0.
     * WHY? The anti-windup formula uses integ_limit = duty_max / Ki.
     * If Ki == 0 we would divide by zero → undefined behaviour / NaN.
     * If Ki < 0 the integrator would run in the wrong direction.
     * Solution: floor Ki at 1e-6 (effectively zero integral action)
     * rather than crashing or producing Inf.
     */
    pPI->params.Ki = (Ki > 0.0f) ? Ki : 1e-6f;  /* guard against div/0 */

    /*
     * duty_max: must be in [0, 1] — a physical constraint of PWM.
     * Values outside this range make no sense for a duty cycle.
     */
    pPI->params.duty_max = pi_clamp(duty_max, 0.0f, 1.0f);

    /*
     * duty_min: must be in [0, duty_max].
     * Note: we clamp against the ALREADY-VALIDATED duty_max, not the
     * raw input. This ensures duty_min ≤ duty_max even if the caller
     * accidentally passes duty_min > duty_max.
     */
    pPI->params.duty_min = pi_clamp(duty_min, 0.0f, pPI->params.duty_max);

    /*
     * Ts: must be > 0. If ≤ 0 the integration step would be zero or
     * negative — meaningless. Floor at 1 µs (1e-6).
     */
    pPI->params.Ts = (Ts > 0.0f) ? Ts : 1e-6f;
}

/*
 * PI_Buck_ResetState()
 * =====================
 * Zeros the integrator and all state memory.
 *
 * WHEN TO CALL THIS
 * -----------------
 * 1. On converter ENABLE after a long disabled period — the integrator
 *    might have drifted; start fresh.
 * 2. On a REFERENCE STEP that is so large that the old integral would
 *    cause an overshoot spike (use with care — may cause a bump).
 * 3. After a FAULT CLEAR — ensure no stale state from the fault period
 *    carries into normal operation.
 * 4. In unit tests — ensures test isolation between test cases.
 *
 * Does NOT touch params — your tuned gains are preserved.
 */
void PI_Buck_ResetState(PI_Buck_Block_T* pPI)
{
    pPI->state.integrator  = 0.0f;   /* discard all accumulated error        */
    pPI->state.prev_error  = 0.0f;   /* clear debug field                    */
    pPI->state.last_output = 0.0f;   /* clear last output field              */
}

/*
 * PI_Buck_Compute()
 * ==================
 * THE CORE FUNCTION — called every control period.
 *
 * This function implements one discrete-time step of the PI algorithm.
 * On Aurix TriCore this would be called from a GTM (Generic Timer Module)
 * interrupt every 100 µs. In EmbedSim it is called by the simulation engine
 * at each dt step.
 *
 * The function reads from pPI->params and pPI->state (both read-only within
 * the algorithm), then UPDATES pPI->state at the end with the new integrator
 * value, previous error, and last output.
 *
 * IMPORTANT: All local variables are declared at the top of the function
 * (C89 style). This is required by MISRA C:2012 Rule 8.1 and by the
 * TASKING compiler's default settings for safety-relevant code.
 */
void PI_Buck_Compute(PI_Buck_Block_T*       pPI,
                     const PI_Buck_Input_T* pIn,
                     real32_T               dt,
                     PI_Buck_Output_T*      pOut)
{
    /*
     * Local variable declarations — all at the top (MISRA C:2012 Rule 8.1).
     * These are temporaries that exist only for the duration of this function
     * call. They live on the CPU stack and cost no heap or global memory.
     */
    real32_T error;        /* voltage error this step  [V]                   */
    real32_T integ_limit;  /* anti-windup ceiling for integrator             */
    real32_T raw_duty;     /* PI output before output clamping               */
    real32_T sample_time;  /* effective timestep used for integration  [s]   */

    /* ──────────────────────────────────────────────────────────────────────
     * STEP 0: Determine sample time
     * ──────────────────────────────────────────────────────────────────────
     * EmbedSim calls this function with the actual simulation dt, which may
     * differ from the configured Ts if the solver uses adaptive stepping or
     * if the user changed dt at the command line.
     *
     * On real hardware you would always pass the fixed ISR period (e.g.,
     * 0.0001f), and this line would just pick it up from params.Ts.
     *
     * dt > 0  → use the provided dt  (runtime override)
     * dt ≤ 0  → fall back to the stored params.Ts
     */
    sample_time = (dt > 0.0f) ? dt : pPI->params.Ts;

    /* ──────────────────────────────────────────────────────────────────────
     * STEP 1: Compute the voltage error
     * ──────────────────────────────────────────────────────────────────────
     * error = V_ref - V_meas
     *
     * Sign convention:
     *   error > 0  → output is BELOW reference → increase duty (push up)
     *   error < 0  → output is ABOVE reference → decrease duty (push down)
     *   error = 0  → output is exactly at reference (goal achieved)
     *
     * Example: V_ref=12V, V_meas=10V → error = +2V → controller increases duty.
     */
    error = pIn->V_ref - pIn->V_meas;

    /* ──────────────────────────────────────────────────────────────────────
     * STEP 2: Update the integrator with anti-windup clamping
     * ──────────────────────────────────────────────────────────────────────
     * The integrator accumulates error over time using forward Euler:
     *
     *   integ(k) = integ(k-1) + error * sample_time
     *
     * Without anti-windup, if the error is positive for a long time (e.g.,
     * converter is starting up), the integrator grows unboundedly. When the
     * error eventually becomes negative (overshoot), the integrator is so
     * large that the controller stays at maximum duty for a long time before
     * "unwinding" — this causes a slow, oscillatory transient.
     *
     * ANTI-WINDUP STRATEGY: Clamp the integrator so that:
     *   Ki * integ  ≤  duty_max
     *   Ki * integ  ≥ -duty_max
     *
     * This means:
     *   integ  ≤  duty_max / Ki    →  integ_limit = duty_max / Ki
     *   integ  ≥ -duty_max / Ki
     *
     * With Ki=8.0, duty_max=0.9:
     *   integ_limit = 0.9 / 8.0 = 0.1125
     *
     * So the integral contribution Ki*integ can contribute at most ±0.9
     * to the duty cycle — which is the full saturation range.
     * The controller can still reach duty_max, but only once AND it unwinds
     * immediately when error reverses.
     */
    integ_limit = pPI->params.duty_max / pPI->params.Ki;

    pPI->state.integrator = pi_clamp(
        pPI->state.integrator + error * sample_time,  /* new candidate value */
        -integ_limit,                                  /* lower bound         */
         integ_limit                                   /* upper bound         */
    );

    /* ──────────────────────────────────────────────────────────────────────
     * STEP 3: Compute the raw PI output
     * ──────────────────────────────────────────────────────────────────────
     * duty = Kp * error  +  Ki * integrator
     *         ↑                ↑
     *    proportional      integral term
     *    term (reacts       (eliminates
     *    immediately)       steady-state
     *                       error)
     *
     * Example with Kp=0.15, Ki=8.0, error=2V, integ=0.05:
     *   raw_duty = 0.15*2.0 + 8.0*0.05 = 0.30 + 0.40 = 0.70
     *   → 70% duty cycle
     */
    raw_duty = pPI->params.Kp * error
             + pPI->params.Ki * pPI->state.integrator;

    /* ──────────────────────────────────────────────────────────────────────
     * STEP 4: Clamp the output to [duty_min, duty_max]
     * ──────────────────────────────────────────────────────────────────────
     * Even with the anti-windup integrator clamp, the proportional term
     * Kp * error can still push raw_duty outside the physical limits.
     * This final output clamp guarantees the PWM hardware never receives
     * an impossible duty cycle command.
     *
     * Example: large error spike → raw_duty = 1.3 → clamped to 0.90
     */
    pOut->duty = pi_clamp(raw_duty,
                          pPI->params.duty_min,
                          pPI->params.duty_max);

    /* ──────────────────────────────────────────────────────────────────────
     * STEP 5: Update state memory for the next step
     * ──────────────────────────────────────────────────────────────────────
     * Save the output and current error for debugging / oscilloscope
     * inspection via the Cython .last_output and scope.
     * These writes do NOT affect the PI algorithm for the CURRENT step.
     */
    pPI->state.last_output = pOut->duty;  /* store what we commanded          */
    pPI->state.prev_error  = error;       /* store error for next-step debug  */
}

/* ── End of pi_buck_controller.c ─────────────────────────────────────────── */

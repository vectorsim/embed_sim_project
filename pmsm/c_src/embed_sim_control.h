/**********************************************************************************************************************
 * \file      embed_sim_control.h
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
 *              - Output pointers: PascalCase_P
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

#ifndef EMBEDSIM_EMBED_SIM_CONTROL_H_
#define EMBEDSIM_EMBED_SIM_CONTROL_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"
#include "embed_sim_matrix.h"
#include "embed_sim_motor_parameter.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Conversion: RPM to rad/s
 *
 * \param[in] RPM  Speed in revolutions per minute.
 *
 * \return  Speed in radians per second.
 */
#define CON_RPM_TO_RAD(RPM)             ((RPM * ES_MATH_2PI_F) / 60.0F)

/**
 * \brief  Conversion: rad/s to RPM
 *
 * \param[in] RAD  Speed in radians per second.
 *
 * \return  Speed in revolutions per minute.
 */
#define CON_RAD_TO_RPM(RAD)             ((RAD * 60.0F) / ES_MATH_2PI_F)


/**
 * @def ES_SVM_START_MOD_FUNC(TAU)
 * @brief Calculates a smooth start modulation factor.
 *
 * Uses a cubic smoothstep function:
 * @code
 * f(TAU) = 3 * TAU^2 - 2 * TAU^3
 * @endcode
 *
 * When @p TAU is in the range [0, 1], the function smoothly transitions
 * from 0 to 1 with zero slope at both endpoints.
 *
 * @param[in] TAU Normalized modulation time, typically in the range [0, 1].
 * @return Smooth modulation factor in the range [0, 1] for TAU in [0, 1].
 */

#define ES_SVM_START_MOD_FUNC(TAU)   ((3.0 * (TAU) * (TAU)) - (2.0 * (TAU) * (TAU) * (TAU)))



#include <math.h>


/**
 * \brief  Maximum speed in RPM
 */
#define MAX_SPEED_RPM                   (3000.0F)

/**
 * \brief  Speed settle tolerance for trajectory generation (RPM)
 */
#define SPEED_SETTLE_TOL                (0.1F)

/**
 * \brief  Maximum jerk limit (RPM/s^3)
 */
#define MAX_JERK_RPM                    (3500.0F)

/**
 * \brief  Maximum jerk limit (RPM/s^2)
 */
#define MAX_ACCEL_RPM                   (800.0F) // 500

/**
 * \brief  Jerk smoothing factor (0.0 to 1.0)
 */
#define JERK_SMOOTHING_FACTOR           (0.2F)

/**
 * \brief   Current PI controller gains (correction on top of flatness)
 *
 * \details These gains provide proportional correction to the flatness feedforward
 *          voltages based on current errors. Higher gains give faster response
 *          but may cause instability.
 *
 * \note    Sensor noise mitigation strategy:
 *          - Very low proportional gains (especially on d-axis)
 *          - Low integral gains to prevent noise amplification
 *          - The feedforward handles most of the control effort (80-90%)
 *          - PI only corrects for model errors and low-frequency disturbances
 *
 * \warning Increasing gains above these values will amplify sensor noise
 *          and may cause audible noise or instability.
 */
#define DFC_CURRENT_KP_D_F              (0.00999F)    /**< d-axis proportional gain */
#define DFC_CURRENT_KI_D_F              (0.00000025F)    /**< d-axis integral gain     */
#define DFC_CURRENT_KP_Q_F              (0.019995F)    /**< q-axis proportional gain */
#define DFC_CURRENT_KI_Q_F              (0.00000025F)    /**< q-axis integral gain     */


/**
 * \brief  Spinning detection parameters
 */
#define DFC_SPINNING_PAST_INDEX  (8950U)     /**< 0.45s debounce time */
#define DFC_STOPPED_PAST_INDEX    (200U)      /**< 0.01s debounce time */

/**
 * \brief   Speed PI controller gains (outer loop)
 *
 * \details These gains correct speed errors caused by load disturbances.
 *          The speed loop generates a torque correction that is added to the
 *          flatness feedforward torque.
 *
 * \note    The speed loop gains are also kept low to avoid noise amplification.
 *          The integral term eliminates steady-state speed error.
 */
#define DFC_SPEED_KP_Q_F                (0.00092F)       /**< Speed proportional gain (for torque correction) */
#define DFC_SPEED_KI_Q_F                (0.00091F)    /**< Speed integral gain */

/**
 * \brief   Maximum integrator anti-windup limit (common for speed and current)
 *
 * \details This limit prevents integral windup in both the speed and current
 *          PI controllers. It is applied to the accumulated error terms.
 *          The value is chosen to allow enough correction without causing
 *          excessive voltage commands.
 */
#define DFC_INTEGRAL_LIMIT_F            (25.0F)


#define DFC_MIN_VELOCITY               (500.0F)

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Control algorithm selection
 */
typedef enum
{
    SIM_CTRL_OPEN_LOOP  = 0x0U,    /**< Open-loop control (no feedback) */
    SIM_CTRL_DFC        = 0x1U     /**< Differential Flatness Control */
} EmbedSimCtrl_T;

/**
 * \brief  Input Structure for Control Loop
 *
 * \details Contains all reference signals, feedback measurements, and
 *          configuration parameters for the motor control system.
 */
typedef struct
{
    real32_T    AngularVelocityRefRpmM;       /**< Mechanical speed reference [RPM], Mechanical, (Application Input)           */
    real32_T    Iu;                           /**< Current Phase U [A], (Application Input)                                    */
    real32_T    Iv;                           /**< Current Phase V [A], (Application Input)                                    */
    real32_T    Iw;                           /**< Current Phase W [A], (Application Input)                                    */
    real32_T    RotorSpeedSensorM;            /**< Rotor Velocity in RPM [RPM] Mechanical, (Sensor), (Application Input)       */
    real32_T    RotorPositionSensorM;         /**< Rotor Position [RAD] (Sensor), Mechanical, (Sensor), (Application Input)    */
    real32_T    SampleTime;                   /**< Control loop sample time [s], (Application Input)                           */
    real32_T    Vdc;                          /**< DC bus voltage [V],  (Application Input)                                    */
    uint32_T    CtrlAlg;                      /**< Control algorithm selection, (Application Input)                            */
    uint32_T    Valid;                        /**< Data validity flag (0x1 = valid),  (Application Input)                      */
    uint32_T    SwitchToClosedLoop;           /**< Flag to indicate to closed-loop, (Intern)                                   */
    uint32_T    ControlReInit;                /**< Flag to clear the Control State, (Intern)                                   */
    real32_T    RotorPositionRefM;            /**< Rotor position reference [rad], Mechanic (Intern,Path Calculation)          */
    real32_T    RotorVelocityRefM;            /**< Angular velocity [rad/s], Mechanical (Intern,Path Calculation)              */
    real32_T    RotorAccelerationRefM;        /**< Angular acceleration [rad/s²], Mechanical  (Intern,Path Calculation)        */
    real32_T    RotorJerkRefM;                /**< Angular jerk [rad/s³], Mechanical    (Intern,Path Calculation)              */
    real32_T    RotorPositionObsEstM;         /**< Rotor Position [RAD], Mechanical (Observer Estimated)                       */
    real32_T    RotorSpeedObsEstM;            /**< Rotor Velocity in RPM [RPM], Mechanical (Observer Estimated) Intern         */
    uint64_T    LoopCounter;                  /**< Loop Counter                                                                */
} EmbedSimCtrlInput_T;

/**
 * \brief  Output Structure for Control Loop
 *
 * \details Contains PWM duty cycles and status information for the inverter.
 */
typedef struct
{
    real32_T    DutyU;                      /**< Phase U PWM duty cycle [0.0 .. 1.0] */
    real32_T    DutyV;                      /**< Phase V PWM duty cycle [0.0 .. 1.0] */
    real32_T    DutyW;                      /**< Phase W PWM duty cycle [0.0 .. 1.0] */
    uint32_T    SvmSector;                  /**< SVM Sector (0-6)                    */
    real32_T    RotorSpeedObsEstM;          /**< Estimated rotor speed [RPM]         */
    real32_T    RotorPositionObsEstM;       /**< Estimated rotor position [rad]      */
    uint32_T    Valid;                      /**< Output validity flag (0x1 = valid)  */
} EmbedSimCtrlOutput_T;

/**
 * \brief   Permanent Magnet Synchronous Motor (PMSM) parameters
 *
 * \details Contains all motor parameters used by the control algorithms.
 */
typedef struct
{
    real32_T    PolePairs;                 /**< Number of pole pairs                         [-]           */
    real32_T    Rs;                        /**< Stator resistance                            [Ohm]         */
    real32_T    Ld;                        /**< Direct-axis inductance                       [H]           */
    real32_T    Lq;                        /**< Quadrature-axis inductance                   [H]           */
    real32_T    FluxPm;                    /**< Permanent magnet flux linkage                [Wb]          */
    real32_T    J;                         /**< Rotor inertia                                [kg·m²]       */
    real32_T    B;                         /**< Viscous damping coefficient                  [N·m·s]       */
    real32_T    Vdc;                       /**< DC bus voltage                               [V]           */
    real32_T    TorqueLoad;                /**< Load torque (external)                       [N·m]         */

    /* Current PI gains */
    real32_T    ParamPidCurrentQProp;      /**< PID Proportional Parameter for Q Current                    */
    real32_T    ParamPidCurrentQInteg;     /**< PID Integral Parameter for Q Current                        */
    real32_T    ParamPidCurrentDProp;      /**< PID Proportional Parameter for D Current                    */
    real32_T    ParamPidCurrentDInteg;     /**< PID Integral Parameter for D Current                        */

    /* Speed PI gains */
    real32_T    ParamPidSpeedQProp;        /**< PID Proportional Parameter for Speed (torque correction)    */
    real32_T    ParamPidSpeedQInteg;       /**< PID Integral Parameter for Speed                            */

    /* Integral Error */
    real32_T    SpeedIntegralError;        /**< Accumulated speed error for the outer speed PI controller   */
    real32_T    IdIntegralError;           /**< Accumulated d-axis current error for the inner current PI   */
    real32_T    IqIntegralError;           /**< Accumulated q-axis current error for the inner current PI   */

    /**
     * \brief  Integral anti‑windup limit (common for speed and current integrators)
     *
     * \details Clamps the accumulated error in each PI controller to prevent
     *          integral windup during saturation or large disturbances.
     *          The same limit is applied to speed, Id, and Iq integrators.
     */
    real32_T    ParamPidIntegralLimit;

    real32_T    SvmRotorThetaE;               /**< Electrical Angle of Rotor (Model Representation)          */

    /* Startup modulation (added for state reporting) */
    real32_T    SvmModulationIndex;         /**< SVM ModulationIndex                                          */
    real32_T    SvmStartUpTimer;            /**< Start  Up Timer for Modulation                               */

} EmbedSimMachineParam_T;

/**
 * \brief  Motor structure grouping all control data
 *
 * \details Contains pointers to input, output, and parameter structures
 *          for easy passing to control functions.
 */
typedef struct
{
    EmbedSimCtrlInput_T*     InputPtr;    /**< Pointer to input structure */
    EmbedSimCtrlOutput_T*    OutputPtr;   /**< Pointer to output structure */
    EmbedSimMachineParam_T*  MachinePtr;  /**< Pointer to motor parameters */
} EmbedSimMachine_T;

/**
 * \brief  Motor state structure for unified reporting
 *
 * \details Contains all motor state information for display and logging.
 *          This structure is used to provide a unified view of motor
 *          operation across C and Python implementations.
 */
typedef struct
{
    /* Mechanical */
    real32_T    SpeedRpm;                    /**< Measured speed [RPM] */
    real32_T    PositionRad;                 /**< Rotor position mechanica [rad] */

    /* PWM */
    real32_T    DutyU;                       /**< Phase U duty [0-1]  */
    real32_T    DutyV;                       /**< Phase V duty [0-1]  */
    real32_T    DutyW;                       /**< Phase W duty [0-1]  */
    uint32_T    SvmSector;                   /**< SVM sector [0-6]    */

    /* Status */
    uint32_T    Valid;                       /**< 1=valid, 0=invalid */
    uint64_T    LoopCounter;                 /**< Loop counter */
    uint32_T    SwitchToClosedLoop;

} EmbedSimMotorState_T;

/**
 * \brief  Global motor instance
 *
 * \details Contains the traction motor control data and parameters.
 */
extern EmbedSimMachine_T TractionMotor_G;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Top-level PMSM application initialisation
 *
 * \details Initializes the SVM and DFC controller modules, and sets up
 *          the global motor structure with default parameters.
 *          Must be called once before any control step.
 *
 * \return  void
 */
extern void EmbedSim_ControlInit(void);

/**
 * \brief   Top-level PMSM control step
 *
 * \details Executes one control cycle. Based on the input validity and
 *          control mode, either open-loop or DFC control is performed.
 *          For DFC mode, estimates rotor position and speed from sensors.
 *
 * \param[in]  motorPtr  Pointer to motor structure containing:
 *                       - InputPtr   : Control input (references & feedback)
 *                       - OutputPtr  : Control output (voltage commands)
 *                       - MachinePtr : Motor parameters
 *
 * \return  void
 */
extern void EmbedSim_ControlStep(EmbedSimMachine_T* const MotorPtr);

/**
 * \brief   Check if the motor is spinning
 *
 * \param[in] InputPtr    Pointer to control input structure.
 * \param[in] Duration    Time in S (Motor is spining since then.
 *
 * \return  1 if motor is spinning fast enough, 0 otherwise.
 */
extern uint32_T EmbedSim_IsMotorSpinning(const EmbedSimCtrlInput_T* const InputPtr, real32_T SpeedRefRPM, real32_T  Duration);


/**
 * \brief   Check if the motor has stopped
 *
 * \param[in] InputPtr    Pointer to control input structure.
 * \param[in] PastIndex   Number of consecutive valid samples required.
 *
 * \return  1 if motor is stopped, 0 otherwise.
 */
extern uint32_T EmbedSim_IsNotSpinning(const EmbedSimCtrlInput_T* const InputPtr, uint32_T PastIndex);

/**
 * \brief   Get current motor state for unified reporting
 *
 * \details Fills the motor state structure with current values from
 *          the control system. This provides a unified view of motor
 *          operation for display and logging.
 *
 * \param[in]  motorPtr   Pointer to motor structure
 * \param[out] statePtr   Pointer to state structure to fill
 *
 * \return  void
 */
extern void EmbedSim_GetMotorState(EmbedSimMachine_T* const motorPtr,  EmbedSimMotorState_T* const statePtr);

/**
 * \brief   Wrap angle to [0, 2pi)
 *
 * \details Normalizes an angle to the range [0, 2π) using fmodf.
 *          Useful for rotor angle and Park transform calculations.
 *
 * \param[in,out] anglePtr  Pointer to angle value to be wrapped (in radians).
 */
 extern void EmbedSim_WrapAngleTwoPi(real32_T* AnglePtr);


 /**
  * @brief Calculates the shortest signed angular distance between two angles.
  *
  * Both input angles are expected to be in the range [0, 2*PI).
  * The returned angular distance is in the range [-PI, PI).
  *
  * A positive result means Angle1 is ahead of Angle2.
  * A negative result means Angle1 is behind Angle2.
  *
  * @param[in] Angle1 First angle in radians.
  * @param[in] Angle2 Second angle in radians.
  *
  * @return Shortest signed angular distance Angle1 - Angle2 in radians.
  */
 extern real32_T EmbedSim_AngleDistance(real32_T Angle1, real32_T Angle2);


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
extern  real32_T EmbedSim_ClampValue(real32_T Val, real32_T MinVal, real32_T MaxVal);


 /**
  * @brief Calculates the online jerk-limited velocity trajectory.
  *
  * @details
  * Generates the reference angular velocity, acceleration, jerk, and position
  * online while moving the velocity reference toward the requested target.
  *
  * The trajectory generator uses the velocity error to determine the required
  * direction of motion. A stopping acceleration is calculated from the remaining
  * velocity error and the configured maximum jerk:
  *
  * @code
  * a_stop = sqrt(2 * Jmax * |velocity_error|)
  * @endcode
  *
  * The calculated stopping acceleration represents the acceleration required
  * to remove the remaining velocity error using the available maximum jerk.
  * The requested acceleration is limited to the configured maximum acceleration.
  *
  * The jerk required to move the current acceleration toward the desired
  * acceleration within one sample period is then calculated and limited to
  * +/-Jmax.
  *
  * Assuming constant jerk during one sample period, the trajectory states are
  * integrated using:
  *
  * @code
  * a(k+1) = a(k) + j(k) * Ts
  *
  * w(k+1) = w(k) + a(k) * Ts + 0.5 * j(k) * Ts^2
  *
  * theta(k+1) = theta(k)
  *            + w(k) * Ts
  *            + 0.5 * a(k) * Ts^2
  *            + (1/6) * j(k) * Ts^3
  * @endcode
  *
  * The previous velocity and acceleration are stored before updating the
  * trajectory states. These previous values are used for consistent numerical
  * integration during the current sample.
  *
  * The generated velocity and acceleration references are constrained by
  * their configured limits. If the calculated velocity crosses the target
  * velocity, the trajectory is clamped to the target and the acceleration
  * and jerk are reset to zero.
  *
  * When the velocity error is within the configured settling tolerance and
  * the acceleration is sufficiently small, the trajectory is considered
  * settled. The velocity is then set directly to the target and the dynamic
  * states are reset.
  *
  * If control re-initialization is requested, the acceleration and jerk states
  * are reset before continuing trajectory generation. The current velocity
  * is retained as the starting point of the new trajectory.
  *
  * @note The target velocity is limited to +/-Vmax.
  * @note The acceleration is limited to +/-Amax.
  * @note The jerk is limited to +/-Jmax.
  * @note The trajectory states use SI units:
  *       velocity in rad/s, acceleration in rad/s^2,
  *       jerk in rad/s^3, and position in rad.
  * @note InputPtr->SampleTime must be greater than zero.
  *
  * @param[in,out] InputPtr
  *        Pointer to the control input and trajectory state structure.
  *
  * @param[in] ParaPtr
  *        Pointer to the motor parameter structure.
  *        Currently unused by this function.
  */
 extern void EmbedSim_CalculateJerkLimitedTrajectory(EmbedSimCtrlInput_T* const InputPtr, const EmbedSimMachineParam_T* const ParaPtr);

#endif /* EMBEDSIM_EMBED_SIM_CONTROL_H_ */

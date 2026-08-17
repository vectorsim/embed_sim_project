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
#define MAX_JERK_RPM                    (3000.0F)

/**
 * \brief  Maximum jerk limit (RPM/s^2)
 */
#define MAX_ACCEL_RPM                   (500.0F)

/**
 * \brief  Jerk smoothing factor (0.0 to 1.0)
 */
#define JERK_SMOOTHING_FACTOR           (0.2F)

/**
 * \brief  Minimum speed to switch to closed-loop control (RPM)
 */
#define CLOSED_LOOP_MIN_SPEED           (60.0F)

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
#define DFC_CURRENT_KP_D_F              (0.0001F)    /**< d-axis proportional gain */
#define DFC_CURRENT_KP_Q_F              (0.0195F)    /**< q-axis proportional gain */
#define DFC_CURRENT_KI_D_F              (0.0005F)    /**< d-axis integral gain */
#define DFC_CURRENT_KI_Q_F              (0.0002F)    /**< q-axis integral gain  */

/**
 * \brief  DFC startup parameters (match Python's 0.3s ramp)
 */
#define DFC_STARTUP_TIME_S       (0.8F)      /**< Startup duration [s]        */
#define DFC_STARTUP_SPEED_RPM    (300.0F)    /**< Fixed speed during startup [RPM] */
#define DFC_STARTUP_MOD_MIN      (0.001F)     /**< Initial modulation index     */
#define DFC_STARTUP_MOD_MAX      (0.25F)     /**< Final modulation index       */


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
#define DFC_SPEED_KP_Q_F                (0.0021F)    /**< Speed proportional gain (for torque correction) */
#define DFC_SPEED_KI_Q_F                (0.0001F)    /**< Speed integral gain */

/**
 * \brief   Maximum integrator anti-windup limit (common for speed and current)
 *
 * \details This limit prevents integral windup in both the speed and current
 *          PI controllers. It is applied to the accumulated error terms.
 *          The value is chosen to allow enough correction without causing
 *          excessive voltage commands.
 */
#define DFC_INTEGRAL_LIMIT_F            (5.0F)

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
    real32_T    RotorAccerlerationRefM;       /**< Angular acceleration [rad/s²], Mechanical  (Intern,Path Calculation)        */
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
    real32_T    PolePairs;                 /**< Number of pole pairs                         [-]      */
    real32_T    Rs;                        /**< Stator resistance                            [Ohm]    */
    real32_T    Ld;                        /**< Direct-axis inductance                       [H]      */
    real32_T    Lq;                        /**< Quadrature-axis inductance                   [H]      */
    real32_T    FluxPm;                    /**< Permanent magnet flux linkage                [Wb]     */
    real32_T    J;                         /**< Rotor inertia                                [kg·m²]  */
    real32_T    B;                         /**< Viscous damping coefficient                  [N·m·s]  */
    real32_T    Vdc;                       /**< DC bus voltage                               [V]      */
    real32_T    TorqueLoad;                /**< Load torque (external)                       [N·m]    */

    /* Current PI gains */
    real32_T    ParamPidCurrentQProp;      /**< PID Proportional Parameter for Q Current              */
    real32_T    ParamPidCurrentQInteg;     /**< PID Integral Parameter for Q Current                  */
    real32_T    ParamPidCurrentDProp;      /**< PID Proportional Parameter for D Current              */
    real32_T    ParamPidCurrentDInteg;     /**< PID Integral Parameter for D Current                  */

    /* Speed PI gains */
    real32_T    ParamPidSpeedQProp;        /**< PID Proportional Parameter for Speed (torque correction) */
    real32_T    ParamPidSpeedQInteg;       /**< PID Integral Parameter for Speed                         */

    /**
     * \brief  Integral anti‑windup limit (common for speed and current integrators)
     *
     * \details Clamps the accumulated error in each PI controller to prevent
     *          integral windup during saturation or large disturbances.
     *          The same limit is applied to speed, Id, and Iq integrators.
     */
    real32_T    ParamPidIntegralLimit;

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
extern void EmbedSim_ControlStep(EmbedSimMachine_T* const motorPtr);



/**
 * \brief Calculates an online time-optimal jerk-limited speed trajectory.
 *
 * \details
 * Generates the reference motor speed and its derivatives online using
 * a jerk-limited S-curve trajectory. The trajectory adapts at every
 * control sample according to the instantaneous speed error and the
 * remaining distance to the target speed.
 *
 * The trajectory is generated using the following states and control:
 *
 *   - State:
 *       omega_ref_dot = acceleration
 *   - Control:
 *       jerk
 *
 * At each control sample, the algorithm:
 *
 *   1. Calculates the speed error.
 *   2. Determines the direction toward the target speed.
 *   3. Calculates the required braking distance.
 *   4. Selects the appropriate jerk:
 *        +Jmax : increase acceleration toward the target.
 *         0    : maintain the current acceleration.
 *        -Jmax : reduce acceleration to prepare for the target.
 *   5. Integrates jerk to obtain acceleration.
 *   6. Integrates acceleration to obtain the reference speed.
 *
 * The resulting trajectory consists of jerk-limited acceleration and
 * deceleration phases followed by a constant-speed phase when the
 * target speed is reached.
 *
 * The algorithm is time-optimal subject to the specified jerk and
 * acceleration constraints. No predefined T1 or total trajectory time
 * is required; the trajectory timing is determined online.
 *
 * \param[in,out] InputPtr Pointer to the control input structure
 *                         containing speed references and feedback signals.
 * \param[in]     ParaPtr  Pointer to the motor parameter structure
 *                         containing trajectory constraints such as
 *                         maximum jerk and acceleration.
 *
 * \note The trajectory is intended to be executed at the controller's
 *       fixed sampling frequency.
 */
extern uint32_T EmbedSim_IsMotorSpinning(const EmbedSimCtrlInput_T* const  InputPtr, uint32_T PastIndex);



extern uint32_T EmbedSim_IsNotSpinning(const EmbedSimCtrlInput_T* const InputPtr,  uint32_T PastIndex);

#endif /* EMBEDSIM_EMBED_SIM_CONTROL_H_ */

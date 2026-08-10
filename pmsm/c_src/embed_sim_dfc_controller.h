/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.h
 * \brief     DFC (Direct Field Control) controller for embedded motor control applications.
 *
 * \details   Implements PI-based speed and current control loops for permanent magnet
 *            synchronous motors (PMSM). Includes anti-windup and output limiting.
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
 *              - Local variables: Lower pascalCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   1.0.0
 * \date      2026-08-09
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBED_SIM_DFC_CONTROLLER_H_
#define EMBED_SIM_DFC_CONTROLLER_H_

#include "embed_sim_sys_types.h"
#include "embed_sim_control.h"

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/



/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize DFC controller
 *
 * \details Resets all PI controllers to zero and sets Initialized flag to 1.
 *          Must be called before DFC_Step().
 */
void DFC_Init(void);

/**
 * \brief   Execute DFC control step
 *
 * \details Computes speed and current control outputs based on input references
 *          and feedback measurements. Implements cascade control structure:
 *          Speed PI → Iq reference → Iq PI → Vq output.
 *
 * \param[in]  InputPtr   Pointer to control input structure (references & feedback).
 * \param[in]  MPtr       Pointer to machine parameters (resistance, inductance, flux).
 * \param[out] OutputPtr  Pointer to control output structure (voltage commands).
 */
void DFC_Step(EmbedSimCtrlInput_T* const InputPtr,
              const EmbedSimMachineParam_T* const MPtr,
              EmbedSimCtrlOutput_T* const OutputPtr);

/**
 * \brief   Configure PI gains
 *
 * \details Sets proportional (Kp) and integral (Ki) gains for all three
 *          PI controllers in the cascade structure.
 *
 * \param[in] KpSpeed  Speed controller proportional gain.
 * \param[in] KiSpeed  Speed controller integral gain.
 * \param[in] KpIq     Iq current controller proportional gain.
 * \param[in] KiIq     Iq current controller integral gain.
 * \param[in] KpId     Id current controller proportional gain.
 * \param[in] KiId     Id current controller integral gain.
 */
void DFC_ConfigurePI(real32_T KpSpeed, real32_T KiSpeed,
                     real32_T KpIq, real32_T KiIq,
                     real32_T KpId, real32_T KiId);

#endif /* EMBED_SIM_DFC_CONTROLLER_H_ */

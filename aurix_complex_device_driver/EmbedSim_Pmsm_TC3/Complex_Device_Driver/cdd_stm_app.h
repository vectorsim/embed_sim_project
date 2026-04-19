/**********************************************************************************************************************
 * \file        cdd_stm_app.h
 * \brief       Public interface for the System Timer Module (STM0) driver.
 *
 * \details     Configures STM0 Compare-0 to generate a periodic 1 ms system
 *              tick interrupt on CPU0.  The ISR calls System_Tick_Handler()
 *              which is owned by cdd_task_handler_app.c.
 *
 *              Hardware capture note (TC3xx Reference Manual ds2 P.60):
 *              Reading STM0_TIM0 automatically latches the upper 32 bits into
 *              STM0_CAP.  Get_Lower_System_Time() reads TIM0 only;
 *              Get_System_Time() reads TIM0 then CAP to reconstruct the full
 *              64-bit counter atomically under interrupt lock.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function, matching .c definition
 *              - Rule  8.6 : Definitions reside in cdd_stm_app.c
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_STM_APP_H_
#define CDD_STM_APP_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialises STM0 and arms the Compare-0 interrupt for a 1 ms period.
 *
 * \details Computes the full time-constant table from the live fSTM frequency,
 *          configures CMCON / ICR / SRC registers, then loads the first compare
 *          value.  Must be called once during system init before global interrupt
 *          enable.
 *
 * \return  None
 */
extern void Initialize_STM_Module(void);

/**
 * \brief   Returns the lower 32 bits of the STM0 free-running counter.
 *
 * \details Reads STM0_TIM0.U directly.  As a side-effect the hardware latches
 *          the upper bits into STM0_CAP — required before any Get_System_Time()
 *          call.  Safe from both task and ISR context.
 *
 * \return  Current lower 32-bit STM tick count  [STM ticks]
 */
extern uint32_T Get_Lower_System_Time(void);

/**
 * \brief   Returns the full 64-bit STM0 free-running counter value.
 *
 * \details Disables interrupts briefly, reads TIM0 (triggers CAP latch), reads
 *          CAP, reconstructs the 64-bit value, then restores interrupt state.
 *
 * \return  Current 64-bit STM tick count  [STM ticks]
 */
extern uint64_T Get_System_Time(void);

#endif /* CDD_STM_APP_H_ */

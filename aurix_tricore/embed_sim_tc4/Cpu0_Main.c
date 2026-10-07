/**********************************************************************************************************************
 * \file Cpu0_Main.c
 *
 * \copyright Copyright (c) 2026 Infineon Technologies AG. All rights reserved.
 *
 *
 *
 *                                 IMPORTANT NOTICE
 *
 * Infineon Technologies AG (Infineon) licenses this file to you under the
 * Infineon Automotive SW Lab License v2025-01 (IFASLL). You may not use
 * this file except in compliance with IFASLL.
 *
 * The full license text is contained in IFASLL202501.pdf delivered with this SW.
 * Unless required by applicable law or agreed to in writing, software distributed
 * under this license is distributed "AS IS" without any warranty or liability of any
 * kind and Infineon hereby expressly disclaims any warranties or representations,
 * whether express, implied, statutory or otherwise, including but not limited to
 * warranties of workmanship, merchantability, fitness for a particular purpose,
 * defects in the licensed items, or non-infringement of third parties'
 * intellectual property rights. See the full license text for the specific
 * language governing permissions and limitations under IFASLL.
 *********************************************************************************************************************/
#include "Ifx_Types.h"
#include "Ifx_Cfg.h"
#include "IfxCpu.h"
#include "IfxWtu.h"
#include "AllowAccess.h"
#include "PpuInterface.h"
#include "IfxCpu_Trap.h"
#include "cdd_app.h"

#include "DAVE.h"

volatile uint32 counterCpu0= 0;

void core0_main(void)
{
    IfxCpu_enableInterrupts();
    /* !!WATCHDOG0 AND SAFETY WATCHDOG ARE DISABLED HERE!!
     * Enable the watchdogs and service them periodically if it is required
     */
    IfxWtu_disableCpuWatchdog(IfxWtu_getCpuWatchdogPassword());
    IfxWtu_disableSystemWatchdog(IfxWtu_getSystemWatchdogPassword());

    allowAccess();

    /* Start up the PPU */
    PpuInterface_startPpu();

    Cdd_Init();

    /* Initialization of DAVE APPs  */
	DAVE_STATUS_t status = DAVE_Init();

	if (status != DAVE_STATUS_SUCCESS) {
		/* Placeholder for error handler code. The while loop below can be replaced with a user error handler. */
		while (1U) {

		}
	}

	Cdd_Start();

    while (1)
    {
        counterCpu0++;
    }
}

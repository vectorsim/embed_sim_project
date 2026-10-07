/**
 * \file PpuInterface.c
 * \brief This file contains the APIs for Ppu Interfacing from TriCore.
 *
 * \copyright Copyright (C) Infineon Technologies AG 2026
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


/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Ifx_Cfg.h"
#include "PpuInterface.h"
#include "SharedData.h"
#include "IfxPpucCore.h"
#include "IfxSrc.h"


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
extern unsigned __PPU_CSM_START[];
extern unsigned __PPU_CSM_END[];
extern unsigned __PPU_VCCM_START[];
extern unsigned __PPU_VCCM_END[];
/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

void PpuInterface_startPpu(void)
{
    /* Drive the Initial Program counter value on PPU's PC interface and
     * release it from the state of reset. The PPU starts execution upon
     * release of reset.
     */

    IfxPpucCore_enableModule(&MODULE_PPU);

    /* Wait untill the module is enabled */
    while(IfxPpucCore_isModuleEnabled(&MODULE_PPU) != TRUE)
    {	}

	/* Clear and ECC-Initialize the CSM */
	//TODO:  This is a workaround because of VMT driver not supporting 3rd Party IPs
	volatile uint64*addressCSM = (volatile uint64*)__PPU_CSM_START;
	while (addressCSM < (volatile uint64*)__PPU_CSM_END) *addressCSM++= 0x0ULL;

	//TODO:  This is a workaround because of VMT driver not supporting 3rd Party IPs
	volatile uint64*addressVCCM = (volatile uint64*)__PPU_VCCM_START;
	while (addressVCCM < (volatile uint64*)__PPU_VCCM_END) *addressVCCM++= 0x0ULL;

	volatile uint32* vectorTableSource = (uint32*)ppuInterfaceData.ppuCopyTable.sourceAddress;
	volatile uint32* vectorTableDest = (uint32*)ppuInterfaceData.ppuCopyTable.destinationAddress;
	/* Copy vector table of PPU to start of CSM
	 * Start of CSM is the default address for the reset vector
	 * Changing the default reset vector address requires a PPU kernel rest
	 * and has impact on the on the debugging */
	while((void *)vectorTableSource < (void *)((uint32)ppuInterfaceData.ppuCopyTable.sourceAddress + ppuInterfaceData.ppuCopyTable.size))
	{
		*vectorTableDest++ = *vectorTableSource++;
	}

	/* Enable SRNs for PPU */
	IfxSrc_init(&MODULE_SRC.PPU.ICI, IfxSrc_Tos_ppu, (Ifx_Priority)0, (IfxSrc_VmId)0);

	IfxPpucCore_startCore(&MODULE_PPU);
	//IfxPpucCore_configureCoreAndRun(&MODULE_PPU, (uint32)__PPU_CSM_START);

    /* Wait until PPU has booted up */
    while(PpuInterface_isPpuBootedUp() == FALSE);
}

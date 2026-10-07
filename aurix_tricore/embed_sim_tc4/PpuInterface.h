/**
 * \file PpuInterface.h
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

#ifndef PPUINTERFACE_H_
#define PPUINTERFACE_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Ifx_Cfg.h"
#include "IfxSrc.h"


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/


/*********************************************************************************************************************/
/*----------------------------------------------------Data Types-----------------------------------------------------*/
/*********************************************************************************************************************/


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/
void PpuInterface_startPpu(void);                                                     /* Configure and Start the PPU */

/* API to signal that PPU bootup is done.
 * This is done by triggering the PPUC Inter core communication interrupt up.
 * If one of the TriCores are configured to receive the interrupt (or poll),
 * then such communication is established.
 */
IFX_INLINE boolean PpuInterface_acknowledgePpuSignal(void)
{
    boolean requested= IfxSrc_isRequested(&MODULE_SRC.GPSR[0].SR0);
    if(requested)
    {
        IfxSrc_clearRequest(&MODULE_SRC.GPSR[0].SR0);
    }
    return requested;
}

/* API to acknowledge that PPU boot-up is done.
 *
 */
IFX_INLINE boolean PpuInterface_isPpuBootedUp(void)
{
    return PpuInterface_acknowledgePpuSignal();
}

/* API to synchronize with PPU.
 * This is done by triggering the PPUC Inter-core communication interrupt.
 * PPU is configured to receive the interrupt (or poll), then such communication is established.
 */
IFX_INLINE void PpuInterface_signalToPpu(void)
{
    /* Wait until PPU takes up last signal */
    while(IfxSrc_isRequested(&MODULE_SRC.PPU.ICI));
    IfxSrc_setRequest(&MODULE_SRC.PPU.ICI);
}

#endif /* PPUINTERFACE_H_ */

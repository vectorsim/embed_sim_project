/**********************************************************************************************************************
 * \file SharedData.h
 * \copyright Copyright (c) 2026 Infineon Technologies AG. All rights reserved.
 *
 *
 *
 *                                 IMPORTANT NOTICE
 *
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

#ifndef SHAREDDATA_H_
#define SHAREDDATA_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Ifx_Cfg.h"
#include "Ifx_Types.h"
#include "Compilers.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
#define ppuInterfaceData (*((SharedData_PpuIfConstants *)__PPU_IF_CONSTANTS))

/*********************************************************************************************************************/
/*----------------------------------------------------Data Types-----------------------------------------------------*/
/*********************************************************************************************************************/
/* PPU SPECIFIC DATA TYPES */

/* Copy-table data of PPU vectors for TriCore to Copy to CSM memory.
 * Such a is located at known fixed address together with other shared pointers
 */
typedef struct
{
    void *const    sourceAddress;
    void *const    destinationAddress;
    const uint32   size;
} SharedData_PpuCopyTable;

/* Shared IPC data produced at PPU application
 * This structure is always located at RAM and updated during runtime.
 * Such data could also be pointers to other structures.
 * The members' position could also be altered based on project need.
 * Each members need to be initialized by PPU during its init process.
 * Update the API: PpuInterface_initSharedData(PpuInterface_PpuSharedData);
 *
 */

/* Forward declaration of the IPC struct */
typedef struct TcPpuIpcData TcPpuIpcData;


/* PPU Interface Constants
 * All the shared pointers that are between TriCore and PPU are listed as below
 * structure members. Please note that, the position of such data is fixed by the
 * linker command file of PPU project.
 */
typedef struct
{
    /* !!! Don't alter the position of structure members !!! */
	SharedData_PpuCopyTable  ppuCopyTable;     /**< \brief PPU Copy-table for TriCore*/
	TcPpuIpcData *const      ppuIpcDataPtr;    /**< \brief Data pointer for PPU IPC */
} SharedData_PpuIfConstants;


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/


/*-------------------------------------------------------------------------------------------------------------------*/
/* Exported pointers from PPU linker file.
 */
//extern unsigned int __PPU_IF_CONSTANTS[];                              /* FIxed memory location at NVM of PPU application*/
extern SharedData_PpuIfConstants __PPU_IF_CONSTANTS[]; // this location (pointer) will be filled by linker, which knows the address


#endif /* SHAREDDATA_H_ */

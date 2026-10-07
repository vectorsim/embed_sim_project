/**
 * \file AllowAccess.c
 * \brief This file contains the APIs for providing access to all the masters towards generally used resources.
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
#include "Ifx_Types.h"
#include "IfxApApu.h"
#include "IfxCpu_cfg.h"
#include "IfxCpu.h"
#include "IfxCpu_reg.h"
#include "IfxAdc.h"
#include "IfxPpucCore.h"
#include "IfxPpucStu.h"
#include "IfxEgtm.h"
#include "IfxQspi.h"
#include "IfxCan.h"
#include "IfxAsclin.h"
#include "IfxSent.h"
#include "IfxI2c.h"
#include "IfxClock.h"
#include "IfxSrc.h"
#include "IfxPort.h"
#include "IfxGeth.h"
#include "IfxPmsEvr.h"
#include "IfxHsphy.h"
#include "IfxDre.h"
#include "IfxEray.h"
#include "IfxDma.h"
#include "IfxFce.h"
#include "IfxHssl.h"
#include "IfxMsc.h"
#include "IfxPcie.h"
#include "IfxPsi5.h"
#include "IfxPsi5s.h"
#include "IfxSdmmc.h"
#include "IfxSent.h"
#include "IfxSmm.h"
#include "IfxStm.h"
#include "IfxVmt.h"
#include "IfxXspi.h"

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

typedef struct
{
    void *module;
    void *startAddr;
    void *endAddr;
} AllowAccess_MemoryRegion;


#define ALLOWACCESS_MASTER_BITS       (0xFFFFFFFFFFFFFFFFULL)
#define ALLOWACCESS_VM_BITS           (uint8)(0xFF)
#define ALLOWACCESS_PRS_BITS          (uint8)(0xFF)

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
const IfxApApu_ApuConfig g_apuConfig=
{
   .wraTagId= (uint32)ALLOWACCESS_MASTER_BITS,
   .wrbTagId= (uint32)(ALLOWACCESS_MASTER_BITS >> 32),
   .rdaTagId= (uint32)ALLOWACCESS_MASTER_BITS,
   .rdbTagId= (uint32)(ALLOWACCESS_MASTER_BITS >> 32),
   .vmWrId= ALLOWACCESS_VM_BITS,
   .vmRdId= ALLOWACCESS_VM_BITS,
   .prsWrId= ALLOWACCESS_PRS_BITS,
   .prsRdId= ALLOWACCESS_PRS_BITS
};

static boolean accessGranted= FALSE;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/
void allowAccess(void)
{
	IfxApApu_ApuMemoryConfig memConfig;
	memConfig.apuConfig= (IfxApApu_ApuConfig *)&g_apuConfig;
	IfxApApu_configureAccessToLmus(&memConfig);
	IfxApApu_configureAccessToDlmus(&memConfig);
	IfxApApu_configureAccessToDsprs(&memConfig);
	IfxApApu_configureAccessToPsprs(&memConfig);
    IfxCpu_configureAccessToCpus((IfxApApu_ApuConfig *)&g_apuConfig);
	IfxStm_configureAccessToStms((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxAdc_configureAccessToAdc((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxPpucCore_configureAccessToPpuc((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxPpucCore_configureAccessToPpucCsm((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxPpucCore_configureAccessToPpucVccm((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxPpucStu_configureAccessToPpucStuDmi((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxEgtm_configureAccessToEgtms((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxQspi_configureAccessToQspis((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxCan_configureAccessToCans((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxAsclin_configureAccessToAsclins((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxSent_configureAccessToSents((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxI2c_configureAccessToI2cs((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxClock_configureAccessToClock((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxSrc_configureAccessToSrcs((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxPort_configureAccessToPorts((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxGeth_configureAccessToGeth((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxPmsEvr_configureAccessToPms((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxHsphy_configureAccessToHsphy((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxDma_configureAccessToDmas((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxFce_configureAccessToFce((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxDre_configureAccessToDre((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxEray_configureAccessToErays((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxHssl_configureAccessToHscts((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxHssl_configureAccessToHssls((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxMsc_configureAccessToMscs((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxPcie_configureAccessToPcies((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxPsi5_configureAccessToPsi5((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxPsi5s_configureAccessToAllPsi5s((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxSdmmc_configureAccessToSdmmc((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxSmm_configureAccessToSmm((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxStm_configureAccessToStms((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxVmt_configureAccessToVmts((IfxApApu_ApuConfig *)&g_apuConfig);
    IfxXspi_configureAccessToXspi((IfxApApu_ApuConfig *)&g_apuConfig);

    accessGranted= TRUE;
}

boolean isAccessGranted(void)
{
    return accessGranted;
}

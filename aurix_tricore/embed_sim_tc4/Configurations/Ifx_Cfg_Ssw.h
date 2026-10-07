 /**********************************************************************************************************************
 * \file Ifx_Cfg_Ssw.h
 * \brief Configuration header file for the Startup Software
 *
 * \copyright Copyright (c) 2026 Infineon Technologies AG. All rights reserved.
 *
 * $Date: 2023-10-12 09:17:32
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

#ifndef IFX_CFG_SSW_H
#define IFX_CFG_SSW_H 1

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Ifx_Cfg.h"
#include "Ifx_Ssw.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
/* Set this macro to 1, to configure the PMS EVR. */
#ifndef IFX_CFG_SSW_ENABLE_PMS_INIT
#define IFX_CFG_SSW_ENABLE_PMS_INIT                      (0U)
#endif

/* Set this macro to 1, to check the PMS EVR configuration. */
#ifndef IFX_CFG_SSW_ENABLE_PMS_INIT_CHECK
#define IFX_CFG_SSW_ENABLE_PMS_INIT_CHECK                (0U)
#endif

/* Set this macro to 1, to do LBIST checks. */
#ifndef IFX_CFG_SSW_ENABLE_LBIST
#define IFX_CFG_SSW_ENABLE_LBIST                         (0U)
#endif

/* Set this macro to 1, to do MONBIST checks. */
#ifndef IFX_CFG_SSW_ENABLE_MONBIST
#define IFX_CFG_SSW_ENABLE_MONBIST                       (0U)
#endif

/* Set this macro to 1, to shutdown unused power domains. */
#ifndef IFX_CFG_SSW_ENABLE_SHUTDOWN_UNUSED_PWRDOMAINS
#define IFX_CFG_SSW_ENABLE_SHUTDOWN_UNUSED_PWRDOMAINS    (0U)
#endif

/* Set this macro to 1, to do MBIST for DSPR0-5 and DMARAM. */
#ifndef IFX_CFG_SSW_ENABLE_MBIST_DSPRS_DMARAM
#define IFX_CFG_SSW_ENABLE_MBIST_DSPRS_DMARAM            (0U)
#endif

/* Set this macro to 1, to check for Stable clock e.g. MMIC. */
#ifndef IFX_CFG_SSW_ENABLE_XTALSRC_CHECK
#define IFX_CFG_SSW_ENABLE_XTALSRC_CHECK                 (0U)
#endif

/* Set this macro to 1, to initialize the PLL. */
#ifndef IFX_CFG_SSW_ENABLE_PLL_INIT
#define IFX_CFG_SSW_ENABLE_PLL_INIT                      (1U)
#endif

/* Set this macro to 1, to configure PMIC Voltage Shift. */
#ifndef IFX_CFG_SSW_ENABLE_PMIC_VOLTAGESHIFT
#define IFX_CFG_SSW_ENABLE_PMIC_VOLTAGESHIFT             (0U)
#endif

/* Set this macro to 1, to do MBIST checks. */
#ifndef IFX_CFG_SSW_ENABLE_MBIST
#define IFX_CFG_SSW_ENABLE_MBIST                         (0U)
#endif

/* Set this macro to 1, to do AUTOSAR Services start. */
#ifndef IFX_CFG_SSW_ENABLE_AUTOSAR_SERVICESTART
#define IFX_CFG_SSW_ENABLE_AUTOSAR_SERVICESTART          (0U)
#endif

/* Set this macro to 1, to do SMU Alarm handling. */
#ifndef IFX_CFG_SSW_ENABLE_SMU
#define IFX_CFG_SSW_ENABLE_SMU                           (0U)
#endif

/* Set this macro to 1, to do STL startup tests. */
#ifndef IFX_CFG_SSW_ENABLE_SAFETY_LIBRARY_TESTS
#define IFX_CFG_SSW_ENABLE_SAFETY_LIBRARY_TESTS          (0U)
#endif

/* Set this macro to 1, to initialize the AP. */
#ifndef IFX_CFG_SSW_ENABLE_AP_INIT
#define IFX_CFG_SSW_ENABLE_AP_INIT                       (1U)
#endif
/* Set this macro to 1, to do Key-off LBIST checks. */
#ifndef IFX_CFG_SSW_ENABLE_KEYOFF_LBIST
#define IFX_CFG_SSW_ENABLE_KEYOFF_LBIST                  (0U)
#endif
/* Set this macro to 1, to do Key-off MBIST checks. */
#ifndef IFX_CFG_SSW_ENABLE_KEYOFF_MBIST
#define IFX_CFG_SSW_ENABLE_KEYOFF_MBIST                  (0U)
#endif
/* Set this macro to 1, to do Key-off MBIST for DSPR0-5 and DMARAM. */
#ifndef IFX_CFG_SSW_ENABLE_KEYOFF_MBIST_DSPRS_DMARAM
#define IFX_CFG_SSW_ENABLE_KEYOFF_MBIST_DSPRS_DMARAM     (0U)
#endif
/* Set this macro to 1, to do Key-off STL startup tests. */
#ifndef IFX_CFG_SSW_ENABLE_KEYOFF_SAFETY_LIBRARY_TESTS
#define IFX_CFG_SSW_ENABLE_KEYOFF_SAFETY_LIBRARY_TESTS   (0U)
#endif

#endif /* IFX_CFG_SSW_H */

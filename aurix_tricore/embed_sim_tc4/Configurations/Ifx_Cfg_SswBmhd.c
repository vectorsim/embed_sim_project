 /**********************************************************************************************************************
 * \file Ifx_Cfg_SswBmhd.c
 * \brief This file contains the Bmhd for Aurix Plus.
 *
 * \copyright Copyright (c) 2026 Infineon Technologies AG. All rights reserved.
 *
 * $Date: 2023-10-27 12:27:41
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
 
/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Ifx_Ssw.h"

/*********************************************************************************************************************/
/*--------------------------------------------------BMHD constants---------------------------------------------------*/
/*********************************************************************************************************************/

#if defined(__TASKING__)
#pragma section farrom "bmhd_0_orig"
#elif defined(__HIGHTEC__) && !defined(__clang__)
#pragma section
#pragma section ".bmhd_0_orig" a
#elif defined(__GNUC__) && !defined(__HIGHTEC__)
#pragma section
#pragma section ".bmhd_0_orig" a
#elif defined(__HIGHTEC__) && defined(__clang__)
#pragma clang section rodata=".bmhd_0_orig"
#elif defined(__DCC__)
#pragma section CONST ".bmhd_0_orig" far-absolute R
#elif defined(__ghs__)
#pragma ghs section rodata= ".bmhd_0_orig"
#endif
const Ifx_Ssw_Bmhd bmhd_0_orig = {
    .sal    = 0xAE404800U,           /* 0x000: System address of BMHD                                       */
    .bmi    = 0x0FU,                 /* 0x004: .bmi: Boot Mode Index (BMI)                                  */
    .bmhdid = 0xB359U,               /* 0x006: .bmhdid: Boot Mode Header ID (CODE) = B359H                  */
    .stad   = 0xA0000000U,           /* 0x008: .stad: User Code start address                               */
    .crc    = 0x90380C26U,           /* 0x00C: .crc: Check Result for the BMI Header (offset 000H - 007H)   */
    .pw     = {
        0x00000000U,                 /* 0x100: .pw0: Password protection word 0 (lsw)                       */
        0x00000000U,                 /* 0x104: .pw1: Password protection word 1                             */
        0x00000000U,                 /* 0x108: .pw2: Password protection word 2                             */
        0x00000000U,                 /* 0x10C: .pw3: Password protection word 3                             */
        0x00000000U,                 /* 0x110: .pw4: Password protection word 4                             */
        0x00000000U,                 /* 0x114: .pw5: Password protection word 5                             */
        0x00000000U,                 /* 0x118: .pw6: Password protection word 6                             */
        0x00000000U,                 /* 0x11C: .pw7: Password protection word 7 (msw)                       */
    },
    .confirmation     = 0x43211234U, /* 0x1F0: .confirmation: 32-bit CODE, (always same)                    */
    .confirmationCopy = 0x43211234U  /* 0x1F0: .confirmation: 32-bit CODE, (always same)                    */
};

#if defined(__TASKING__)
#pragma section farrom "bmhd_1_orig"
#elif defined(__HIGHTEC__) && !defined(__clang__)
#pragma section
#pragma section ".bmhd_1_orig" a
#elif defined(__GNUC__) && !defined(__HIGHTEC__)
#pragma section
#pragma section ".bmhd_1_orig" a
#elif defined(__HIGHTEC__) && defined(__clang__)
#pragma clang section rodata=".bmhd_1_orig"
#elif defined(__DCC__)
#pragma section CONST ".bmhd_1_orig" far-absolute R
#elif defined(__ghs__)
#pragma ghs section rodata= ".bmhd_1_orig"
#endif
const Ifx_Ssw_Bmhd bmhd_1_orig = {
    .sal    = 0xAE405000U,           /* 0x000: System address of BMHD                                       */
    .bmi    = 0x0FU,                 /* 0x000: .bmi: Boot Mode Index (BMI)                                  */
    .bmhdid = 0xB359U,               /* 0x002: .bmhdid: Boot Mode Header ID (CODE) = B359H                  */
    .stad   = 0xA0000000U,           /* 0x004: .stad: User Code start address                               */
    .crc    = 0xA441CD3AU,           /* 0x008: .crc: Check Result for the BMI Header (offset 000H - 007H)   */
    .pw     = {
        0x00000000U,                 /* 0x100: .pw0: Password protection word 0 (lsw)                       */
        0x00000000U,                 /* 0x104: .pw1: Password protection word 1                             */
        0x00000000U,                 /* 0x108: .pw2: Password protection word 2                             */
        0x00000000U,                 /* 0x10C: .pw3: Password protection word 3                             */
        0x00000000U,                 /* 0x110: .pw4: Password protection word 4                             */
        0x00000000U,                 /* 0x114: .pw5: Password protection word 5                             */
        0x00000000U,                 /* 0x118: .pw6: Password protection word 6                             */
        0x00000000U,                 /* 0x11C: .pw7: Password protection word 7 (msw)                       */
    },
    .confirmation     = 0x43211234U, /* 0x1F0: .confirmation: 32-bit CODE, (always same)                    */
    .confirmationCopy = 0x43211234U  /* 0x1F0: .confirmation: 32-bit CODE, (always same)                    */
};


#if defined(__TASKING__)
#pragma section farrom "bmhd_2_orig"
#elif defined(__HIGHTEC__) && !defined(__clang__)
#pragma section
#pragma section ".bmhd_2_orig" a
#elif defined(__GNUC__) && !defined(__HIGHTEC__)
#pragma section
#pragma section ".bmhd_2_orig" a
#elif defined(__HIGHTEC__) && defined(__clang__)
#pragma clang section rodata=".bmhd_2_orig"
#elif defined(__DCC__)
#pragma section CONST ".bmhd_2_orig" far-absolute R
#elif defined(__ghs__)
#pragma ghs section rodata= ".bmhd_2_orig"
#endif
const Ifx_Ssw_Bmhd bmhd_2_orig = {
    .sal    = 0xAE405800U,           /* 0x000: System address of BMHD                                       */
    .bmi    = 0x0FU,                 /* 0x000: .bmi: Boot Mode Index (BMI)                                  */
    .bmhdid = 0xB359U,               /* 0x002: .bmhdid: Boot Mode Header ID (CODE) = B359H                  */
    .stad   = 0xA0000000U,           /* 0x004: .stad: User Code start address                               */
    .crc    = 0xB7968DCEU,           /* 0x008: .crc: Check Result for the BMI Header (offset 000H - 007H)   */
    .pw     = {
        0x00000000U,                 /* 0x100: .pw0: Password protection word 0 (lsw)                       */
        0x00000000U,                 /* 0x104: .pw1: Password protection word 1                             */
        0x00000000U,                 /* 0x108: .pw2: Password protection word 2                             */
        0x00000000U,                 /* 0x10C: .pw3: Password protection word 3                             */
        0x00000000U,                 /* 0x110: .pw4: Password protection word 4                             */
        0x00000000U,                 /* 0x114: .pw5: Password protection word 5                             */
        0x00000000U,                 /* 0x118: .pw6: Password protection word 6                             */
        0x00000000U,                 /* 0x11C: .pw7: Password protection word 7 (msw)                       */
    },
    .confirmation     = 0x43211234U, /* 0x1F0: .confirmation: 32-bit CODE, (always same)                    */
    .confirmationCopy = 0x43211234U  /* 0x1F0: .confirmation: 32-bit CODE, (always same)                    */
};

#if defined(__TASKING__)
#pragma section farrom "bmhd_3_orig"
#elif defined(__HIGHTEC__) && !defined(__clang__)
#pragma section
#pragma section ".bmhd_3_orig" a
#elif defined(__GNUC__) && !defined(__HIGHTEC__)
#pragma section
#pragma section ".bmhd_3_orig" a
#elif defined(__HIGHTEC__) && defined(__clang__)
#pragma clang section rodata=".bmhd_3_orig"
#elif defined(__DCC__)
#pragma section CONST ".bmhd_3_orig" far-absolute R
#elif defined(__ghs__)
#pragma ghs section rodata= ".bmhd_3_orig"
#endif
const Ifx_Ssw_Bmhd bmhd_3_orig = {
    .sal    = 0xAE406000U,           /* 0x000: System address of BMHD                                       */
    .bmi    = 0x0FU,                 /* 0x000: .bmi: Boot Mode Index (BMI)                                  */
    .bmhdid = 0xB359U,               /* 0x002: .bmhdid: Boot Mode Header ID (CODE) = B359H                  */
    .stad   = 0xA0000000U,           /* 0x004: .stad: User Code start address                               */
    .crc    = 0xCCB24F02U,           /* 0x008: .crc: Check Result for the BMI Header (offset 000H - 007H)   */
    .pw     = {
        0x00000000U,                 /* 0x100: .pw0: Password protection word 0 (lsw)                       */
        0x00000000U,                 /* 0x104: .pw1: Password protection word 1                             */
        0x00000000U,                 /* 0x108: .pw2: Password protection word 2                             */
        0x00000000U,                 /* 0x10C: .pw3: Password protection word 3                             */
        0x00000000U,                 /* 0x110: .pw4: Password protection word 4                             */
        0x00000000U,                 /* 0x114: .pw5: Password protection word 5                             */
        0x00000000U,                 /* 0x118: .pw6: Password protection word 6                             */
        0x00000000U,                 /* 0x11C: .pw7: Password protection word 7 (msw)                       */
    },
    .confirmation     = 0x43211234U, /* 0x1F0: .confirmation: 32-bit CODE, (always same)                    */
    .confirmationCopy = 0x43211234U  /* 0x1F0: .confirmation: 32-bit CODE, (always same)                    */
};


#if defined(__TASKING__)
#pragma section farrom "usercfg_0_orig"
#elif defined(__HIGHTEC__) && !defined(__clang__)
#pragma section
#pragma section ".usercfg_0_orig" a
#elif defined(__GNUC__) && !defined(__HIGHTEC__)
#pragma section
#pragma section ".usercfg_0_orig" a
#elif defined(__HIGHTEC__) && defined(__clang__)
#pragma clang section rodata=".usercfg_0_orig"
#elif defined(__DCC__)
#pragma section CONST ".usercfg_0_orig" far-absolute R
#elif defined(__ghs__)
#pragma ghs section rodata= ".usercfg_0_orig"
#endif
const Ifx_Ssw_UserCfg usercfg_0_orig = {
    .rtcUserCfg     = 0xAE408800U,
    .lockstepEna    = 0x3FU,
    .smmEsr0CntCtrl = 0x0U,
    .pmsPadEsr2Con  = 0x0U,
    .ramInit        = 0x0U,
    .lbistExe       = 0x0U,
    .hostRamInit    = 0x2U,
    .swapEna        = 0x5U,
    .addrrEna       = 0x0U,
    .crcRtcUserCfg  = 0x51DCC217U,
    .pw             = {
        0x00000000U,                /* 0x100: .pw0: Password protection word 0 (lsw)   */
        0x00000000U,                /* 0x104: .pw1: Password protection word 1         */
        0x00000000U,                /* 0x108: .pw2: Password protection word 2         */
        0x00000000U,                /* 0x10C: .pw3: Password protection word 3         */
        0x00000000U,                /* 0x110: .pw4: Password protection word 4         */
        0x00000000U,                /* 0x114: .pw5: Password protection word 5         */
        0x00000000U,                /* 0x118: .pw6: Password protection word 6         */
        0x00000000U,                /* 0x11C: .pw7: Password protection word 7 (msw)   */
    },
    .confirmation   = 0x43211234U
};


#if defined(__TASKING__)
#pragma section farrom "usercfg_0_copy"
#elif defined(__HIGHTEC__) && !defined(__clang__)
#pragma section
#pragma section ".usercfg_0_copy" a
#elif defined(__GNUC__) && !defined(__HIGHTEC__)
#pragma section
#pragma section ".usercfg_0_copy" a
#elif defined(__HIGHTEC__) && defined(__clang__)
#pragma clang section rodata=".usercfg_0_copy"
#elif defined(__DCC__)
#pragma section CONST ".usercfg_0_copy" far-absolute R
#elif defined(__ghs__)
#pragma ghs section rodata= ".usercfg_0_copy"
#endif
const Ifx_Ssw_UserCfg usercfg_0_copy = {
    .rtcUserCfg     = 0xAE409000U,
    .lockstepEna    = 0x3FU,
    .smmEsr0CntCtrl = 0x0U,
    .pmsPadEsr2Con  = 0x0U,
    .ramInit        = 0x0U,
    .lbistExe       = 0x0U,
    .hostRamInit    = 0x2U,
    .swapEna        = 0x5U,
    .addrrEna       = 0x0U,
    .crcRtcUserCfg  = 0x530FCBECU,
    .pw             = {
        0x00000000U,                /* 0x100: .pw0: Password protection word 0 (lsw)   */
        0x00000000U,                /* 0x104: .pw1: Password protection word 1         */
        0x00000000U,                /* 0x108: .pw2: Password protection word 2         */
        0x00000000U,                /* 0x10C: .pw3: Password protection word 3         */
        0x00000000U,                /* 0x110: .pw4: Password protection word 4         */
        0x00000000U,                /* 0x114: .pw5: Password protection word 5         */
        0x00000000U,                /* 0x118: .pw6: Password protection word 6         */
        0x00000000U,                /* 0x11C: .pw7: Password protection word 7 (msw)   */
    },
    .confirmation   = 0x43211234U
};
#if defined(__TASKING__)
#elif defined(__HIGHTEC__) && !defined(__clang__)
#pragma section
#elif defined(__GNUC__) && !defined(__HIGHTEC__)
#pragma section
#endif
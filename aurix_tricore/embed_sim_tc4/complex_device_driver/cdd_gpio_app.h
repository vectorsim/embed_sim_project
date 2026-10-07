/**********************************************************************************************************************
 * \file      cdd_gpio_app.h
 * \brief     Public interface for the GPIO application module.
 *
 * \details   Bare-metal GPIO / alternate-output pin configuration for TC49x.
 *            Each function hands the pad to the SCR (PCSRx = 0x1) and sets
 *            the pad as an alternate output through PADCFGx.DRVCFG. TOUTSEL
 *            is not touched here; the eGTM routing is handled elsewhere.
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter -> Uppercase)
 *              - Output pointers: PascalCasePtr
 *              - Local variables: lowerPascalCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   1.0.0
 * \date      2026-10-07
 * \author    Paul Abraham
 *            EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef COMPLEX_DEVICE_DRIVER_CDD_GPIO_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_GPIO_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Configure P02.0 as ATOM0 CH0 master channel output.
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtomMasterP02_00(void);

/**
 * \brief   Configure P02.5 as phase U high-side ATOM output.
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtomUhP02_05(void);

/**
 * \brief   Configure P00.2 as phase U low-side ATOM output.
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtomUlP00_02(void);

/**
 * \brief   Configure P02.6 as phase V high-side ATOM output.
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtomVhP02_06(void);

/**
 * \brief   Configure P00.4 as phase V low-side ATOM output.
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtomVlP00_04(void);

/**
 * \brief   Configure P02.7 as phase W high-side ATOM output.
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtomWhP02_07(void);

/**
 * \brief   Configure P00.6 as phase W low-side ATOM output.
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtomWlP00_06(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_GPIO_APP_H_ */

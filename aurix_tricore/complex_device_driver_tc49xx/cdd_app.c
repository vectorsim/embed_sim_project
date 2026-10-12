/**********************************************************************************************************************
 * \file      cdd_app.c
 * \brief     Implementation of the CDD layer top-level init / start.
 *
 * \note      EmbedSim naming convention is used throughout.
 *
 * \version   1.0.0
 * \date      2026-10-04
 * \author    Paul Abraham
 *            EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#include "cdd_app.h"
#include "cdd_sys_util.h"
#include "cdd_egtm_app.h"
#include "cdd_gpio_app.h"
#include "cdd_stm_app.h"
#include "cdd_adc_app.h"

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
CddInverterCtrl_T  Inv_G;

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

void Cdd_Init(void)
{
    /* Initialize Heart Beat LED */
    CddGpio_Init_P03_10_Led_HeartBeat();

    /* Initialize ADC Module */
    CddAdc_Init();

    /* Initialize System Timer Module */
    CddStm_Init();

    /* Initialize EGTM Module */
    CddEgtm_Init();
}


void Cdd_Start(void)
{
    CddEgtm_Start();
}

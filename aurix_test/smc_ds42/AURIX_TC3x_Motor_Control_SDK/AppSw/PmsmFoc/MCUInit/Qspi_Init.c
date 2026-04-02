/*
 * \file Qspi_Init.c
 * \brief QSPI initialization for PMSM FOC motor control applications.
 * \details
 * This file implements the initialization of QSPI2 and QSPI4 modules used in
 * PMSM FOC motor control. It configures the SPI communication settings,
 * including DMA usage, interrupt priorities, and SPI pins.
 *
 * \copyright
 * Copyright (c) 2019 Infineon Technologies AG. All rights reserved.
 *
 *                            IMPORTANT NOTICE
 *
 * Use of this file is subject to the terms of use agreed between (i) you or
 * the company in which ordinary course of business you are acting and (ii)
 * Infineon Technologies AG or its licensees. If and as long as no such terms
 * of use are agreed, use of this file is subject to following:
 *
 * Boost Software License - Version 1.0 - August 17th, 2003
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "Qspi_Init.h"

/******************************************************************************/
/*-------------------------------Global Variables-----------------------------*/
/******************************************************************************/

IfxQspi_SpiMaster           spiMasterQspi2;
IfxQspi_SpiMaster           spiMasterQspi4;

/******************************************************************************/
/*-----------------------------Private Functions------------------------------*/
/******************************************************************************/

/**
 * \brief Initialize QSPI2 module.
 *
 * \details
 * Configures QSPI2 for serial communication, including pin configurations,
 * maximum baudrate, and interrupt priorities. DMA settings are applied if enabled.
 */
IFX_INLINE void PmsmFoc_Qspi_initQspi2(void)
{
    IfxQspi_SpiMaster_Config spiConfig;
    IfxQspi_SpiMaster_initModuleConfig(&spiConfig, &SPI_2_MODULE);

    const IfxQspi_SpiMaster_Pins spiPins =
    {
        &SPI_2_CLOCK_PIN, IfxPort_OutputMode_pushPull,
        &SPI_2_MOSI_PIN, IfxPort_OutputMode_pushPull,
        &SPI_2_MISO_PIN, IfxPort_InputMode_pullDown,
        IfxPort_PadDriver_cmosAutomotiveSpeed1
    };

    spiConfig.base.maximumBaudrate = 50.0e6;
    spiConfig.base.txPriority = INTERRUPT_PRIORITY_QSPI2_TX;
    spiConfig.base.rxPriority = INTERRUPT_PRIORITY_QSPI2_RX;
    spiConfig.base.erPriority = INTERRUPT_PRIORITY_QSPI2_ERR;
    spiConfig.base.isrProvider = SPI_2_HOST_CPU;

#if (SPI_2_USE_DMA == TRUE)
    spiConfig.dma.txDmaChannelId = SPI_2_TX_DMA_CH;
    spiConfig.dma.rxDmaChannelId = SPI_2_RX_DMA_CH;
    spiConfig.dma.useDma = TRUE;
#endif

    spiConfig.pins = &spiPins;
    IfxQspi_SpiMaster_initModule(&spiMasterQspi2, &spiConfig);
}

/**
 * \brief Initialize QSPI4 module.
 *
 * \details
 * Configures QSPI4 for serial communication, including pin configurations,
 * maximum baudrate, and interrupt priorities. DMA settings are applied if enabled.
 */
IFX_INLINE void PmsmFoc_Qspi_initQspi4(void)
{
    IfxQspi_SpiMaster_Config spiConfig;
    IfxQspi_SpiMaster_initModuleConfig(&spiConfig, &SPI_4_MODULE);

    const IfxQspi_SpiMaster_Pins spiPins =
    {
        &SPI_4_CLOCK_PIN, IfxPort_OutputMode_pushPull,
        &SPI_4_MOSI_PIN, IfxPort_OutputMode_pushPull,
        &SPI_4_MISO_PIN, IfxPort_InputMode_pullDown,
        IfxPort_PadDriver_cmosAutomotiveSpeed1
    };

    spiConfig.base.maximumBaudrate = 50.0e6;
    spiConfig.base.txPriority = INTERRUPT_PRIORITY_QSPI4_TX;
    spiConfig.base.rxPriority = INTERRUPT_PRIORITY_QSPI4_RX;
    spiConfig.base.erPriority = INTERRUPT_PRIORITY_QSPI4_ERR;
    spiConfig.base.isrProvider = SPI_4_HOST_CPU;

#if (SPI_4_USE_DMA == TRUE)
    spiConfig.dma.txDmaChannelId = SPI_4_TX_DMA_CH;
    spiConfig.dma.rxDmaChannelId = SPI_4_RX_DMA_CH;
    spiConfig.dma.useDma = TRUE;
#endif

    spiConfig.pins = &spiPins;
    IfxQspi_SpiMaster_initModule(&spiMasterQspi4, &spiConfig);
}

/******************************************************************************/
/*-----------------------------Public Functions-------------------------------*/
/******************************************************************************/

/**
 * \brief Initialize QSPI2 and QSPI4 modules.
 *
 * \details
 * Calls the internal functions to initialize QSPI2 and QSPI4 modules, ensuring
 * they are configured for PMSM FOC motor control communication.
 */
void PmsmFoc_Qspi_initQspi(void)
{
    PmsmFoc_Qspi_initQspi2();
    PmsmFoc_Qspi_initQspi4();
}

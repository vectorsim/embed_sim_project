################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.c" 

COMPILED_SRCS += \
"Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.src" 

C_DEPS += \
"./Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.d" 

OBJS += \
"Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.src":"../Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.c" "Libraries/oneeye_lib/xmc/xmcdaveapp/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.o":"Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.src" "Libraries/oneeye_lib/xmc/xmcdaveapp/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-oneeye_lib-2f-xmc-2f-xmcdaveapp

clean-Libraries-2f-oneeye_lib-2f-xmc-2f-xmcdaveapp:
	-$(RM) ./Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.d ./Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.o ./Libraries/oneeye_lib/xmc/xmcdaveapp/ifx_oe_al_xmc_xmcdaveapp_uartdpipe.src

.PHONY: clean-Libraries-2f-oneeye_lib-2f-xmc-2f-xmcdaveapp


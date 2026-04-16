################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../OS/FreeRTOS/portable/TriCore/port.c" 

COMPILED_SRCS += \
"OS/FreeRTOS/portable/TriCore/port.src" 

C_DEPS += \
"./OS/FreeRTOS/portable/TriCore/port.d" 

OBJS += \
"OS/FreeRTOS/portable/TriCore/port.o" 


# Each subdirectory must supply rules for building sources it contributes
"OS/FreeRTOS/portable/TriCore/port.src":"../OS/FreeRTOS/portable/TriCore/port.c" "OS/FreeRTOS/portable/TriCore/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"OS/FreeRTOS/portable/TriCore/port.o":"OS/FreeRTOS/portable/TriCore/port.src" "OS/FreeRTOS/portable/TriCore/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-OS-2f-FreeRTOS-2f-portable-2f-TriCore

clean-OS-2f-FreeRTOS-2f-portable-2f-TriCore:
	-$(RM) ./OS/FreeRTOS/portable/TriCore/port.d ./OS/FreeRTOS/portable/TriCore/port.o ./OS/FreeRTOS/portable/TriCore/port.src

.PHONY: clean-OS-2f-FreeRTOS-2f-portable-2f-TriCore


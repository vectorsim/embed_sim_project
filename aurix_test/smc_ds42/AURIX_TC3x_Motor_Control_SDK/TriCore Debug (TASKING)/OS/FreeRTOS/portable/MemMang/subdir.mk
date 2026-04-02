################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../OS/FreeRTOS/portable/MemMang/heap_4.c" 

COMPILED_SRCS += \
"OS/FreeRTOS/portable/MemMang/heap_4.src" 

C_DEPS += \
"./OS/FreeRTOS/portable/MemMang/heap_4.d" 

OBJS += \
"OS/FreeRTOS/portable/MemMang/heap_4.o" 


# Each subdirectory must supply rules for building sources it contributes
"OS/FreeRTOS/portable/MemMang/heap_4.src":"../OS/FreeRTOS/portable/MemMang/heap_4.c" "OS/FreeRTOS/portable/MemMang/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"OS/FreeRTOS/portable/MemMang/heap_4.o":"OS/FreeRTOS/portable/MemMang/heap_4.src" "OS/FreeRTOS/portable/MemMang/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-OS-2f-FreeRTOS-2f-portable-2f-MemMang

clean-OS-2f-FreeRTOS-2f-portable-2f-MemMang:
	-$(RM) ./OS/FreeRTOS/portable/MemMang/heap_4.d ./OS/FreeRTOS/portable/MemMang/heap_4.o ./OS/FreeRTOS/portable/MemMang/heap_4.src

.PHONY: clean-OS-2f-FreeRTOS-2f-portable-2f-MemMang


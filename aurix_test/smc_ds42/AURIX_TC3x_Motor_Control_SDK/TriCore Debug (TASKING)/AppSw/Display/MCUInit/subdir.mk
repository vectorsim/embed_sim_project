################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../AppSw/Display/MCUInit/Display_Qspi_Init.c" 

COMPILED_SRCS += \
"AppSw/Display/MCUInit/Display_Qspi_Init.src" 

C_DEPS += \
"./AppSw/Display/MCUInit/Display_Qspi_Init.d" 

OBJS += \
"AppSw/Display/MCUInit/Display_Qspi_Init.o" 


# Each subdirectory must supply rules for building sources it contributes
"AppSw/Display/MCUInit/Display_Qspi_Init.src":"../AppSw/Display/MCUInit/Display_Qspi_Init.c" "AppSw/Display/MCUInit/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/MCUInit/Display_Qspi_Init.o":"AppSw/Display/MCUInit/Display_Qspi_Init.src" "AppSw/Display/MCUInit/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-AppSw-2f-Display-2f-MCUInit

clean-AppSw-2f-Display-2f-MCUInit:
	-$(RM) ./AppSw/Display/MCUInit/Display_Qspi_Init.d ./AppSw/Display/MCUInit/Display_Qspi_Init.o ./AppSw/Display/MCUInit/Display_Qspi_Init.src

.PHONY: clean-AppSw-2f-Display-2f-MCUInit


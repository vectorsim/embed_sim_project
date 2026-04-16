################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/Service/CpuGeneric/If/SpiIf.c" 

COMPILED_SRCS += \
"Libraries/Service/CpuGeneric/If/SpiIf.src" 

C_DEPS += \
"./Libraries/Service/CpuGeneric/If/SpiIf.d" 

OBJS += \
"Libraries/Service/CpuGeneric/If/SpiIf.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/Service/CpuGeneric/If/SpiIf.src":"../Libraries/Service/CpuGeneric/If/SpiIf.c" "Libraries/Service/CpuGeneric/If/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Service/CpuGeneric/If/SpiIf.o":"Libraries/Service/CpuGeneric/If/SpiIf.src" "Libraries/Service/CpuGeneric/If/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-Service-2f-CpuGeneric-2f-If

clean-Libraries-2f-Service-2f-CpuGeneric-2f-If:
	-$(RM) ./Libraries/Service/CpuGeneric/If/SpiIf.d ./Libraries/Service/CpuGeneric/If/SpiIf.o ./Libraries/Service/CpuGeneric/If/SpiIf.src

.PHONY: clean-Libraries-2f-Service-2f-CpuGeneric-2f-If


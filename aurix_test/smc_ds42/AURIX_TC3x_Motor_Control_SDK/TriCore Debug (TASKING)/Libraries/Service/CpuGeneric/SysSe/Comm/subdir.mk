################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.c" \
"../Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.c" 

COMPILED_SRCS += \
"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.src" \
"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.src" 

C_DEPS += \
"./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.d" \
"./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.d" 

OBJS += \
"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.o" \
"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.src":"../Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.c" "Libraries/Service/CpuGeneric/SysSe/Comm/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.o":"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.src" "Libraries/Service/CpuGeneric/SysSe/Comm/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.src":"../Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.c" "Libraries/Service/CpuGeneric/SysSe/Comm/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.o":"Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.src" "Libraries/Service/CpuGeneric/SysSe/Comm/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-Comm

clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-Comm:
	-$(RM) ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.d ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.o ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Console.src ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.d ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.o ./Libraries/Service/CpuGeneric/SysSe/Comm/Ifx_Shell.src

.PHONY: clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-Comm


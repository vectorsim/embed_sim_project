################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.c" \
"../Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.c" 

COMPILED_SRCS += \
"Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.src" \
"Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.src" 

C_DEPS += \
"./Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.d" \
"./Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.d" 

OBJS += \
"Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.o" \
"Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.src":"../Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.c" "Libraries/Service/CpuGeneric/SysSe/Bsp/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.o":"Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.src" "Libraries/Service/CpuGeneric/SysSe/Bsp/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.src":"../Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.c" "Libraries/Service/CpuGeneric/SysSe/Bsp/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.o":"Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.src" "Libraries/Service/CpuGeneric/SysSe/Bsp/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-Bsp

clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-Bsp:
	-$(RM) ./Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.d ./Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.o ./Libraries/Service/CpuGeneric/SysSe/Bsp/Assert.src ./Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.d ./Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.o ./Libraries/Service/CpuGeneric/SysSe/Bsp/Bsp.src

.PHONY: clean-Libraries-2f-Service-2f-CpuGeneric-2f-SysSe-2f-Bsp


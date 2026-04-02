################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.c" 

COMPILED_SRCS += \
"Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.src" 

C_DEPS += \
"./Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.d" 

OBJS += \
"Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.src":"../Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.c" "Libraries/ExtDevDrivers/TLE9180/Tricore/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.o":"Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.src" "Libraries/ExtDevDrivers/TLE9180/Tricore/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-ExtDevDrivers-2f-TLE9180-2f-Tricore

clean-Libraries-2f-ExtDevDrivers-2f-TLE9180-2f-Tricore:
	-$(RM) ./Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.d ./Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.o ./Libraries/ExtDevDrivers/TLE9180/Tricore/TLE9180.src

.PHONY: clean-Libraries-2f-ExtDevDrivers-2f-TLE9180-2f-Tricore


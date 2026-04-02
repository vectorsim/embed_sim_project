################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../AppSw/OneEye/OneEye.c" 

COMPILED_SRCS += \
"AppSw/OneEye/OneEye.src" 

C_DEPS += \
"./AppSw/OneEye/OneEye.d" 

OBJS += \
"AppSw/OneEye/OneEye.o" 


# Each subdirectory must supply rules for building sources it contributes
"AppSw/OneEye/OneEye.src":"../AppSw/OneEye/OneEye.c" "AppSw/OneEye/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/OneEye/OneEye.o":"AppSw/OneEye/OneEye.src" "AppSw/OneEye/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-AppSw-2f-OneEye

clean-AppSw-2f-OneEye:
	-$(RM) ./AppSw/OneEye/OneEye.d ./AppSw/OneEye/OneEye.o ./AppSw/OneEye/OneEye.src

.PHONY: clean-AppSw-2f-OneEye


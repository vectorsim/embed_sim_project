################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.c" 

COMPILED_SRCS += \
"Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.src" 

C_DEPS += \
"./Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.d" 

OBJS += \
"Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.src":"../Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.c" "Libraries/oneeye_lib/arduino/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.o":"Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.src" "Libraries/oneeye_lib/arduino/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-oneeye_lib-2f-arduino

clean-Libraries-2f-oneeye_lib-2f-arduino:
	-$(RM) ./Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.d ./Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.o ./Libraries/oneeye_lib/arduino/ifx_oe_al_arduino.src

.PHONY: clean-Libraries-2f-oneeye_lib-2f-arduino


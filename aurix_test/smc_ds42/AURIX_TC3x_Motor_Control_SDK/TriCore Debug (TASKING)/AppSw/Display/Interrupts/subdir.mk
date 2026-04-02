################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../AppSw/Display/Interrupts/Display_Interrupts.c" 

COMPILED_SRCS += \
"AppSw/Display/Interrupts/Display_Interrupts.src" 

C_DEPS += \
"./AppSw/Display/Interrupts/Display_Interrupts.d" 

OBJS += \
"AppSw/Display/Interrupts/Display_Interrupts.o" 


# Each subdirectory must supply rules for building sources it contributes
"AppSw/Display/Interrupts/Display_Interrupts.src":"../AppSw/Display/Interrupts/Display_Interrupts.c" "AppSw/Display/Interrupts/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/Interrupts/Display_Interrupts.o":"AppSw/Display/Interrupts/Display_Interrupts.src" "AppSw/Display/Interrupts/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-AppSw-2f-Display-2f-Interrupts

clean-AppSw-2f-Display-2f-Interrupts:
	-$(RM) ./AppSw/Display/Interrupts/Display_Interrupts.d ./AppSw/Display/Interrupts/Display_Interrupts.o ./AppSw/Display/Interrupts/Display_Interrupts.src

.PHONY: clean-AppSw-2f-Display-2f-Interrupts


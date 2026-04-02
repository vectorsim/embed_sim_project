################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.c" \
"../AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.c" \
"../AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.c" \
"../AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.c" 

COMPILED_SRCS += \
"AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.src" \
"AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.src" \
"AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.src" \
"AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.src" 

C_DEPS += \
"./AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.d" \
"./AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.d" \
"./AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.d" \
"./AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.d" 

OBJS += \
"AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.o" \
"AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.o" \
"AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.o" \
"AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.o" 


# Each subdirectory must supply rules for building sources it contributes
"AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.src":"../AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.c" "AppSw/PmsmFoc/ControlModules/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.o":"AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.src" "AppSw/PmsmFoc/ControlModules/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.src":"../AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.c" "AppSw/PmsmFoc/ControlModules/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.o":"AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.src" "AppSw/PmsmFoc/ControlModules/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.src":"../AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.c" "AppSw/PmsmFoc/ControlModules/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.o":"AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.src" "AppSw/PmsmFoc/ControlModules/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.src":"../AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.c" "AppSw/PmsmFoc/ControlModules/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.o":"AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.src" "AppSw/PmsmFoc/ControlModules/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-AppSw-2f-PmsmFoc-2f-ControlModules

clean-AppSw-2f-PmsmFoc-2f-ControlModules:
	-$(RM) ./AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.d ./AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.o ./AppSw/PmsmFoc/ControlModules/PmsmFoc_Functions.src ./AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.d ./AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.o ./AppSw/PmsmFoc/ControlModules/PmsmFoc_Interface.src ./AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.d ./AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.o ./AppSw/PmsmFoc/ControlModules/PmsmFoc_SpeedControl.src ./AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.d ./AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.o ./AppSw/PmsmFoc/ControlModules/PmsmFoc_StateMachine.src

.PHONY: clean-AppSw-2f-PmsmFoc-2f-ControlModules


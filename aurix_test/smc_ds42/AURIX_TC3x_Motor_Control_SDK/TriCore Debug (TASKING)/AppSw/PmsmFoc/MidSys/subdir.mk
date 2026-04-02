################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.c" \
"../AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.c" \
"../AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.c" \
"../AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.c" \
"../AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.c" \
"../AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.c" 

COMPILED_SRCS += \
"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.src" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.src" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.src" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.src" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.src" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.src" 

C_DEPS += \
"./AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.d" \
"./AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.d" \
"./AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.d" \
"./AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.d" \
"./AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.d" \
"./AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.d" 

OBJS += \
"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.o" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.o" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.o" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.o" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.o" \
"AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.o" 


# Each subdirectory must supply rules for building sources it contributes
"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.src":"../AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.c" "AppSw/PmsmFoc/MidSys/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.o":"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.src" "AppSw/PmsmFoc/MidSys/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.src":"../AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.c" "AppSw/PmsmFoc/MidSys/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.o":"AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.src" "AppSw/PmsmFoc/MidSys/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.src":"../AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.c" "AppSw/PmsmFoc/MidSys/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.o":"AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.src" "AppSw/PmsmFoc/MidSys/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.src":"../AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.c" "AppSw/PmsmFoc/MidSys/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.o":"AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.src" "AppSw/PmsmFoc/MidSys/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.src":"../AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.c" "AppSw/PmsmFoc/MidSys/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.o":"AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.src" "AppSw/PmsmFoc/MidSys/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.src":"../AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.c" "AppSw/PmsmFoc/MidSys/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.o":"AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.src" "AppSw/PmsmFoc/MidSys/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-AppSw-2f-PmsmFoc-2f-MidSys

clean-AppSw-2f-PmsmFoc-2f-MidSys:
	-$(RM) ./AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.d ./AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.o ./AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentDcLinkSense.src ./AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.d ./AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.o ./AppSw/PmsmFoc/MidSys/PmsmFoc_CurrentThreeshuntSense.src ./AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.d ./AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.o ./AppSw/PmsmFoc/MidSys/PmsmFoc_PositionAndSpeedAcquisition.src ./AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.d ./AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.o ./AppSw/PmsmFoc/MidSys/PmsmFoc_PwmSvm.src ./AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.d ./AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.o ./AppSw/PmsmFoc/MidSys/PmsmFoc_SensorAdc.src ./AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.d ./AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.o ./AppSw/PmsmFoc/MidSys/PmsmFoc_VoltageSense.src

.PHONY: clean-AppSw-2f-PmsmFoc-2f-MidSys


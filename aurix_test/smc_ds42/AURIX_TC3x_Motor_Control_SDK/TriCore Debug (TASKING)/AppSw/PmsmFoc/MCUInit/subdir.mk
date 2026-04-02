################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../AppSw/PmsmFoc/MCUInit/Evadc_Init.c" \
"../AppSw/PmsmFoc/MCUInit/Gpt12_Init.c" \
"../AppSw/PmsmFoc/MCUInit/Gtm_Init.c" \
"../AppSw/PmsmFoc/MCUInit/Mcu_Init.c" \
"../AppSw/PmsmFoc/MCUInit/Qspi_Init.c" 

COMPILED_SRCS += \
"AppSw/PmsmFoc/MCUInit/Evadc_Init.src" \
"AppSw/PmsmFoc/MCUInit/Gpt12_Init.src" \
"AppSw/PmsmFoc/MCUInit/Gtm_Init.src" \
"AppSw/PmsmFoc/MCUInit/Mcu_Init.src" \
"AppSw/PmsmFoc/MCUInit/Qspi_Init.src" 

C_DEPS += \
"./AppSw/PmsmFoc/MCUInit/Evadc_Init.d" \
"./AppSw/PmsmFoc/MCUInit/Gpt12_Init.d" \
"./AppSw/PmsmFoc/MCUInit/Gtm_Init.d" \
"./AppSw/PmsmFoc/MCUInit/Mcu_Init.d" \
"./AppSw/PmsmFoc/MCUInit/Qspi_Init.d" 

OBJS += \
"AppSw/PmsmFoc/MCUInit/Evadc_Init.o" \
"AppSw/PmsmFoc/MCUInit/Gpt12_Init.o" \
"AppSw/PmsmFoc/MCUInit/Gtm_Init.o" \
"AppSw/PmsmFoc/MCUInit/Mcu_Init.o" \
"AppSw/PmsmFoc/MCUInit/Qspi_Init.o" 


# Each subdirectory must supply rules for building sources it contributes
"AppSw/PmsmFoc/MCUInit/Evadc_Init.src":"../AppSw/PmsmFoc/MCUInit/Evadc_Init.c" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MCUInit/Evadc_Init.o":"AppSw/PmsmFoc/MCUInit/Evadc_Init.src" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/MCUInit/Gpt12_Init.src":"../AppSw/PmsmFoc/MCUInit/Gpt12_Init.c" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MCUInit/Gpt12_Init.o":"AppSw/PmsmFoc/MCUInit/Gpt12_Init.src" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/MCUInit/Gtm_Init.src":"../AppSw/PmsmFoc/MCUInit/Gtm_Init.c" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MCUInit/Gtm_Init.o":"AppSw/PmsmFoc/MCUInit/Gtm_Init.src" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/MCUInit/Mcu_Init.src":"../AppSw/PmsmFoc/MCUInit/Mcu_Init.c" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MCUInit/Mcu_Init.o":"AppSw/PmsmFoc/MCUInit/Mcu_Init.src" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/MCUInit/Qspi_Init.src":"../AppSw/PmsmFoc/MCUInit/Qspi_Init.c" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/MCUInit/Qspi_Init.o":"AppSw/PmsmFoc/MCUInit/Qspi_Init.src" "AppSw/PmsmFoc/MCUInit/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-AppSw-2f-PmsmFoc-2f-MCUInit

clean-AppSw-2f-PmsmFoc-2f-MCUInit:
	-$(RM) ./AppSw/PmsmFoc/MCUInit/Evadc_Init.d ./AppSw/PmsmFoc/MCUInit/Evadc_Init.o ./AppSw/PmsmFoc/MCUInit/Evadc_Init.src ./AppSw/PmsmFoc/MCUInit/Gpt12_Init.d ./AppSw/PmsmFoc/MCUInit/Gpt12_Init.o ./AppSw/PmsmFoc/MCUInit/Gpt12_Init.src ./AppSw/PmsmFoc/MCUInit/Gtm_Init.d ./AppSw/PmsmFoc/MCUInit/Gtm_Init.o ./AppSw/PmsmFoc/MCUInit/Gtm_Init.src ./AppSw/PmsmFoc/MCUInit/Mcu_Init.d ./AppSw/PmsmFoc/MCUInit/Mcu_Init.o ./AppSw/PmsmFoc/MCUInit/Mcu_Init.src ./AppSw/PmsmFoc/MCUInit/Qspi_Init.d ./AppSw/PmsmFoc/MCUInit/Qspi_Init.o ./AppSw/PmsmFoc/MCUInit/Qspi_Init.src

.PHONY: clean-AppSw-2f-PmsmFoc-2f-MCUInit


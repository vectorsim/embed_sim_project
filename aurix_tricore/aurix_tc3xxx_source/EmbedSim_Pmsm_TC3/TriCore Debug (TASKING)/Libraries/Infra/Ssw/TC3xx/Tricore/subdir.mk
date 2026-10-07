################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.c" \
"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.c" \
"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.c" \
"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.c" \
"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.c" \
"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.c" \
"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.c" 

COMPILED_SRCS += \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.src" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.src" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.src" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.src" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.src" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.src" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.src" 

C_DEPS += \
"./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.d" \
"./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.d" \
"./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.d" \
"./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.d" \
"./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.d" \
"./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.d" \
"./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.d" 

OBJS += \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.o" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.o" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.o" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.o" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.o" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.o" \
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.src":"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.c" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.o":"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.src" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.src":"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.c" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.o":"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.src" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.src":"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.c" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.o":"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.src" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.src":"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.c" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.o":"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.src" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.src":"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.c" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.o":"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.src" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.src":"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.c" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.o":"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.src" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.src":"../Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.c" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.o":"Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.src" "Libraries/Infra/Ssw/TC3xx/Tricore/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-Infra-2f-Ssw-2f-TC3xx-2f-Tricore

clean-Libraries-2f-Infra-2f-Ssw-2f-TC3xx-2f-Tricore:
	-$(RM) ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.d ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.o ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Infra.src ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.d ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.o ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc0.src ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.d ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.o ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc1.src ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.d ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.o ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc2.src ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.d ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.o ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc3.src ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.d ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.o ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc4.src ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.d ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.o ./Libraries/Infra/Ssw/TC3xx/Tricore/Ifx_Ssw_Tc5.src

.PHONY: clean-Libraries-2f-Infra-2f-Ssw-2f-TC3xx-2f-Tricore


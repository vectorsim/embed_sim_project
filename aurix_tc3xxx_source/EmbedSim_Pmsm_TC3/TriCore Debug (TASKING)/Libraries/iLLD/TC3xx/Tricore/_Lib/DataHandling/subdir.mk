################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.c" \
"../Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.c" \
"../Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.c" 

COMPILED_SRCS += \
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.src" \
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.src" \
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.src" 

C_DEPS += \
"./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.d" \
"./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.d" \
"./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.d" 

OBJS += \
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.o" \
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.o" \
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.src":"../Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.c" "Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.o":"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.src" "Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.src":"../Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.c" "Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.o":"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.src" "Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.src":"../Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.c" "Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.o":"Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.src" "Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-_Lib-2f-DataHandling

clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-_Lib-2f-DataHandling:
	-$(RM) ./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.d ./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.o ./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.asm.src ./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.d ./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.o ./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_CircularBuffer.src ./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.d ./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.o ./Libraries/iLLD/TC3xx/Tricore/_Lib/DataHandling/Ifx_Fifo.src

.PHONY: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-_Lib-2f-DataHandling


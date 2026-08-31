################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.c" 

COMPILED_SRCS += \
"Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.src" 

C_DEPS += \
"./Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.d" 

OBJS += \
"Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.src":"../Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.c" "Libraries/iLLD/TC3xx/Tricore/Dma/Dma/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/EmbedSim_MotorControl/EmbedSim_Pmsm_TC3/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Wc-w508 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.o":"Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.src" "Libraries/iLLD/TC3xx/Tricore/Dma/Dma/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Dma-2f-Dma

clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Dma-2f-Dma:
	-$(RM) ./Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.d ./Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.o ./Libraries/iLLD/TC3xx/Tricore/Dma/Dma/IfxDma_Dma.src

.PHONY: clean-Libraries-2f-iLLD-2f-TC3xx-2f-Tricore-2f-Dma-2f-Dma


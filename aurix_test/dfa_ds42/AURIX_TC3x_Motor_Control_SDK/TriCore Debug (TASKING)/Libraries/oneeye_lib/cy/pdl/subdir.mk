################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.c" \
"../Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.c" 

COMPILED_SRCS += \
"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.src" \
"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.src" 

C_DEPS += \
"./Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.d" \
"./Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.d" 

OBJS += \
"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.o" \
"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.src":"../Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.c" "Libraries/oneeye_lib/cy/pdl/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.o":"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.src" "Libraries/oneeye_lib/cy/pdl/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.src":"../Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.c" "Libraries/oneeye_lib/cy/pdl/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.o":"Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.src" "Libraries/oneeye_lib/cy/pdl/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-oneeye_lib-2f-cy-2f-pdl

clean-Libraries-2f-oneeye_lib-2f-cy-2f-pdl:
	-$(RM) ./Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.d ./Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.o ./Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl.src ./Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.d ./Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.o ./Libraries/oneeye_lib/cy/pdl/ifx_oe_al_cy_pdl_sbcdmadpipe.src

.PHONY: clean-Libraries-2f-oneeye_lib-2f-cy-2f-pdl


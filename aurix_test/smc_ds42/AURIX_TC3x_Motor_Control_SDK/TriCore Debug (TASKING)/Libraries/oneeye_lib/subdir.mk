################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../Libraries/oneeye_lib/ifx_oe_circularbuffer.c" \
"../Libraries/oneeye_lib/ifx_oe_dasfifo.c" \
"../Libraries/oneeye_lib/ifx_oe_daspipe.c" \
"../Libraries/oneeye_lib/ifx_oe_fifo.c" \
"../Libraries/oneeye_lib/ifx_oe_fifodpipe.c" \
"../Libraries/oneeye_lib/ifx_oe_linkedlist.c" \
"../Libraries/oneeye_lib/ifx_oe_log.c" \
"../Libraries/oneeye_lib/ifx_oe_malloc.c" \
"../Libraries/oneeye_lib/ifx_oe_osci.c" \
"../Libraries/oneeye_lib/ifx_oe_oscibb.c" \
"../Libraries/oneeye_lib/ifx_oe_shell.c" \
"../Libraries/oneeye_lib/ifx_oe_shellbb.c" \
"../Libraries/oneeye_lib/ifx_oe_stdif_dpipe.c" \
"../Libraries/oneeye_lib/ifx_oe_syncprotocol.c" \
"../Libraries/oneeye_lib/ifx_oe_time.c" 

COMPILED_SRCS += \
"Libraries/oneeye_lib/ifx_oe_circularbuffer.src" \
"Libraries/oneeye_lib/ifx_oe_dasfifo.src" \
"Libraries/oneeye_lib/ifx_oe_daspipe.src" \
"Libraries/oneeye_lib/ifx_oe_fifo.src" \
"Libraries/oneeye_lib/ifx_oe_fifodpipe.src" \
"Libraries/oneeye_lib/ifx_oe_linkedlist.src" \
"Libraries/oneeye_lib/ifx_oe_log.src" \
"Libraries/oneeye_lib/ifx_oe_malloc.src" \
"Libraries/oneeye_lib/ifx_oe_osci.src" \
"Libraries/oneeye_lib/ifx_oe_oscibb.src" \
"Libraries/oneeye_lib/ifx_oe_shell.src" \
"Libraries/oneeye_lib/ifx_oe_shellbb.src" \
"Libraries/oneeye_lib/ifx_oe_stdif_dpipe.src" \
"Libraries/oneeye_lib/ifx_oe_syncprotocol.src" \
"Libraries/oneeye_lib/ifx_oe_time.src" 

C_DEPS += \
"./Libraries/oneeye_lib/ifx_oe_circularbuffer.d" \
"./Libraries/oneeye_lib/ifx_oe_dasfifo.d" \
"./Libraries/oneeye_lib/ifx_oe_daspipe.d" \
"./Libraries/oneeye_lib/ifx_oe_fifo.d" \
"./Libraries/oneeye_lib/ifx_oe_fifodpipe.d" \
"./Libraries/oneeye_lib/ifx_oe_linkedlist.d" \
"./Libraries/oneeye_lib/ifx_oe_log.d" \
"./Libraries/oneeye_lib/ifx_oe_malloc.d" \
"./Libraries/oneeye_lib/ifx_oe_osci.d" \
"./Libraries/oneeye_lib/ifx_oe_oscibb.d" \
"./Libraries/oneeye_lib/ifx_oe_shell.d" \
"./Libraries/oneeye_lib/ifx_oe_shellbb.d" \
"./Libraries/oneeye_lib/ifx_oe_stdif_dpipe.d" \
"./Libraries/oneeye_lib/ifx_oe_syncprotocol.d" \
"./Libraries/oneeye_lib/ifx_oe_time.d" 

OBJS += \
"Libraries/oneeye_lib/ifx_oe_circularbuffer.o" \
"Libraries/oneeye_lib/ifx_oe_dasfifo.o" \
"Libraries/oneeye_lib/ifx_oe_daspipe.o" \
"Libraries/oneeye_lib/ifx_oe_fifo.o" \
"Libraries/oneeye_lib/ifx_oe_fifodpipe.o" \
"Libraries/oneeye_lib/ifx_oe_linkedlist.o" \
"Libraries/oneeye_lib/ifx_oe_log.o" \
"Libraries/oneeye_lib/ifx_oe_malloc.o" \
"Libraries/oneeye_lib/ifx_oe_osci.o" \
"Libraries/oneeye_lib/ifx_oe_oscibb.o" \
"Libraries/oneeye_lib/ifx_oe_shell.o" \
"Libraries/oneeye_lib/ifx_oe_shellbb.o" \
"Libraries/oneeye_lib/ifx_oe_stdif_dpipe.o" \
"Libraries/oneeye_lib/ifx_oe_syncprotocol.o" \
"Libraries/oneeye_lib/ifx_oe_time.o" 


# Each subdirectory must supply rules for building sources it contributes
"Libraries/oneeye_lib/ifx_oe_circularbuffer.src":"../Libraries/oneeye_lib/ifx_oe_circularbuffer.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_circularbuffer.o":"Libraries/oneeye_lib/ifx_oe_circularbuffer.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_dasfifo.src":"../Libraries/oneeye_lib/ifx_oe_dasfifo.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_dasfifo.o":"Libraries/oneeye_lib/ifx_oe_dasfifo.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_daspipe.src":"../Libraries/oneeye_lib/ifx_oe_daspipe.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_daspipe.o":"Libraries/oneeye_lib/ifx_oe_daspipe.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_fifo.src":"../Libraries/oneeye_lib/ifx_oe_fifo.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_fifo.o":"Libraries/oneeye_lib/ifx_oe_fifo.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_fifodpipe.src":"../Libraries/oneeye_lib/ifx_oe_fifodpipe.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_fifodpipe.o":"Libraries/oneeye_lib/ifx_oe_fifodpipe.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_linkedlist.src":"../Libraries/oneeye_lib/ifx_oe_linkedlist.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_linkedlist.o":"Libraries/oneeye_lib/ifx_oe_linkedlist.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_log.src":"../Libraries/oneeye_lib/ifx_oe_log.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_log.o":"Libraries/oneeye_lib/ifx_oe_log.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_malloc.src":"../Libraries/oneeye_lib/ifx_oe_malloc.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_malloc.o":"Libraries/oneeye_lib/ifx_oe_malloc.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_osci.src":"../Libraries/oneeye_lib/ifx_oe_osci.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_osci.o":"Libraries/oneeye_lib/ifx_oe_osci.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_oscibb.src":"../Libraries/oneeye_lib/ifx_oe_oscibb.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_oscibb.o":"Libraries/oneeye_lib/ifx_oe_oscibb.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_shell.src":"../Libraries/oneeye_lib/ifx_oe_shell.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_shell.o":"Libraries/oneeye_lib/ifx_oe_shell.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_shellbb.src":"../Libraries/oneeye_lib/ifx_oe_shellbb.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_shellbb.o":"Libraries/oneeye_lib/ifx_oe_shellbb.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_stdif_dpipe.src":"../Libraries/oneeye_lib/ifx_oe_stdif_dpipe.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_stdif_dpipe.o":"Libraries/oneeye_lib/ifx_oe_stdif_dpipe.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_syncprotocol.src":"../Libraries/oneeye_lib/ifx_oe_syncprotocol.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_syncprotocol.o":"Libraries/oneeye_lib/ifx_oe_syncprotocol.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_time.src":"../Libraries/oneeye_lib/ifx_oe_time.c" "Libraries/oneeye_lib/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"Libraries/oneeye_lib/ifx_oe_time.o":"Libraries/oneeye_lib/ifx_oe_time.src" "Libraries/oneeye_lib/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-Libraries-2f-oneeye_lib

clean-Libraries-2f-oneeye_lib:
	-$(RM) ./Libraries/oneeye_lib/ifx_oe_circularbuffer.d ./Libraries/oneeye_lib/ifx_oe_circularbuffer.o ./Libraries/oneeye_lib/ifx_oe_circularbuffer.src ./Libraries/oneeye_lib/ifx_oe_dasfifo.d ./Libraries/oneeye_lib/ifx_oe_dasfifo.o ./Libraries/oneeye_lib/ifx_oe_dasfifo.src ./Libraries/oneeye_lib/ifx_oe_daspipe.d ./Libraries/oneeye_lib/ifx_oe_daspipe.o ./Libraries/oneeye_lib/ifx_oe_daspipe.src ./Libraries/oneeye_lib/ifx_oe_fifo.d ./Libraries/oneeye_lib/ifx_oe_fifo.o ./Libraries/oneeye_lib/ifx_oe_fifo.src ./Libraries/oneeye_lib/ifx_oe_fifodpipe.d ./Libraries/oneeye_lib/ifx_oe_fifodpipe.o ./Libraries/oneeye_lib/ifx_oe_fifodpipe.src ./Libraries/oneeye_lib/ifx_oe_linkedlist.d ./Libraries/oneeye_lib/ifx_oe_linkedlist.o ./Libraries/oneeye_lib/ifx_oe_linkedlist.src ./Libraries/oneeye_lib/ifx_oe_log.d ./Libraries/oneeye_lib/ifx_oe_log.o ./Libraries/oneeye_lib/ifx_oe_log.src ./Libraries/oneeye_lib/ifx_oe_malloc.d ./Libraries/oneeye_lib/ifx_oe_malloc.o ./Libraries/oneeye_lib/ifx_oe_malloc.src ./Libraries/oneeye_lib/ifx_oe_osci.d ./Libraries/oneeye_lib/ifx_oe_osci.o ./Libraries/oneeye_lib/ifx_oe_osci.src ./Libraries/oneeye_lib/ifx_oe_oscibb.d ./Libraries/oneeye_lib/ifx_oe_oscibb.o ./Libraries/oneeye_lib/ifx_oe_oscibb.src ./Libraries/oneeye_lib/ifx_oe_shell.d ./Libraries/oneeye_lib/ifx_oe_shell.o ./Libraries/oneeye_lib/ifx_oe_shell.src ./Libraries/oneeye_lib/ifx_oe_shellbb.d ./Libraries/oneeye_lib/ifx_oe_shellbb.o ./Libraries/oneeye_lib/ifx_oe_shellbb.src ./Libraries/oneeye_lib/ifx_oe_stdif_dpipe.d ./Libraries/oneeye_lib/ifx_oe_stdif_dpipe.o ./Libraries/oneeye_lib/ifx_oe_stdif_dpipe.src ./Libraries/oneeye_lib/ifx_oe_syncprotocol.d ./Libraries/oneeye_lib/ifx_oe_syncprotocol.o ./Libraries/oneeye_lib/ifx_oe_syncprotocol.src ./Libraries/oneeye_lib/ifx_oe_time.d ./Libraries/oneeye_lib/ifx_oe_time.o ./Libraries/oneeye_lib/ifx_oe_time.src

.PHONY: clean-Libraries-2f-oneeye_lib


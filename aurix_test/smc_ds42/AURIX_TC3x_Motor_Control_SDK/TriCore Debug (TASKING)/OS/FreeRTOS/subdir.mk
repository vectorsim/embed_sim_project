################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../OS/FreeRTOS/croutine.c" \
"../OS/FreeRTOS/event_groups.c" \
"../OS/FreeRTOS/list.c" \
"../OS/FreeRTOS/queue.c" \
"../OS/FreeRTOS/stream_buffer.c" \
"../OS/FreeRTOS/tasks.c" \
"../OS/FreeRTOS/timers.c" 

COMPILED_SRCS += \
"OS/FreeRTOS/croutine.src" \
"OS/FreeRTOS/event_groups.src" \
"OS/FreeRTOS/list.src" \
"OS/FreeRTOS/queue.src" \
"OS/FreeRTOS/stream_buffer.src" \
"OS/FreeRTOS/tasks.src" \
"OS/FreeRTOS/timers.src" 

C_DEPS += \
"./OS/FreeRTOS/croutine.d" \
"./OS/FreeRTOS/event_groups.d" \
"./OS/FreeRTOS/list.d" \
"./OS/FreeRTOS/queue.d" \
"./OS/FreeRTOS/stream_buffer.d" \
"./OS/FreeRTOS/tasks.d" \
"./OS/FreeRTOS/timers.d" 

OBJS += \
"OS/FreeRTOS/croutine.o" \
"OS/FreeRTOS/event_groups.o" \
"OS/FreeRTOS/list.o" \
"OS/FreeRTOS/queue.o" \
"OS/FreeRTOS/stream_buffer.o" \
"OS/FreeRTOS/tasks.o" \
"OS/FreeRTOS/timers.o" 


# Each subdirectory must supply rules for building sources it contributes
"OS/FreeRTOS/croutine.src":"../OS/FreeRTOS/croutine.c" "OS/FreeRTOS/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"OS/FreeRTOS/croutine.o":"OS/FreeRTOS/croutine.src" "OS/FreeRTOS/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"OS/FreeRTOS/event_groups.src":"../OS/FreeRTOS/event_groups.c" "OS/FreeRTOS/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"OS/FreeRTOS/event_groups.o":"OS/FreeRTOS/event_groups.src" "OS/FreeRTOS/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"OS/FreeRTOS/list.src":"../OS/FreeRTOS/list.c" "OS/FreeRTOS/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"OS/FreeRTOS/list.o":"OS/FreeRTOS/list.src" "OS/FreeRTOS/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"OS/FreeRTOS/queue.src":"../OS/FreeRTOS/queue.c" "OS/FreeRTOS/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"OS/FreeRTOS/queue.o":"OS/FreeRTOS/queue.src" "OS/FreeRTOS/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"OS/FreeRTOS/stream_buffer.src":"../OS/FreeRTOS/stream_buffer.c" "OS/FreeRTOS/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"OS/FreeRTOS/stream_buffer.o":"OS/FreeRTOS/stream_buffer.src" "OS/FreeRTOS/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"OS/FreeRTOS/tasks.src":"../OS/FreeRTOS/tasks.c" "OS/FreeRTOS/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"OS/FreeRTOS/tasks.o":"OS/FreeRTOS/tasks.src" "OS/FreeRTOS/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"OS/FreeRTOS/timers.src":"../OS/FreeRTOS/timers.c" "OS/FreeRTOS/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"OS/FreeRTOS/timers.o":"OS/FreeRTOS/timers.src" "OS/FreeRTOS/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-OS-2f-FreeRTOS

clean-OS-2f-FreeRTOS:
	-$(RM) ./OS/FreeRTOS/croutine.d ./OS/FreeRTOS/croutine.o ./OS/FreeRTOS/croutine.src ./OS/FreeRTOS/event_groups.d ./OS/FreeRTOS/event_groups.o ./OS/FreeRTOS/event_groups.src ./OS/FreeRTOS/list.d ./OS/FreeRTOS/list.o ./OS/FreeRTOS/list.src ./OS/FreeRTOS/queue.d ./OS/FreeRTOS/queue.o ./OS/FreeRTOS/queue.src ./OS/FreeRTOS/stream_buffer.d ./OS/FreeRTOS/stream_buffer.o ./OS/FreeRTOS/stream_buffer.src ./OS/FreeRTOS/tasks.d ./OS/FreeRTOS/tasks.o ./OS/FreeRTOS/tasks.src ./OS/FreeRTOS/timers.d ./OS/FreeRTOS/timers.o ./OS/FreeRTOS/timers.src

.PHONY: clean-OS-2f-FreeRTOS


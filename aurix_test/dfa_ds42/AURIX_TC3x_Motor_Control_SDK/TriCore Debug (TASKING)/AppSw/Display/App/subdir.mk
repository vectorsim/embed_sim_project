################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../AppSw/Display/App/Display_Functions.c" \
"../AppSw/Display/App/DrawLogo.c" \
"../AppSw/Display/App/basebar.c" \
"../AppSw/Display/App/conio_tft.c" \
"../AppSw/Display/App/fifo.c" \
"../AppSw/Display/App/font_8_12.c" \
"../AppSw/Display/App/keyboard.c" \
"../AppSw/Display/App/libtft_ascii.c" \
"../AppSw/Display/App/libtft_graphics.c" \
"../AppSw/Display/App/menue.c" \
"../AppSw/Display/App/switchoff.c" \
"../AppSw/Display/App/tfthw.c" \
"../AppSw/Display/App/touch.c" 

COMPILED_SRCS += \
"AppSw/Display/App/Display_Functions.src" \
"AppSw/Display/App/DrawLogo.src" \
"AppSw/Display/App/basebar.src" \
"AppSw/Display/App/conio_tft.src" \
"AppSw/Display/App/fifo.src" \
"AppSw/Display/App/font_8_12.src" \
"AppSw/Display/App/keyboard.src" \
"AppSw/Display/App/libtft_ascii.src" \
"AppSw/Display/App/libtft_graphics.src" \
"AppSw/Display/App/menue.src" \
"AppSw/Display/App/switchoff.src" \
"AppSw/Display/App/tfthw.src" \
"AppSw/Display/App/touch.src" 

C_DEPS += \
"./AppSw/Display/App/Display_Functions.d" \
"./AppSw/Display/App/DrawLogo.d" \
"./AppSw/Display/App/basebar.d" \
"./AppSw/Display/App/conio_tft.d" \
"./AppSw/Display/App/fifo.d" \
"./AppSw/Display/App/font_8_12.d" \
"./AppSw/Display/App/keyboard.d" \
"./AppSw/Display/App/libtft_ascii.d" \
"./AppSw/Display/App/libtft_graphics.d" \
"./AppSw/Display/App/menue.d" \
"./AppSw/Display/App/switchoff.d" \
"./AppSw/Display/App/tfthw.d" \
"./AppSw/Display/App/touch.d" 

OBJS += \
"AppSw/Display/App/Display_Functions.o" \
"AppSw/Display/App/DrawLogo.o" \
"AppSw/Display/App/basebar.o" \
"AppSw/Display/App/conio_tft.o" \
"AppSw/Display/App/fifo.o" \
"AppSw/Display/App/font_8_12.o" \
"AppSw/Display/App/keyboard.o" \
"AppSw/Display/App/libtft_ascii.o" \
"AppSw/Display/App/libtft_graphics.o" \
"AppSw/Display/App/menue.o" \
"AppSw/Display/App/switchoff.o" \
"AppSw/Display/App/tfthw.o" \
"AppSw/Display/App/touch.o" 


# Each subdirectory must supply rules for building sources it contributes
"AppSw/Display/App/Display_Functions.src":"../AppSw/Display/App/Display_Functions.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/Display_Functions.o":"AppSw/Display/App/Display_Functions.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/DrawLogo.src":"../AppSw/Display/App/DrawLogo.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/DrawLogo.o":"AppSw/Display/App/DrawLogo.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/basebar.src":"../AppSw/Display/App/basebar.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/basebar.o":"AppSw/Display/App/basebar.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/conio_tft.src":"../AppSw/Display/App/conio_tft.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/conio_tft.o":"AppSw/Display/App/conio_tft.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/fifo.src":"../AppSw/Display/App/fifo.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/fifo.o":"AppSw/Display/App/fifo.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/font_8_12.src":"../AppSw/Display/App/font_8_12.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/font_8_12.o":"AppSw/Display/App/font_8_12.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/keyboard.src":"../AppSw/Display/App/keyboard.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/keyboard.o":"AppSw/Display/App/keyboard.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/libtft_ascii.src":"../AppSw/Display/App/libtft_ascii.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/libtft_ascii.o":"AppSw/Display/App/libtft_ascii.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/libtft_graphics.src":"../AppSw/Display/App/libtft_graphics.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/libtft_graphics.o":"AppSw/Display/App/libtft_graphics.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/menue.src":"../AppSw/Display/App/menue.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/menue.o":"AppSw/Display/App/menue.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/switchoff.src":"../AppSw/Display/App/switchoff.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/switchoff.o":"AppSw/Display/App/switchoff.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/tfthw.src":"../AppSw/Display/App/tfthw.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/tfthw.o":"AppSw/Display/App/tfthw.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/Display/App/touch.src":"../AppSw/Display/App/touch.c" "AppSw/Display/App/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/Display/App/touch.o":"AppSw/Display/App/touch.src" "AppSw/Display/App/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-AppSw-2f-Display-2f-App

clean-AppSw-2f-Display-2f-App:
	-$(RM) ./AppSw/Display/App/Display_Functions.d ./AppSw/Display/App/Display_Functions.o ./AppSw/Display/App/Display_Functions.src ./AppSw/Display/App/DrawLogo.d ./AppSw/Display/App/DrawLogo.o ./AppSw/Display/App/DrawLogo.src ./AppSw/Display/App/basebar.d ./AppSw/Display/App/basebar.o ./AppSw/Display/App/basebar.src ./AppSw/Display/App/conio_tft.d ./AppSw/Display/App/conio_tft.o ./AppSw/Display/App/conio_tft.src ./AppSw/Display/App/fifo.d ./AppSw/Display/App/fifo.o ./AppSw/Display/App/fifo.src ./AppSw/Display/App/font_8_12.d ./AppSw/Display/App/font_8_12.o ./AppSw/Display/App/font_8_12.src ./AppSw/Display/App/keyboard.d ./AppSw/Display/App/keyboard.o ./AppSw/Display/App/keyboard.src ./AppSw/Display/App/libtft_ascii.d ./AppSw/Display/App/libtft_ascii.o ./AppSw/Display/App/libtft_ascii.src ./AppSw/Display/App/libtft_graphics.d ./AppSw/Display/App/libtft_graphics.o ./AppSw/Display/App/libtft_graphics.src ./AppSw/Display/App/menue.d ./AppSw/Display/App/menue.o ./AppSw/Display/App/menue.src ./AppSw/Display/App/switchoff.d ./AppSw/Display/App/switchoff.o ./AppSw/Display/App/switchoff.src ./AppSw/Display/App/tfthw.d ./AppSw/Display/App/tfthw.o ./AppSw/Display/App/tfthw.src ./AppSw/Display/App/touch.d ./AppSw/Display/App/touch.o ./AppSw/Display/App/touch.src

.PHONY: clean-AppSw-2f-Display-2f-App


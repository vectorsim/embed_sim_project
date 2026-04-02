################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
"../AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.c" \
"../AppSw/PmsmFoc/Libraries/MotorDrive/Park.c" \
"../AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.c" \
"../AppSw/PmsmFoc/Libraries/MotorDrive/Tables.c" \
"../AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.c" 

COMPILED_SRCS += \
"AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.src" \
"AppSw/PmsmFoc/Libraries/MotorDrive/Park.src" \
"AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.src" \
"AppSw/PmsmFoc/Libraries/MotorDrive/Tables.src" \
"AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.src" 

C_DEPS += \
"./AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.d" \
"./AppSw/PmsmFoc/Libraries/MotorDrive/Park.d" \
"./AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.d" \
"./AppSw/PmsmFoc/Libraries/MotorDrive/Tables.d" \
"./AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.d" 

OBJS += \
"AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.o" \
"AppSw/PmsmFoc/Libraries/MotorDrive/Park.o" \
"AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.o" \
"AppSw/PmsmFoc/Libraries/MotorDrive/Tables.o" \
"AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.o" 


# Each subdirectory must supply rules for building sources it contributes
"AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.src":"../AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.c" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.o":"AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.src" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/Libraries/MotorDrive/Park.src":"../AppSw/PmsmFoc/Libraries/MotorDrive/Park.c" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/Libraries/MotorDrive/Park.o":"AppSw/PmsmFoc/Libraries/MotorDrive/Park.src" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.src":"../AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.c" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.o":"AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.src" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/Libraries/MotorDrive/Tables.src":"../AppSw/PmsmFoc/Libraries/MotorDrive/Tables.c" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/Libraries/MotorDrive/Tables.o":"AppSw/PmsmFoc/Libraries/MotorDrive/Tables.src" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"
"AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.src":"../AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.c" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	cctc -cs --dep-file="$*.d" --misrac-version=2012 -D__CPU__=tc38x "-fC:/Aurix_EmbedSim/AURIX_TC3x_Motor_Control_SDK/TriCore Debug (TASKING)/TASKING_C_C___Compiler-Include_paths__-I_.opt" --iso=99 --c++14 --language=+volatile --exceptions --anachronisms --fp-model=3 -O0 --tradeoff=4 --compact-max-size=200 -g -Wc-w544 -Wc-w557 -Ctc38x -Y0 -N0 -Z0 -o "$@" "$<"
"AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.o":"AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.src" "AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk"
	astc -Og -Os --no-warnings= --error-limit=42 -o  "$@" "$<"

clean: clean-AppSw-2f-PmsmFoc-2f-Libraries-2f-MotorDrive

clean-AppSw-2f-PmsmFoc-2f-Libraries-2f-MotorDrive:
	-$(RM) ./AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.d ./AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.o ./AppSw/PmsmFoc/Libraries/MotorDrive/Clarke.src ./AppSw/PmsmFoc/Libraries/MotorDrive/Park.d ./AppSw/PmsmFoc/Libraries/MotorDrive/Park.o ./AppSw/PmsmFoc/Libraries/MotorDrive/Park.src ./AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.d ./AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.o ./AppSw/PmsmFoc/Libraries/MotorDrive/SpaceVectorModulation.src ./AppSw/PmsmFoc/Libraries/MotorDrive/Tables.d ./AppSw/PmsmFoc/Libraries/MotorDrive/Tables.o ./AppSw/PmsmFoc/Libraries/MotorDrive/Tables.src ./AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.d ./AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.o ./AppSw/PmsmFoc/Libraries/MotorDrive/Tables_const.src

.PHONY: clean-AppSw-2f-PmsmFoc-2f-Libraries-2f-MotorDrive


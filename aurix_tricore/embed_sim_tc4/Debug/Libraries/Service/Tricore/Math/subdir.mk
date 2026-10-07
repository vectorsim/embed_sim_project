################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/Service/Tricore/Math/Ifx_AngleTrkF32.c \
../Libraries/Service/Tricore/Math/Ifx_Cf32.c \
../Libraries/Service/Tricore/Math/Ifx_Crc.c \
../Libraries/Service/Tricore/Math/Ifx_FftF32.c \
../Libraries/Service/Tricore/Math/Ifx_FftF32_BitReverseTable.c \
../Libraries/Service/Tricore/Math/Ifx_FftF32_TwiddleTable.c \
../Libraries/Service/Tricore/Math/Ifx_IntegralF32.c \
../Libraries/Service/Tricore/Math/Ifx_LowPassPt1F32.c \
../Libraries/Service/Tricore/Math/Ifx_LutAtan2F32.c \
../Libraries/Service/Tricore/Math/Ifx_LutAtan2F32_Table.c \
../Libraries/Service/Tricore/Math/Ifx_LutLSincosF32.c \
../Libraries/Service/Tricore/Math/Ifx_LutLinearF32.c \
../Libraries/Service/Tricore/Math/Ifx_LutSincosF32.c \
../Libraries/Service/Tricore/Math/Ifx_LutSincosF32_Table.c \
../Libraries/Service/Tricore/Math/Ifx_RampF32.c \
../Libraries/Service/Tricore/Math/Ifx_WndF32_BlackmanHarrisTable.c \
../Libraries/Service/Tricore/Math/Ifx_WndF32_HannTable.c 

C_DEPS += \
./Libraries/Service/Tricore/Math/Ifx_AngleTrkF32.d \
./Libraries/Service/Tricore/Math/Ifx_Cf32.d \
./Libraries/Service/Tricore/Math/Ifx_Crc.d \
./Libraries/Service/Tricore/Math/Ifx_FftF32.d \
./Libraries/Service/Tricore/Math/Ifx_FftF32_BitReverseTable.d \
./Libraries/Service/Tricore/Math/Ifx_FftF32_TwiddleTable.d \
./Libraries/Service/Tricore/Math/Ifx_IntegralF32.d \
./Libraries/Service/Tricore/Math/Ifx_LowPassPt1F32.d \
./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32.d \
./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32_Table.d \
./Libraries/Service/Tricore/Math/Ifx_LutLSincosF32.d \
./Libraries/Service/Tricore/Math/Ifx_LutLinearF32.d \
./Libraries/Service/Tricore/Math/Ifx_LutSincosF32.d \
./Libraries/Service/Tricore/Math/Ifx_LutSincosF32_Table.d \
./Libraries/Service/Tricore/Math/Ifx_RampF32.d \
./Libraries/Service/Tricore/Math/Ifx_WndF32_BlackmanHarrisTable.d \
./Libraries/Service/Tricore/Math/Ifx_WndF32_HannTable.d 

OBJS += \
./Libraries/Service/Tricore/Math/Ifx_AngleTrkF32.o \
./Libraries/Service/Tricore/Math/Ifx_Cf32.o \
./Libraries/Service/Tricore/Math/Ifx_Crc.o \
./Libraries/Service/Tricore/Math/Ifx_FftF32.o \
./Libraries/Service/Tricore/Math/Ifx_FftF32_BitReverseTable.o \
./Libraries/Service/Tricore/Math/Ifx_FftF32_TwiddleTable.o \
./Libraries/Service/Tricore/Math/Ifx_IntegralF32.o \
./Libraries/Service/Tricore/Math/Ifx_LowPassPt1F32.o \
./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32.o \
./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32_Table.o \
./Libraries/Service/Tricore/Math/Ifx_LutLSincosF32.o \
./Libraries/Service/Tricore/Math/Ifx_LutLinearF32.o \
./Libraries/Service/Tricore/Math/Ifx_LutSincosF32.o \
./Libraries/Service/Tricore/Math/Ifx_LutSincosF32_Table.o \
./Libraries/Service/Tricore/Math/Ifx_RampF32.o \
./Libraries/Service/Tricore/Math/Ifx_WndF32_BlackmanHarrisTable.o \
./Libraries/Service/Tricore/Math/Ifx_WndF32_HannTable.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/Service/Tricore/Math/%.o: ../Libraries/Service/Tricore/Math/%.c Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_AngleTrkF32: ./Libraries/Service/Tricore/Math/Ifx_AngleTrkF32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_Cf32: ./Libraries/Service/Tricore/Math/Ifx_Cf32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_Crc: ./Libraries/Service/Tricore/Math/Ifx_Crc.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_FftF32: ./Libraries/Service/Tricore/Math/Ifx_FftF32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_FftF32_BitReverseTable: ./Libraries/Service/Tricore/Math/Ifx_FftF32_BitReverseTable.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_FftF32_TwiddleTable: ./Libraries/Service/Tricore/Math/Ifx_FftF32_TwiddleTable.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_IntegralF32: ./Libraries/Service/Tricore/Math/Ifx_IntegralF32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_LowPassPt1F32: ./Libraries/Service/Tricore/Math/Ifx_LowPassPt1F32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_LutAtan2F32: ./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_LutAtan2F32_Table: ./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32_Table.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_LutLSincosF32: ./Libraries/Service/Tricore/Math/Ifx_LutLSincosF32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_LutLinearF32: ./Libraries/Service/Tricore/Math/Ifx_LutLinearF32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_LutSincosF32: ./Libraries/Service/Tricore/Math/Ifx_LutSincosF32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_LutSincosF32_Table: ./Libraries/Service/Tricore/Math/Ifx_LutSincosF32_Table.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_RampF32: ./Libraries/Service/Tricore/Math/Ifx_RampF32.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_WndF32_BlackmanHarrisTable: ./Libraries/Service/Tricore/Math/Ifx_WndF32_BlackmanHarrisTable.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/Service/Tricore/Math/Ifx_WndF32_HannTable: ./Libraries/Service/Tricore/Math/Ifx_WndF32_HannTable.o $(USER_OBJS) $(OBJS) Libraries/Service/Tricore/Math/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-Service-2f-Tricore-2f-Math

clean-Libraries-2f-Service-2f-Tricore-2f-Math:
	-$(RM) ./Libraries/Service/Tricore/Math/Ifx_AngleTrkF32 ./Libraries/Service/Tricore/Math/Ifx_AngleTrkF32.d ./Libraries/Service/Tricore/Math/Ifx_AngleTrkF32.o ./Libraries/Service/Tricore/Math/Ifx_Cf32 ./Libraries/Service/Tricore/Math/Ifx_Cf32.d ./Libraries/Service/Tricore/Math/Ifx_Cf32.o ./Libraries/Service/Tricore/Math/Ifx_Crc ./Libraries/Service/Tricore/Math/Ifx_Crc.d ./Libraries/Service/Tricore/Math/Ifx_Crc.o ./Libraries/Service/Tricore/Math/Ifx_FftF32 ./Libraries/Service/Tricore/Math/Ifx_FftF32.d ./Libraries/Service/Tricore/Math/Ifx_FftF32.o ./Libraries/Service/Tricore/Math/Ifx_FftF32_BitReverseTable ./Libraries/Service/Tricore/Math/Ifx_FftF32_BitReverseTable.d ./Libraries/Service/Tricore/Math/Ifx_FftF32_BitReverseTable.o ./Libraries/Service/Tricore/Math/Ifx_FftF32_TwiddleTable ./Libraries/Service/Tricore/Math/Ifx_FftF32_TwiddleTable.d ./Libraries/Service/Tricore/Math/Ifx_FftF32_TwiddleTable.o ./Libraries/Service/Tricore/Math/Ifx_IntegralF32 ./Libraries/Service/Tricore/Math/Ifx_IntegralF32.d ./Libraries/Service/Tricore/Math/Ifx_IntegralF32.o ./Libraries/Service/Tricore/Math/Ifx_LowPassPt1F32 ./Libraries/Service/Tricore/Math/Ifx_LowPassPt1F32.d ./Libraries/Service/Tricore/Math/Ifx_LowPassPt1F32.o ./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32 ./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32.d ./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32.o ./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32_Table ./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32_Table.d ./Libraries/Service/Tricore/Math/Ifx_LutAtan2F32_Table.o ./Libraries/Service/Tricore/Math/Ifx_LutLSincosF32 ./Libraries/Service/Tricore/Math/Ifx_LutLSincosF32.d ./Libraries/Service/Tricore/Math/Ifx_LutLSincosF32.o ./Libraries/Service/Tricore/Math/Ifx_LutLinearF32 ./Libraries/Service/Tricore/Math/Ifx_LutLinearF32.d ./Libraries/Service/Tricore/Math/Ifx_LutLinearF32.o ./Libraries/Service/Tricore/Math/Ifx_LutSincosF32 ./Libraries/Service/Tricore/Math/Ifx_LutSincosF32.d ./Libraries/Service/Tricore/Math/Ifx_LutSincosF32.o ./Libraries/Service/Tricore/Math/Ifx_LutSincosF32_Table ./Libraries/Service/Tricore/Math/Ifx_LutSincosF32_Table.d ./Libraries/Service/Tricore/Math/Ifx_LutSincosF32_Table.o ./Libraries/Service/Tricore/Math/Ifx_RampF32 ./Libraries/Service/Tricore/Math/Ifx_RampF32.d ./Libraries/Service/Tricore/Math/Ifx_RampF32.o ./Libraries/Service/Tricore/Math/Ifx_WndF32_BlackmanHarrisTable ./Libraries/Service/Tricore/Math/Ifx_WndF32_BlackmanHarrisTable.d ./Libraries/Service/Tricore/Math/Ifx_WndF32_BlackmanHarrisTable.o ./Libraries/Service/Tricore/Math/Ifx_WndF32_HannTable ./Libraries/Service/Tricore/Math/Ifx_WndF32_HannTable.d ./Libraries/Service/Tricore/Math/Ifx_WndF32_HannTable.o

.PHONY: clean-Libraries-2f-Service-2f-Tricore-2f-Math


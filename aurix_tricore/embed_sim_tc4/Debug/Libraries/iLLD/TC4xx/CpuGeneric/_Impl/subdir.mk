################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAdcCdspFw_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAp_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAsclin_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxDma_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxEgtm_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGeth_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGpt12_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGtm_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxHsphy_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxLeth_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxPcie_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRdma_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRif_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxSpu_cfg.c \
../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxXspi_cfg.c 

C_DEPS += \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAdcCdspFw_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAp_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAsclin_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxDma_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxEgtm_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGeth_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGpt12_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGtm_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxHsphy_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxLeth_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxPcie_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRdma_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRif_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxSpu_cfg.d \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxXspi_cfg.d 

OBJS += \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAdcCdspFw_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAp_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAsclin_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxDma_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxEgtm_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGeth_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGpt12_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGtm_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxHsphy_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxLeth_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxPcie_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRdma_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRif_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxSpu_cfg.o \
./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxXspi_cfg.o 


# Each subdirectory must supply rules for building sources it contributes
Libraries/iLLD/TC4xx/CpuGeneric/_Impl/%.o: ../Libraries/iLLD/TC4xx/CpuGeneric/_Impl/%.c Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Compiler'
	"C:\Infineon\AURIX-Configuration-Studio-1.0.24\eclipse\tools/ccache/ccache.exe" tricore-elf-gcc.exe -MMD -MT "$@" "@C:/Users/CSO212/ACS-v1.0.24-workspace/EmbedSim_TC4/EmbedSim_TC4_TC4/Debug/TriCore_GCC_C_Compiler-Include_paths__-I_.opt" -Og -fdata-sections -ffunction-sections -fstrict-volatile-bitfields -g -gdwarf-2 -Wall -std=gnu99 -Wa,-adhlns="$@.lst" -pipe -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -mtc18 -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAdcCdspFw_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAdcCdspFw_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAp_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAp_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAsclin_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAsclin_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxDma_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxDma_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxEgtm_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxEgtm_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGeth_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGeth_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGpt12_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGpt12_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGtm_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGtm_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxHsphy_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxHsphy_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxLeth_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxLeth_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxPcie_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxPcie_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRdma_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRdma_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRif_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRif_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxSpu_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxSpu_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '

Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxXspi_cfg: ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxXspi_cfg.o $(USER_OBJS) $(OBJS) Libraries/iLLD/TC4xx/CpuGeneric/_Impl/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: TriCore GCC C Linker'
	tricore-elf-gcc -T"../Lcf_Gcc_Tricore_Tc.lsl" -nocrt0 -Xlinker --gc-sections -Xlinker --no-warn-rwx-segments -Wl,-Map,"EmbedSim_TC4_TC4.map" -mtc18 -o "$@" "$<" $(USER_OBJS) $(LIBS) $(OBJS)
	@echo 'Finished building: $<'
	@echo ' '


clean: clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-_Impl

clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-_Impl:
	-$(RM) ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAdcCdspFw_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAdcCdspFw_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAdcCdspFw_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAp_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAp_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAp_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAsclin_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAsclin_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxAsclin_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxDma_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxDma_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxDma_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxEgtm_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxEgtm_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxEgtm_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGeth_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGeth_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGeth_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGpt12_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGpt12_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGpt12_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGtm_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGtm_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxGtm_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxHsphy_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxHsphy_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxHsphy_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxLeth_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxLeth_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxLeth_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxPcie_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxPcie_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxPcie_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRdma_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRdma_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRdma_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRif_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRif_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxRif_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxSpu_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxSpu_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxSpu_cfg.o ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxXspi_cfg ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxXspi_cfg.d ./Libraries/iLLD/TC4xx/CpuGeneric/_Impl/IfxXspi_cfg.o

.PHONY: clean-Libraries-2f-iLLD-2f-TC4xx-2f-CpuGeneric-2f-_Impl


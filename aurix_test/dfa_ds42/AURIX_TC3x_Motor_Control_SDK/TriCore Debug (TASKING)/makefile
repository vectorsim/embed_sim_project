################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include OS/FreeRTOS/portable/TriCore/subdir.mk
-include OS/FreeRTOS/portable/MemMang/subdir.mk
-include OS/FreeRTOS/subdir.mk
-include Libraries/oneeye_lib/xmc/xmclib/subdir.mk
-include Libraries/oneeye_lib/xmc/xmcdaveapp/subdir.mk
-include Libraries/oneeye_lib/cy/pdl/subdir.mk
-include Libraries/oneeye_lib/aurix/illd/subdir.mk
-include Libraries/oneeye_lib/arduino/subdir.mk
-include Libraries/oneeye_lib/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/_PinMap/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/_Lib/DataHandling/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/_Impl/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Stm/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Src/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Scu/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Qspi/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Qspi/SpiMaster/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Port/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Pms/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Mtu/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Gtm/Trig/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Gtm/Tom/Timer/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Gtm/Tom/PwmHl/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Gtm/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Gpt12/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Gpt12/IncrEnc/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Evadc/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Evadc/Adc/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Edsadc/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Edsadc/Edsadc/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Dma/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Dma/Dma/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Cpu/Trap/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Cpu/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Cpu/Irq/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Asclin/Std/subdir.mk
-include Libraries/iLLD/TC38A/Tricore/Asclin/Asc/subdir.mk
-include Libraries/Service/CpuGeneric/SysSe/Math/subdir.mk
-include Libraries/Service/CpuGeneric/SysSe/Comm/subdir.mk
-include Libraries/Service/CpuGeneric/SysSe/Bsp/subdir.mk
-include Libraries/Service/CpuGeneric/StdIf/subdir.mk
-include Libraries/Service/CpuGeneric/If/subdir.mk
-include Libraries/Infra/Ssw/TC38A/Tricore/subdir.mk
-include Libraries/Infra/Platform/Tricore/Compilers/subdir.mk
-include Libraries/ExtDevDrivers/TLF35584/Tricore/subdir.mk
-include Libraries/ExtDevDrivers/TLE9180/Tricore/subdir.mk
-include Configurations/Debug/subdir.mk
-include Configurations/subdir.mk
-include AppSw/PmsmFoc/MidSys/subdir.mk
-include AppSw/PmsmFoc/MCUInit/subdir.mk
-include AppSw/PmsmFoc/Libraries/MotorDrive/subdir.mk
-include AppSw/PmsmFoc/Libraries/Math/subdir.mk
-include AppSw/PmsmFoc/Interrupts/subdir.mk
-include AppSw/PmsmFoc/ExtDevInit/subdir.mk
-include AppSw/PmsmFoc/EmbedSim/subdir.mk
-include AppSw/PmsmFoc/ControlModules/subdir.mk
-include AppSw/OneEye/subdir.mk
-include AppSw/OSTasks/subdir.mk
-include AppSw/Display/MCUInit/subdir.mk
-include AppSw/Display/Interrupts/subdir.mk
-include AppSw/Display/App/subdir.mk
-include subdir.mk
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++M_DEPS)),)
-include $(C++M_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(CCM_DEPS)),)
-include $(CCM_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXXM_DEPS)),)
-include $(CXXM_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := AURIX_TC3x_Motor_Control_SDK
BUILD_ARTIFACT_EXTENSION := elf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 
SECONDARY_SIZE += \
AURIX_TC3x_Motor_Control_SDK.siz \


# All Target
all: main-build

# Main-build Target
main-build: AURIX_TC3x_Motor_Control_SDK.elf secondary-outputs

# Tool invocations
AURIX_TC3x_Motor_Control_SDK.elf: $(OBJS) makefile $(OPTIONAL_TOOL_DEPS) .AURIX_TC3x_Motor_Control_SDK.elf.opt
	@echo 'Building target: $@'
	@echo 'Invoking: TASKING Linker'
	cctc -f .AURIX_TC3x_Motor_Control_SDK.elf.opt
	@echo 'Finished building target: $@'
	@echo ' '

.AURIX_TC3x_Motor_Control_SDK.elf.opt:
	@argfile .AURIX_TC3x_Motor_Control_SDK.elf.opt --lsl-file="../Lcf_Tasking_Tricore_Tc.lsl" -Wl-Oc -Wl-OL -Wl-Ot -Wl-Ox -Wl-Oy -Wl--map-file="AURIX_TC3x_Motor_Control_SDK.map" -Wl-mc -Wl-mf -Wl-mi -Wl-mk -Wl-ml -Wl-mm -Wl-md -Wl-mr -Wl-mu --no-warnings= -Wl--error-limit=42 --fp-model=3 -lrt --lsl-core=vtc --exceptions --strict --anachronisms --force-c++ -Ctc38x -o"AURIX_TC3x_Motor_Control_SDK.elf" -Wl-o"AURIX_TC3x_Motor_Control_SDK.hex:IHEX" $(OBJS)

.PHONY: .AURIX_TC3x_Motor_Control_SDK.elf.opt

AURIX_TC3x_Motor_Control_SDK.siz: AURIX_TC3x_Motor_Control_SDK.elf makefile $(OPTIONAL_TOOL_DEPS)
	@echo 'Invoking: Print Size'
	elfsize  "AURIX_TC3x_Motor_Control_SDK.elf"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) AURIX_TC3x_Motor_Control_SDK.siz AURIX_TC3x_Motor_Control_SDK.elf
	-@echo ' '

secondary-outputs: $(SECONDARY_SIZE)

.PHONY: all clean dependents main-build

-include ../makefile.targets

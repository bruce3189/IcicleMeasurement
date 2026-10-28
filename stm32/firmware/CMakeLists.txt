cmake_minimum_required(VERSION 3.24)
project(ice_control LANGUAGES C ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(STM32CUBE_F1_ROOT "D:/STM32Tools/Repository/STM32Cube_FW_F1_V1.8.7"
    CACHE PATH "Existing official STM32CubeF1 package")
set(HAL "${STM32CUBE_F1_ROOT}/Drivers/STM32F1xx_HAL_Driver")
set(DEVICE "${STM32CUBE_F1_ROOT}/Drivers/CMSIS/Device/ST/STM32F1xx")
if(NOT EXISTS "${HAL}/Src/stm32f1xx_hal.c")
    message(FATAL_ERROR "Set STM32CUBE_F1_ROOT to the installed STM32CubeF1 SDK")
endif()

add_library(stm32_hal STATIC
    "${HAL}/Src/stm32f1xx_hal.c"
    "${DEVICE}/Source/Templates/system_stm32f1xx.c")
foreach(module cortex dma flash flash_ex gpio i2c pwr rcc rcc_ex tim tim_ex uart)
    target_sources(stm32_hal PRIVATE "${HAL}/Src/stm32f1xx_hal_${module}.c")
endforeach()
target_include_directories(stm32_hal PUBLIC config "${HAL}/Inc"
    "${STM32CUBE_F1_ROOT}/Drivers/CMSIS/Include" "${DEVICE}/Include")
target_compile_definitions(stm32_hal PUBLIC USE_HAL_DRIVER STM32F103xB)

add_executable(ice_control Core/main.c Core/stm32f1xx_it.c Core/runtime.c
    Board/board.c App/app_control.c Modules/buttons.c Modules/communication.c
    Modules/protocol.c Modules/display.c Modules/oled_port.c Modules/status_led.c
    Modules/lighting.c "${DEVICE}/Source/Templates/gcc/startup_stm32f103xb.s")
set_target_properties(ice_control PROPERTIES SUFFIX ".elf")
target_include_directories(ice_control PRIVATE App Board Modules)
target_link_libraries(ice_control PRIVATE stm32_hal)
foreach(target stm32_hal ice_control)
    target_compile_options(${target} PRIVATE -mcpu=cortex-m3 -mthumb -mfloat-abi=soft
        -ffunction-sections -fdata-sections -fno-common -Os -g3 -Wall -Wextra)
endforeach()
target_compile_options(ice_control PRIVATE $<$<COMPILE_LANGUAGE:C>:-Werror>)
set(LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/linker/STM32F103C8Tx.ld")
set_property(TARGET ice_control APPEND PROPERTY LINK_DEPENDS "${LINKER_SCRIPT}")
target_link_options(ice_control PRIVATE -mcpu=cortex-m3 -mthumb -mfloat-abi=soft
    -nostartfiles --specs=nano.specs "-T${LINKER_SCRIPT}" -Wl,--gc-sections
    "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/ice_control.map" -Wl,--print-memory-usage)
add_custom_command(TARGET ice_control POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:ice_control> "${CMAKE_CURRENT_BINARY_DIR}/ice_control.hex"
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:ice_control>)

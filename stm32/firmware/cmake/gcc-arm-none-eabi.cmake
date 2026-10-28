set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(STM32_GCC_ROOT "D:/STM32Tools/Bundles/gnu-tools-for-stm32/14.3.1+st.2"
    CACHE PATH "Existing GNU Tools for STM32")
set(CMAKE_C_COMPILER "${STM32_GCC_ROOT}/bin/arm-none-eabi-gcc.exe")
set(CMAKE_ASM_COMPILER "${STM32_GCC_ROOT}/bin/arm-none-eabi-gcc.exe")
set(CMAKE_OBJCOPY "${STM32_GCC_ROOT}/bin/arm-none-eabi-objcopy.exe")
set(CMAKE_SIZE "${STM32_GCC_ROOT}/bin/arm-none-eabi-size.exe")

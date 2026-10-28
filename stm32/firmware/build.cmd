@echo off
setlocal
set "PATH=D:\STM32Tools\Bundles\gnu-tools-for-stm32\14.3.1+st.2\bin;D:\STM32Tools\Bundles\cmake\4.3.1+st.1\bin;D:\STM32Tools\Bundles\ninja\1.13.2+st.1\bin;%PATH%"
set "CMAKE=D:\STM32Tools\Bundles\cmake\4.3.1+st.1\bin\cmake.exe"
set "NINJA=D:\STM32Tools\Bundles\ninja\1.13.2+st.1\bin\ninja.exe"
if not exist "%CMAKE%" (
  echo Missing CMake: %CMAKE%
  exit /b 1
)
if exist "%~dp0build\CMakeCache.txt" (
  "%CMAKE%" -S "%~dp0." -B "%~dp0build" %*
) else (
  "%CMAKE%" -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_TOOLCHAIN_FILE="%~dp0cmake\gcc-arm-none-eabi.cmake" -DCMAKE_MAKE_PROGRAM="%NINJA%" %*
)
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --build "%~dp0build" --parallel 4
exit /b %errorlevel%

@echo off
REM HyperBlocks CUDA Build Script (Batch Version)
REM This script automates the build process for the HyperBlocks project

setlocal enabledelayedexpansion

REM Check if we're in the correct directory
if not exist "CMakeLists.txt" (
    echo Error: CMakeLists.txt not found. Please run this script from the project root directory.
    exit /b 1
)

echo [%time%] Starting HyperBlocks CUDA build process...

REM Check for CMake
set "cmake_path="
where cmake >nul 2>&1
if %errorlevel% equ 0 (
    set "cmake_path=cmake"
) else if exist "C:\Program Files\CMake\bin\cmake.exe" (
    set "cmake_path=C:\Program Files\CMake\bin\cmake.exe"
) else (
    echo Error: CMake not found. Please install CMake and ensure it's in your PATH.
    exit /b 1
)

REM Check for nvcc (CUDA compiler)
where nvcc >nul 2>&1
if not %errorlevel% equ 0 (
    echo Error: nvcc (CUDA compiler) not found. Please install CUDA Toolkit.
    exit /b 1
)

echo [%time%] Required tools found successfully

REM Create build directory if it doesn't exist
if not exist "build" (
    echo [%time%] Creating build directory...
    mkdir build
)

REM Navigate to build directory
cd build

REM Configure with CMake
echo [%time%] Configuring project with CMake...
"%cmake_path%" .. -DCMAKE_BUILD_TYPE=Debug
if %errorlevel% neq 0 (
    echo Error: CMake configuration failed
    cd ..
    exit /b 1
)

REM Build the project
echo [%time%] Building project...
"%cmake_path%" --build . --config Debug
if %errorlevel% neq 0 (
    echo Error: Project build failed
    cd ..
    exit /b 1
)

REM Copy executable to project root if it exists
if exist "Hyperblocks.exe" (
    echo [%time%] Copying executable to project root...
    copy "Hyperblocks.exe" "..\Hyperblocks.exe" >nul
)

cd ..

echo [%time%] Build completed successfully!
echo [%time%] You can now run: Hyperblocks.exe

pause

@echo off
echo Starting HyperBlocks CUDA build process...

REM Check for CMake
set cmake_path=cmake
where cmake >nul 2>&1
if errorlevel 1 (
    set cmake_path="C:\Program Files\CMake\bin\cmake.exe"
)

REM Check for nvcc
where nvcc >nul 2>&1
if errorlevel 1 (
    echo Error: nvcc not found. Please install CUDA Toolkit.
    pause
    exit /b 1
)

echo Required tools found successfully

REM Create build directory
if exist "build" (
    echo Build directory exists
) else (
    echo Creating build directory...
    mkdir build
)

REM Navigate to build directory
cd build

REM Configure with CMake
echo Configuring project with CMake...
%cmake_path% .. -DCMAKE_BUILD_TYPE=Debug
if errorlevel 1 (
    echo Error: CMake configuration failed
    cd ..
    pause
    exit /b 1
)

REM Build the project
echo Building project...
%cmake_path% --build . --config Debug
if errorlevel 1 (
    echo Error: Project build failed
    cd ..
    pause
    exit /b 1
)

REM Copy executable to project root
if exist "Hyperblocks.exe" (
    echo Copying executable to project root...
    copy "Hyperblocks.exe" "..\Hyperblocks.exe" >nul
)

cd ..

echo Build completed successfully!
echo You can now run: Hyperblocks.exe
pause

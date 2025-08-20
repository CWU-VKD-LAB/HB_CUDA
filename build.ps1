# HyperBlocks CUDA Build Script
# This script automates the build process for the HyperBlocks project

param(
    [string]$BuildType = "Debug",
    [switch]$Clean,
    [switch]$Help
)

# Show help if requested
if ($Help) {
    Write-Host "HyperBlocks CUDA Build Script" -ForegroundColor Green
    Write-Host "Usage: .\build.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -BuildType <type>    Build type: Debug, Release, RelWithDebInfo, MinSizeRel (default: Debug)" -ForegroundColor White
    Write-Host "  -Clean               Clean build directory before building" -ForegroundColor White
    Write-Host "  -Help                Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\build.ps1                    # Build with Debug configuration" -ForegroundColor White
    Write-Host "  .\build.ps1 -BuildType Release # Build with Release configuration" -ForegroundColor White
    Write-Host "  .\build.ps1 -Clean             # Clean and rebuild" -ForegroundColor White
    exit 0
}

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to print colored output
function Write-Status($message, $color = "White") {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $message" -ForegroundColor $color
}

# Function to check for errors
function Test-ExitCode($exitCode, $operation) {
    if ($exitCode -ne 0) {
        Write-Status "Error: $operation failed with exit code $exitCode" "Red"
        exit $exitCode
    }
}

# Main build script
Write-Status "Starting HyperBlocks CUDA build process..." "Green"

# Check if we're in the correct directory
if (-not (Test-Path "CMakeLists.txt")) {
    Write-Status "Error: CMakeLists.txt not found. Please run this script from the project root directory." "Red"
    exit 1
}

# Check for required tools
Write-Status "Checking for required tools..." "Yellow"

# Check for CMake
$cmakePath = $null
if (Test-Command "cmake") {
    $cmakePath = "cmake"
} elseif (Test-Path "C:\Program Files\CMake\bin\cmake.exe") {
    $cmakePath = "& `"C:\Program Files\CMake\bin\cmake.exe`""
} else {
    Write-Status "Error: CMake not found. Please install CMake and ensure it's in your PATH." "Red"
    exit 1
}

# Check for nvcc (CUDA compiler)
if (-not (Test-Command "nvcc")) {
    Write-Status "Error: nvcc (CUDA compiler) not found. Please install CUDA Toolkit." "Red"
    exit 1
}

Write-Status "Required tools found successfully" "Green"

# Clean build directory if requested
if ($Clean) {
    Write-Status "Cleaning build directory..." "Yellow"
    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
        Write-Status "Build directory cleaned" "Green"
    }
}

# Create build directory if it doesn't exist
if (-not (Test-Path "build")) {
    Write-Status "Creating build directory..." "Yellow"
    New-Item -ItemType Directory -Path "build" | Out-Null
}

# Navigate to build directory
Push-Location "build"

try {
    # Configure with CMake
    Write-Status "Configuring project with CMake..." "Yellow"
    $cmakeArgs = @("..", "-DCMAKE_BUILD_TYPE=$BuildType")
    
    if ($cmakePath -eq "cmake") {
        $result = & cmake @cmakeArgs
    } else {
        $result = Invoke-Expression "$cmakePath @cmakeArgs"
    }
    
    Test-ExitCode $LASTEXITCODE "CMake configuration"
    Write-Status "CMake configuration completed successfully" "Green"
    
    # Build the project
    Write-Status "Building project..." "Yellow"
    if ($cmakePath -eq "cmake") {
        $result = & cmake --build . --config $BuildType
    } else {
        $result = Invoke-Expression "$cmakePath --build . --config $BuildType"
    }
    
    Test-ExitCode $LASTEXITCODE "Project build"
    Write-Status "Project built successfully!" "Green"
    
    # Copy executable to project root if it exists
    $exeName = "Hyperblocks.exe"
    if (Test-Path $exeName) {
        Write-Status "Copying executable to project root..." "Yellow"
        Copy-Item $exeName "..\Hyperblocks.exe" -Force
        Write-Status "Executable copied to project root" "Green"
    }
    
    Write-Status "Build completed successfully!" "Green"
    Write-Status "You can now run: .\Hyperblocks.exe" "Cyan"
    
} catch {
    Write-Status "Error during build process: $($_.Exception.Message)" "Red"
    exit 1
} finally {
    # Return to original directory
    Pop-Location
}

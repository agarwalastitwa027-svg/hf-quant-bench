# Windows (PowerShell) version of build_llama_cpp.sh: one-time build of llama.cpp for the GGUF target.
# Builds CPU-only with MSVC (GGUF conversion and quantization are CPU-only anyway).
# Needs: Git, CMake, Ninja and Visual Studio Build Tools with the C++ workload.
#
# Usage: .\scripts\build_llama_cpp.ps1 [-InstallDir C:\dev\llama.cpp]
param([string]$InstallDir = 'C:\dev\llama.cpp')
$ErrorActionPreference = 'Stop'

if (Test-Path (Join-Path $InstallDir '.git')) {
    Write-Host "[build_llama_cpp] $InstallDir already exists, pulling latest"
    git -C $InstallDir pull --ff-only
} else {
    git clone https://github.com/ggml-org/llama.cpp $InstallDir
}

# Load the MSVC compiler environment if cl.exe isn't already on PATH.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'Visual Studio C++ build tools not found' }
    & (Join-Path $vs 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
}

cmake -S $InstallDir -B (Join-Path $InstallDir 'build') -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
if ($LASTEXITCODE) { throw 'cmake configure failed' }
cmake --build (Join-Path $InstallDir 'build') --config Release
if ($LASTEXITCODE) { throw 'cmake build failed' }

Write-Host ''
Write-Host 'Build complete. Set this (once, for your user) before running the gguf target:'
Write-Host "  [Environment]::SetEnvironmentVariable('LLAMA_CPP_DIR', '$InstallDir', 'User')"

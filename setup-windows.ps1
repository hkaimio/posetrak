# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
    One-time (and re-run-safe) setup for building posetrak natively on
    Windows with MSVC. See CONTRIBUTING.md's "Windows (native, MSVC)"
    section for what each step does and why.

.DESCRIPTION
    - Verifies Visual Studio's C++ toolchain, Meson, and Ninja are on PATH.
    - Creates (if missing) a dedicated conda environment holding Pinocchio
      3.9.0 headers + a compiled Boost Serialization -- this project never
      imports the Python package, only the C++ headers/libs it ships.
    - Configures AND builds builddir/ (debug -- day-to-day unit testing/
      debugging) and optbuild/ (release -- for actual tracking runs) with
      the options native Windows needs that Linux/WSL gets for free.
    - Copies the two runtime DLLs (boost_serialization.dll, yaml-cpp.dll)
      next to each built posetrak-tracker.exe / test_posetrak.exe.
      Windows always searches an executable's own directory for its DLL
      dependencies first, so this makes the binaries runnable as-is --
      including when launched as a subprocess from the Python UI, which
      does not (and should not have to) know about this conda environment
      or modify its own PATH to find them.

    Safe to re-run after a meson.build/meson_options.txt change: existing
    build directories are reconfigured (not recreated) in place, an
    existing conda environment is left alone, and the DLL copy is a
    harmless overwrite if already done.

.PARAMETER PinocchioVersion
    Pinocchio version to install. Must match whatever the project's
    Linux/WSL dev environment actually has at /opt/openrobots -- check with
    whoever maintains that environment, or the openrobots package site,
    before changing this. A mismatch shows up as real compile errors
    (e.g. a renamed struct field), not just a warning.
#>
param(
    [string]$PinocchioVersion = "3.9.0",
    [string]$CondaEnvName = "posetrak-pinocchio"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

function Test-CommandExists($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "== Checking toolchain ==" -ForegroundColor Cyan

if (-not (Test-CommandExists "cl")) {
    Write-Host "cl.exe not on PATH -- activating the VS 2022 x64 dev environment for this session." -ForegroundColor Yellow
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Visual Studio Installer not found. Install VS 2022+ with the 'Desktop development with C++' workload first."
    }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) {
        throw "No VS installation with the C++ workload found. Install the 'Desktop development with C++' workload first."
    }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    cmd /c "`"$vcvars`" && set" | ForEach-Object {
        if ($_ -match "^(.*?)=(.*)$") {
            Set-Item -Force -Path "env:\$($matches[1])" -Value $matches[2]
        }
    }
    if (-not (Test-CommandExists "cl")) {
        throw "Still no cl.exe after activating VS -- check the VS installation."
    }
}
Write-Host "  cl.exe: OK"

foreach ($tool in @("meson", "ninja")) {
    if (-not (Test-CommandExists $tool)) {
        throw "$tool not found on PATH. Install with: pip install meson ninja"
    }
    Write-Host "  ${tool}: OK"
}

if (-not (Test-CommandExists "conda")) {
    throw "conda not found on PATH. Install Miniconda/Anaconda first (used only to fetch Pinocchio headers/libs)."
}
Write-Host "  conda: OK"

Write-Host "`n== Pinocchio headers (conda env '$CondaEnvName') ==" -ForegroundColor Cyan

$envExists = (conda env list) -match ("^" + [regex]::Escape($CondaEnvName) + "\s")
if ($envExists) {
    Write-Host "  Environment '$CondaEnvName' already exists -- leaving it as-is."
    Write-Host "  (Delete it with 'conda env remove -n $CondaEnvName' first if you need a different Pinocchio version.)"
} else {
    Write-Host "  Creating '$CondaEnvName' with pinocchio=$PinocchioVersion from conda-forge (this downloads ~1GB, takes a few minutes)..."
    conda create -y -n $CondaEnvName -c conda-forge "pinocchio=$PinocchioVersion"
    if ($LASTEXITCODE -ne 0) {
        throw "conda create failed -- see output above."
    }
}

$condaBase = (conda info --base).Trim()
$pinocchioEnv = Join-Path $condaBase "envs\$CondaEnvName\Library"
if (-not (Test-Path (Join-Path $pinocchioEnv "include\pinocchio\config.hpp"))) {
    throw "Pinocchio headers not found under $pinocchioEnv -- environment creation may have failed silently."
}
Write-Host "  Headers found at: $pinocchioEnv\include"

Write-Host "`n== Configuring builds ==" -ForegroundColor Cyan

$commonArgs = @(
    "-Dpinocchio_includedir=$pinocchioEnv/include",
    "-Dboost_includedir=$pinocchioEnv/include",
    "-Dboost_libdir=$pinocchioEnv/lib",
    "-Ddefault_library=static"
)

$runtimeDlls = @(
    (Join-Path $pinocchioEnv "bin\boost_serialization.dll"),
    (Join-Path $condaBase "Library\bin\yaml-cpp.dll")
)
foreach ($dll in $runtimeDlls) {
    if (-not (Test-Path $dll)) {
        throw "Expected runtime DLL not found: $dll -- conda environment may be incomplete."
    }
}

function Copy-RuntimeDlls($Dir) {
    foreach ($subdir in @("cli", "tests")) {
        $target = Join-Path $RepoRoot "$Dir\$subdir"
        if (Test-Path $target) {
            Copy-Item -Force $runtimeDlls -Destination $target
        }
    }
}

function Set-MesonBuild($Dir, $Buildtype) {
    $buildFile = Join-Path $RepoRoot "$Dir\build.ninja"
    Push-Location $RepoRoot
    try {
        if (Test-Path $buildFile) {
            Write-Host "  Reconfiguring $Dir (buildtype=$Buildtype)..."
            & meson setup --reconfigure $Dir "-Dbuildtype=$Buildtype" @commonArgs
        } else {
            Write-Host "  Configuring $Dir (buildtype=$Buildtype)..."
            & meson setup $Dir "-Dbuildtype=$Buildtype" @commonArgs
        }
        if ($LASTEXITCODE -ne 0) {
            throw "meson setup failed for $Dir -- see output above."
        }
        Write-Host "  Building $Dir (this takes a while the first time)..."
        & meson compile -C $Dir
        if ($LASTEXITCODE -ne 0) {
            throw "meson compile failed for $Dir -- see output above."
        }
    } finally {
        Pop-Location
    }
    Copy-RuntimeDlls $Dir
}

Set-MesonBuild "builddir" "debug"
Set-MesonBuild "optbuild" "release"

Write-Host "`n== Done ==" -ForegroundColor Green
Write-Host "Both builddir/ (debug) and optbuild/ (release) are built, and the two runtime"
Write-Host "DLLs are copied next to each posetrak-tracker.exe / test_posetrak.exe -- no PATH"
Write-Host "changes needed to run them directly, including from the Python UI."
Write-Host ""
Write-Host "Rebuild:  meson compile -C builddir   /   meson compile -C optbuild"
Write-Host "Test:     meson test -C builddir"
Write-Host "(Re-run this script -- or just re-copy the two DLLs above into cli/ and tests/ --"
Write-Host " if you ever delete and recreate either build directory.)"

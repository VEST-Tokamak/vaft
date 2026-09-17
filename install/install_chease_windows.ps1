<#
.SYNOPSIS
    Build CHEASE for native Windows and make VAFT able to run it.

.DESCRIPTION
    Builds an existing CHEASE checkout with the MinGW-w64 gfortran toolchain
    from MSYS2, installs the resulting chease.exe and the libraries it needs
    into a self-contained prefix, and sets CHEASEHOME so that VAFT, JupyterLab
    and a plain terminal all find it.

    You obtain CHEASE yourself and pass its path. This script never clones,
    fetches, pulls, or changes the revision of your source tree: with more than
    one checkout on a machine, which one was built is a fact the operator
    states, not one a script infers. The only files it writes inside your
    source tree are the object files and the binary the upstream Makefile
    itself produces.

    Nothing is installed system-wide unless you pass -InstallToolchain.

    Prerequisites you install yourself: Git for Windows, Miniconda (for the
    verification step), and either MSYS2 or the -InstallToolchain switch.

.PARAMETER SourcePath
    Path to your existing CHEASE checkout. Required; never guessed.

.PARAMETER Prefix
    Where to install. Defaults to %LOCALAPPDATA%\vaft\external\chease. Must be
    outside both the VAFT checkout and the CHEASE source tree.

.PARAMETER Msys2Root
    MSYS2 installation root, when it is somewhere this script does not look.

.PARAMETER MinGWEnvironment
    Which MinGW-w64 environment to build with. ucrt64 by default, because
    CPython on Windows links the same UCRT.

.PARAMETER InstallToolchain
    Install MSYS2 with winget and the compiler packages with pacman. This is
    the only switch that changes anything outside the prefix.

.PARAMETER MaterializeSymlinks
    Replace the symbolic-link placeholders a Windows Git checkout leaves behind
    with copies of the files they name. This changes tracked files in your
    CHEASE checkout, so it is opt-in.

.PARAMETER Jobs
    Parallel compiler jobs. Defaults to 1: the upstream Makefile has a module
    rule that is not safe to run in parallel.

.PARAMETER Clean
    Run the upstream clean target first, and remove a stale chease.exe that
    the clean target does not know about.

.PARAMETER NoEnvironmentWiring
    Build and install, but do not set CHEASEHOME. The command to set it
    yourself is printed instead.

.PARAMETER CheckOnly
    Run install\check_chease.py and change nothing.

.PARAMETER Uninstall
    Remove what this script installed: the prefix and, when it still points
    there, the CHEASEHOME user variable. Your source tree is left alone.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\install_chease_windows.ps1 C:\git\CHEASE

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\install_chease_windows.ps1 C:\git\CHEASE -InstallToolchain

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\install_chease_windows.ps1 C:\git\CHEASE -CheckOnly
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)] [string] $SourcePath,
    [string] $Prefix,
    [string] $Msys2Root,
    [ValidateSet('ucrt64', 'mingw64')] [string] $MinGWEnvironment = 'ucrt64',
    [switch] $InstallToolchain,
    [switch] $MaterializeSymlinks,
    [ValidateRange(1, 64)] [int] $Jobs = 1,
    [switch] $Clean,
    [switch] $NoEnvironmentWiring,
    [switch] $CheckOnly,
    [switch] $Uninstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot '_external_code_common.ps1')

$CodeName = 'chease'
$HomeVariable = 'CHEASEHOME'
$Title = 'CHEASE (Windows native)'

# The plain `chease` target needs a Fortran compiler and make, and nothing
# else: its gfortran branch links no external BLAS, and the LAPACK routine it
# uses is compiled from the source tree.
$Packages = @('make', 'git', "$(Get-Msys2PackagePrefix -MinGWEnvironment $MinGWEnvironment)gcc-fortran")

Write-Host "VAFT external code: $Title"
Write-Host "Repository: $RepositoryRoot"
Write-Host ''

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

if ($Uninstall) {
    $target = $Prefix
    if (-not $target) { $target = Join-Path $env:LOCALAPPDATA "vaft\external\$CodeName" }
    if (Test-Path -LiteralPath $target) {
        $resolved = (Resolve-Path -LiteralPath $target).Path
        Remove-ExternalCodeEnvironment -Name $HomeVariable -ExpectedValue $resolved
        Remove-Item -LiteralPath $resolved -Recurse -Force
        Write-Result -Status PASS -Name 'Install prefix' -Detail "removed $resolved"
    }
    else {
        Write-Result -Status SKIP -Name 'Install prefix' -Detail "nothing at $target"
    }
    Write-ExternalSummary -Title $Title
    Write-Host ''
    Write-Host 'Your CHEASE source tree, including anything built inside it, was not touched.'
    exit 0
}

# ---------------------------------------------------------------------------
# Check only
# ---------------------------------------------------------------------------

if ($CheckOnly) {
    foreach ($pair in @(@('InstallToolchain', $InstallToolchain), @('MaterializeSymlinks', $MaterializeSymlinks), @('Clean', $Clean))) {
        if ($pair[1]) {
            Stop-WithGuidance "-CheckOnly changes nothing, so it cannot be combined with -$($pair[0])."
        }
    }
    $arguments = @('python', (Join-Path $RepositoryRoot 'install\check_chease.py'))
    if ($SourcePath) { $arguments += @('--source', $SourcePath) }
    if ($Prefix) { $arguments += @('--prefix', $Prefix) }
    Invoke-InVaft @($arguments)
    exit $LASTEXITCODE
}

if (-not $SourcePath) {
    Stop-WithGuidance @'
The path to your CHEASE checkout is required.

    powershell -ExecutionPolicy Bypass -File install\install_chease_windows.ps1 C:\git\CHEASE

This script never obtains CHEASE for you and never guesses where it lives.
See install\README.md.
'@
}

# ---------------------------------------------------------------------------
# Source, toolchain, prefix
# ---------------------------------------------------------------------------

$source = Assert-SourceCheckout -SourcePath $SourcePath -Project 'CHEASE' `
    -ExpectedFiles @('src-f90\Makefile', 'src-f90\Makefile.define_FLAGS', 'src-f90\chease_prog_effxml.f90')
$revision = Get-SourceRevision -SourcePath $source
Write-RevisionResult -Project 'CHEASE' -Revision $revision

$prefixPath = Resolve-InstallPrefix -Prefix $Prefix -CodeName $CodeName -RepositoryRoot $RepositoryRoot -SourcePath $source
$binDirectory = Join-Path $prefixPath 'bin'
New-Item -ItemType Directory -Path $binDirectory -Force | Out-Null

$root = Resolve-Toolchain -Explicit $Msys2Root -MinGWEnvironment $MinGWEnvironment -Packages $Packages -InstallToolchain:$InstallToolchain

# ---------------------------------------------------------------------------
# Symbolic links
#
# CHEASE commits symbolic links, one of which (euitm_schemas.f90) is compiled
# into the plain `chease` target. Git for Windows only creates real links when
# symlink support is enabled, so a default clone leaves a small text file
# holding the target's name, and gfortran fails on it with a syntax error that
# says nothing about the real cause.
# ---------------------------------------------------------------------------

function Resolve-LinkPlaceholder {
    param([Parameter(Mandatory)] [string] $Path)

    $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    if ($null -eq $item -or $item.Length -gt 512) { return $null }
    $content = (Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue)
    if (-not $content) { return $null }
    $name = $content.Trim()
    if ($name -match '[\r\n]' -or -not $name) { return $null }
    $sibling = Join-Path (Split-Path -Parent $Path) $name
    if (Test-Path -LiteralPath $sibling -PathType Leaf) { return $sibling }
    return $null
}

$sourceDirectory = Join-Path $source 'src-f90'
$placeholders = @()
foreach ($candidate in (Get-ChildItem -LiteralPath $sourceDirectory -Filter '*.f90' -File)) {
    if (Resolve-LinkPlaceholder -Path $candidate.FullName) { $placeholders += $candidate.FullName }
}

if ($placeholders.Count -gt 0) {
    if (-not $MaterializeSymlinks) {
        $listed = ($placeholders | ForEach-Object { Split-Path -Leaf $_ }) -join ', '
        Stop-WithGuidance @"
Your CHEASE checkout has symbolic links stored as plain text files:

    $listed

CHEASE commits these as links, and at least one of them is compiled into the
plain chease target. Git for Windows writes a small text file naming the target
instead of a link unless symlink support is on, so the build fails with a
Fortran syntax error that says nothing about the real cause.

Nothing is wrong with CHEASE and nothing is wrong with your toolchain.

Fix it once, in your own CHEASE tree, by obtaining it again with symlink
support enabled (see install\README.md), or rerun this script with
-MaterializeSymlinks to copy each target over its placeholder. That rewrites
the tracked files listed above in your CHEASE tree.
"@
    }
    # Resolve repeatedly: several of these links point at other links.
    for ($pass = 0; $pass -lt 4; $pass++) {
        $remaining = 0
        foreach ($candidate in (Get-ChildItem -LiteralPath $sourceDirectory -Filter '*.f90' -File)) {
            $target = Resolve-LinkPlaceholder -Path $candidate.FullName
            if ($target) {
                Copy-Item -LiteralPath $target -Destination $candidate.FullName -Force
                $remaining++
            }
        }
        if ($remaining -eq 0) { break }
    }
    Write-Result -Status PASS -Name 'Symbolic-link placeholders' -Detail "$($placeholders.Count) file(s) replaced with their targets"
}
else {
    Write-Result -Status PASS -Name 'Symbolic-link placeholders' -Detail 'none'
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

$logPath = Join-Path $prefixPath ('logs\chease-build-' + (Get-Date).ToString('yyyyMMdd-HHmmss') + '.log')
$built = Join-Path $sourceDirectory 'chease.exe'
if (Test-Path -LiteralPath $built) { Remove-Item -LiteralPath $built -Force }

$steps = New-Object System.Collections.Generic.List[string]
$steps.Add('set -euo pipefail')
# Leaked configuration silently changes what gets built.
$steps.Add('unset IMAS_HOME F90 F90FLAGS LIBS')
$steps.Add('cd "$(cygpath -u "$VAFT_SRC")/src-f90"')
if ($Clean) { $steps.Add('CHEASE_F90=gfortran CHEASE_MACHINE=linux_nohdf5 make clean || true') }
# CHEASE_MACHINE is not cosmetic: only the linux_nohdf5 branch supplies the
# double-precision flags, and the default `none` would build a numerically
# different code that still compiles. Exporting both variables also stops the
# Makefile including its host-detection fragments, which call dnsdomainname --
# a command MSYS2 does not ship.
#
# The literal goal must be `chease`: the Makefile keys its XML handling off
# MAKECMDGOALS being exactly that, and `make all` pulls in libxml2 instead.
$steps.Add("CHEASE_F90=gfortran CHEASE_MACHINE=linux_nohdf5 make -j$Jobs chease")

Write-Step 'Building CHEASE (a few minutes) ...'
Invoke-Msys2 -Msys2Root $root -MinGWEnvironment $MinGWEnvironment `
    -Command ($steps -join '; ') -Variables @{ VAFT_SRC = $source } -LogPath $logPath | Out-Null

if (-not (Test-Path -LiteralPath $built)) {
    Stop-WithGuidance "The build reported success but $built was not produced. The full log is in $logPath."
}
Write-Result -Status PASS -Name 'CHEASE build' -Detail $built

Copy-Item -LiteralPath $built -Destination (Join-Path $binDirectory 'chease.exe') -Force
Write-Result -Status PASS -Name 'Installed' -Detail (Join-Path $binDirectory 'chease.exe')

Copy-RuntimeDependencies -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -BinDirectory $binDirectory | Out-Null
if (-not (Test-ExecutableLoads -Executables @((Join-Path $binDirectory 'chease.exe')))) {
    Stop-WithGuidance 'The installed executable could not load its runtime libraries. See install\README.md.'
}

# ---------------------------------------------------------------------------
# Record and wire up
# ---------------------------------------------------------------------------

$record = @{
    code             = 'chease'
    prefix           = $prefixPath
    source           = $source
    source_revision  = if ($revision) { $revision.Revision } else { $null }
    source_described = if ($revision) { $revision.Described } else { $null }
    source_dirty     = if ($revision) { $revision.Dirty } else { $null }
    msys2_root       = $root
    mingw_env        = $MinGWEnvironment
    make_command     = "CHEASE_F90=gfortran CHEASE_MACHINE=linux_nohdf5 make -j$Jobs chease"
    executables      = @('bin\chease.exe')
    home_variable    = $HomeVariable
    build_log        = $logPath
}
Write-InstallManifest -Prefix $prefixPath -Record $record

if ($NoEnvironmentWiring) {
    Write-Result -Status SKIP -Name "$HomeVariable (user environment)" -Detail 'requested with -NoEnvironmentWiring'
}
else {
    Set-ExternalCodeEnvironment -Name $HomeVariable -Value $prefixPath
}

Write-ExternalSummary -Title $Title -NextSteps @(
    'powershell -ExecutionPolicy Bypass -File install\install_chease_windows.ps1 ' + $source + ' -CheckOnly',
    'conda activate vaft; jupyter lab'
)

if (-not $NoEnvironmentWiring) {
    Write-Host ''
    Write-Host 'Open a new terminal, or restart JupyterLab, so that CHEASEHOME reaches'
    Write-Host 'processes started from now on.'
}

if ($script:Failed) { exit 1 }

Write-Host ''
Write-Step 'Verifying the installation ...'
Write-Host ''
Invoke-InVaft @('python', (Join-Path $RepositoryRoot 'install\check_chease.py'), '--source', $source, '--prefix', $prefixPath)
exit $LASTEXITCODE

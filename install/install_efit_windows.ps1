<#
.SYNOPSIS
    Build EFIT and EFUND for native Windows and make VAFT able to run them.

.DESCRIPTION
    Builds an existing EFIT source tree with the MinGW-w64 toolchain from
    MSYS2, installs the reconstruction code and the Green-table generator into
    a self-contained prefix, and sets EFITHOME so that VAFT, JupyterLab and a
    plain terminal all find them.

    EFIT is licensed software. You obtain it yourself, through the EFIT-AI
    channel, and you state that you have agreed to its users agreement. This
    script never clones, fetches, pulls, mirrors or changes the revision of
    your source tree, and it will not build until you pass
    -AcceptEfitUsersAgreement. It writes nothing inside your tree: the build
    directory and the prefix both live outside it.

    Four things about a native Windows build differ from the Linux one, and all
    four are reported rather than hidden:

      * The stack is reserved at link time. On POSIX vaft.code.efit raises it
        per run with `ulimit -s`; a native Windows image takes its reserve from
        the PE header, fixed by the linker, so -StackReserveMB is passed
        through to -Wl,--stack and the adapter runs the executable directly.

      * BLAS is named again at the end of the link line. CMake places it ahead
        of liblsode and libr8slatec, which call dscal_, daxpy_ and dswap_;
        GNU ld makes a single pass and discards an archive whose members
        resolve nothing yet, so those symbols come out undefined. A Linux build
        does not notice because there BLAS is a shared object.

      * ctest is not run. All 102 of upstream's tests are #!/bin/bash drivers,
        and CTest on Windows hands a .sh to CreateProcess. install\check_efit.py
        verifies the build instead, including an EFUND run whose output tables
        are checked byte for byte.

      * The prefix is %LOCALAPPDATA%\vaft\external\efit, not the <source>/vaft-install
        that install_efit.sh uses. A build must never land inside a checkout,
        which is what Resolve-InstallPrefix enforces for every external code.
        EFITHOME is explicit either way.

    Nothing is installed system-wide unless you pass -InstallToolchain.

.PARAMETER SourcePath
    Path to your existing EFIT source tree. Required; never guessed.

.PARAMETER AcceptEfitUsersAgreement
    A statement by you, recorded in the build manifest, that you have read and
    agreed to the EFIT users agreement (LICENSE.rst in the source) and obtained
    the source through the authorized channel. This script never accepts it for
    you and will not build without it.

.PARAMETER Prefix
    Where to install. Defaults to %LOCALAPPDATA%\vaft\external\efit. Must be
    outside both the VAFT checkout and the EFIT source tree.

.PARAMETER NetcdfHome
    A netCDF installation to build against, enabling m-file output. Defaults to
    the MinGW environment's own when it carries the netCDF v2 Fortran API that
    EFIT calls. Without netCDF, EFIT writes no m-file.

.PARAMETER Msys2Root
    MSYS2 installation root, when it is somewhere this script does not look.

.PARAMETER MinGWEnvironment
    Which MinGW-w64 environment to build with. ucrt64 by default, because
    CPython on Windows links the same UCRT.

.PARAMETER InstallToolchain
    Install MSYS2 with winget and the compiler, CMake and library packages with
    pacman. This is the only switch that changes anything outside the prefix.

.PARAMETER Jobs
    Parallel compiler jobs. Defaults to the processor count, capped at 8.

.PARAMETER StackReserveMB
    Stack reserved for each executable. EFIT recurses deeply enough to need far
    more than the 1 MB Windows gives by default.

.PARAMETER WithoutNetcdf
    Build without netCDF even when one is available.

.PARAMETER Clean
    Remove the build directory first and configure from scratch.

.PARAMETER NoEnvironmentWiring
    Build and install, but do not set EFITHOME.

.PARAMETER CheckOnly
    Run install\check_efit.py and change nothing.

.PARAMETER Uninstall
    Remove what this script installed: the prefix and, when it still points
    there, the EFITHOME user variable. Your source tree is left alone.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\install_efit_windows.ps1 C:\git\efit -AcceptEfitUsersAgreement

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\install_efit_windows.ps1 C:\git\efit -CheckOnly
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)] [string] $SourcePath,
    [switch] $AcceptEfitUsersAgreement,
    [string] $Prefix,
    [string] $NetcdfHome,
    [string] $Msys2Root,
    [ValidateSet('ucrt64', 'mingw64')] [string] $MinGWEnvironment = 'ucrt64',
    [switch] $InstallToolchain,
    [ValidateRange(0, 64)] [int] $Jobs = 0,
    [ValidateRange(1, 2048)] [int] $StackReserveMB = 512,
    [switch] $WithoutNetcdf,
    [switch] $Clean,
    [switch] $NoEnvironmentWiring,
    [switch] $CheckOnly,
    [switch] $Uninstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot '_external_code_common.ps1')

$CodeName = 'efit'
$HomeVariable = 'EFITHOME'
$Title = 'EFIT/EFUND (Windows native)'

# Both roles, from one configure of one revision. A Green-function table is
# only as reproducible as the pair that produced and consumed it, which is why
# they are never installed separately.
$Executables = @('efit', 'efund')

# Where CMake leaves them; upstream has no install(TARGETS) for either.
$BuiltAt = @{ efit = 'efit/efit.exe'; efund = 'green/efund.exe' }

$prefixToken = Get-Msys2PackagePrefix -MinGWEnvironment $MinGWEnvironment
$Packages = @(
    'make', 'git', 'diffutils',
    ($prefixToken + 'gcc-fortran'),
    ($prefixToken + 'cmake'),
    ($prefixToken + 'openblas'),
    ($prefixToken + 'netcdf-fortran')
)

if ($Jobs -eq 0) {
    $Jobs = [Math]::Min([Environment]::ProcessorCount, 8)
}

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
    Write-Host 'Your EFIT source tree was not touched; this script never wrote inside it.'
    exit 0
}

# ---------------------------------------------------------------------------
# Check only
# ---------------------------------------------------------------------------

if ($CheckOnly) {
    foreach ($pair in @(@('InstallToolchain', $InstallToolchain), @('Clean', $Clean))) {
        if ($pair[1]) {
            Stop-WithGuidance "-CheckOnly changes nothing, so it cannot be combined with -$($pair[0])."
        }
    }
    $arguments = @('python', (Join-Path $RepositoryRoot 'install\check_efit.py'))
    if ($SourcePath) { $arguments += @('--source', $SourcePath) }
    if ($Prefix) { $arguments += @('--prefix', $Prefix) }
    Invoke-InVaft @($arguments)
    exit $LASTEXITCODE
}

if (-not $SourcePath) {
    Stop-WithGuidance @'
The path to your EFIT source tree is required.

    powershell -ExecutionPolicy Bypass -File install\install_efit_windows.ps1 C:\git\efit -AcceptEfitUsersAgreement

EFIT is licensed software. This script never obtains it for you and never
guesses where it lives. See install\README.md.
'@
}

if (-not $AcceptEfitUsersAgreement) {
    Stop-WithGuidance @'
EFIT is distributed under a users agreement that forbids redistributing the
original or modified sources and asks every user to register with the authors.

Read the agreement that accompanies your source (LICENSE.rst), and if you have
agreed to it and obtained the source through the authorized EFIT-AI channel,
rerun with -AcceptEfitUsersAgreement. That switch is a statement by you, and it
is recorded in the build manifest. This script never accepts it for you.
'@
}

# ---------------------------------------------------------------------------
# Source, toolchain, prefix
# ---------------------------------------------------------------------------

$source = Assert-SourceCheckout -SourcePath $SourcePath -Project 'EFIT' `
    -ExpectedFiles @('CMakeLists.txt', 'efit/efit.F90', 'green/efund.f90', 'LICENSE.rst')
$revision = Get-SourceRevision -SourcePath $source
Write-RevisionResult -Project 'EFIT' -Revision $revision
if (-not $revision) {
    Write-Host '      A source snapshot rather than a repository: the manifest can record'
    Write-Host '      what was built but not which revision it came from.'
}

$prefixPath = Resolve-InstallPrefix -Prefix $Prefix -CodeName $CodeName `
    -RepositoryRoot $RepositoryRoot -SourcePath $source
$binDirectory = Join-Path $prefixPath 'bin'
New-Item -ItemType Directory -Path $binDirectory -Force | Out-Null

$root = Resolve-Toolchain -Explicit $Msys2Root -MinGWEnvironment $MinGWEnvironment `
    -Packages $Packages -InstallToolchain:$InstallToolchain

# ---------------------------------------------------------------------------
# netCDF
# ---------------------------------------------------------------------------

$netcdfUnix = ''
if ($WithoutNetcdf) {
    Write-Result -Status SKIP -Name 'netCDF' -Detail 'requested with -WithoutNetcdf'
}
else {
    $candidate = $NetcdfHome
    if (-not $candidate) { $candidate = Join-Path $root $MinGWEnvironment }
    # EFIT calls the netCDF v2 Fortran API (NCCRE, NCAPTC in write_m.F90), which
    # a netCDF-Fortran built without --enable-v2 does not export. Check for the
    # header rather than discovering it at the link.
    if (Test-Path -LiteralPath (Join-Path $candidate 'include\netcdf.inc')) {
        $netcdfUnix = ConvertTo-Msys2Path -WindowsPath $candidate
        Write-Result -Status PASS -Name 'netCDF' -Detail $candidate
    }
    else {
        Write-Result -Status SKIP -Name 'netCDF' -Detail `
            "no netcdf.inc under $candidate; building without m-file output"
    }
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $prefixPath "logs\efit-build-$timestamp.log"
$buildDirectory = Join-Path $prefixPath 'build'
if ($Clean -and (Test-Path -LiteralPath $buildDirectory)) {
    Remove-Item -LiteralPath $buildDirectory -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
foreach ($name in $Executables) {
    $stale = Join-Path $buildDirectory $BuiltAt[$name]
    if (Test-Path -LiteralPath $stale) { Remove-Item -LiteralPath $stale -Force }
}

$steps = New-Object System.Collections.Generic.List[string]
$steps.Add('set -euo pipefail')
$steps.Add('src="$(cygpath -u "$VAFT_SRC")"')
$steps.Add('build="$(cygpath -u "$VAFT_BUILD")"')
$steps.Add('cd "$build"')
$steps.Add('args=(-G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=gfortran)')
# Upstream has no find_package(MPI) and uses `include "mpif.h"`; MS-MPI ships no
# gfortran-usable one. MDSplus and the DIII-D libraries are not obtainable here,
# which restricts EFIT to k-file input -- what every non-snap test uses anyway.
$steps.Add('args+=(-DENABLE_PARALLEL=OFF -DENABLE_MDSPLUS=OFF -DTEST_EFUND=ON)')
$steps.Add('args+=(-DCMAKE_EXE_LINKER_FLAGS="-Wl,--stack,$VAFT_STACK")')
# CMake puts BLAS ahead of liblsode and libr8slatec, which call into it. GNU ld
# is single-pass, so naming it again at the very end -- which is where
# CMAKE_Fortran_STANDARD_LIBRARIES lands -- is what resolves dscal_, daxpy_ and
# dswap_. A Linux build never notices, because there BLAS is a shared object.
$steps.Add('args+=(-DCMAKE_Fortran_STANDARD_LIBRARIES="-L/$VAFT_ENV/lib -lopenblas")')
# ${VAFT_NETCDF:-}, not "$VAFT_NETCDF": Windows deletes an environment variable
# that is set to the empty string, so with no netCDF the name does not reach the
# shell at all -- and `set -u` aborts the build on an unset variable.
$steps.Add('if [ -n "${VAFT_NETCDF:-}" ]; then args+=(-DENABLE_NETCDF=ON -DNetCDF_DIR="$VAFT_NETCDF"); else args+=(-DENABLE_NETCDF=OFF); fi')
$steps.Add('cmake "$src" "${args[@]}"')
$steps.Add('cmake --build . -j "$VAFT_JOBS"')

Write-Step "Building EFIT and EFUND (this takes several minutes) ..."
Invoke-Msys2 -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -Command ($steps -join "`n") `
    -Variables @{
        VAFT_SRC    = $source
        VAFT_BUILD  = $buildDirectory
        VAFT_ENV    = $MinGWEnvironment
        VAFT_NETCDF = $netcdfUnix
        VAFT_JOBS   = $Jobs
        VAFT_STACK  = ('0x' + ('{0:X}' -f ($StackReserveMB * 1MB)))
    } -LogPath $logPath -AllowFailure | Out-Null

# Judge by the artifacts, not by CMake's exit code.
$missing = @()
foreach ($name in $Executables) {
    $built = Join-Path $buildDirectory $BuiltAt[$name]
    if ((Test-Path -LiteralPath $built) -and ((Get-Item -LiteralPath $built).Length -gt 0)) {
        Copy-Item -LiteralPath $built -Destination (Join-Path $binDirectory "$name.exe") -Force
    }
    else {
        $missing += $name
    }
}

$built = @($Executables | Where-Object { $missing -notcontains $_ })
if ($built.Count -gt 0) {
    Copy-RuntimeDependencies -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -BinDirectory $binDirectory | Out-Null
}

if ($missing.Count -gt 0) {
    # One upstream defect stops efit while letting efund through, and a reader
    # deserves to be told which it is rather than handed a Fortran diagnostic.
    $log = ''
    if (Test-Path -LiteralPath $logPath) { $log = Get-Content -LiteralPath $logPath -Raw }
    if ($log -match "Symbol 'jtime'.*has no IMPLICIT type") {
        Stop-WithGuidance @"
EFIT did not build, and the reason is in your source rather than in this script.

    efit/efit.F90: 'jtime' is declared only inside #if defined(USEMPI), but
    write_m and write_ot are called with it unconditionally.

So efit.F90 cannot compile with MPI disabled, which is the default
(ENABLE_PARALLEL is FALSE in upstream's own CMakeLists.txt). This is not a
Windows problem: the same configure fails on Linux and macOS.

EFUND built and was installed, so Green-function tables can still be generated.

VAFT cannot ship a patch for this -- the users agreement forbids distributing
modified sources. As a licensee you may fix your own tree, by moving the jtime
declaration out of the USEMPI block; please also report it to
efit-support@fusion.gat.com so the fix reaches everyone.

The full build log is at $logPath.
"@
    }
    Stop-WithGuidance "The build did not produce: $($missing -join ', '). The full log is at $logPath."
}
Write-Result -Status PASS -Name 'EFIT build' -Detail (($Executables | ForEach-Object { "$_.exe" }) -join ', ')

if (-not (Test-ExecutableLoads -Executables (@($Executables | ForEach-Object { Join-Path $binDirectory "$_.exe" })))) {
    Stop-WithGuidance 'The installed executables could not load their runtime libraries. See install\README.md.'
}

# ---------------------------------------------------------------------------
# Record and wire up
# ---------------------------------------------------------------------------

$record = @{
    code                          = $CodeName
    prefix                        = $prefixPath
    source                        = $source
    source_revision               = $(if ($revision) { $revision.Revision } else { $null })
    source_described              = $(if ($revision) { $revision.Described } else { $null })
    source_dirty                  = $(if ($revision) { $revision.Dirty } else { $null })
    efit_users_agreement_accepted = $true
    build_dir                     = $buildDirectory
    build_type                    = 'Release'
    netcdf                        = $(if ($netcdfUnix) { $netcdfUnix } else { $false })
    stack_reserve_bytes           = ($StackReserveMB * 1MB)
    msys2_root                    = $root
    mingw_env                     = $MinGWEnvironment
    executables                   = @($Executables | ForEach-Object { "bin\$_.exe" })
    home_variable                 = $HomeVariable
    build_log                     = $logPath
}
Write-InstallManifest -Prefix $prefixPath -Record $record

if ($NoEnvironmentWiring) {
    Write-Result -Status SKIP -Name "$HomeVariable (user environment)" -Detail 'requested with -NoEnvironmentWiring'
}
else {
    Set-ExternalCodeEnvironment -Name $HomeVariable -Value $prefixPath
}

Write-ExternalSummary -Title $Title -NextSteps @(
    "powershell -ExecutionPolicy Bypass -File install\install_efit_windows.ps1 $source -CheckOnly",
    'conda activate vaft; jupyter lab'
)

Write-Host ''
Write-Host 'Green-function tables are not shipped with EFIT: generate them with efund'
Write-Host 'for the grid you intend to run. Tables and efit must come from the same'
Write-Host 'build, because upstream writes them big-endian by default.'
Write-Host ''
Write-Host "This build is serial, and upstream's own test suite is not run: its 102"
Write-Host 'tests are bash drivers that CTest cannot start on Windows. check_efit.py'
Write-Host 'verifies the build instead.'

if ($script:Failed) { exit 1 }

Write-Host ''
Write-Step 'Verifying the installation ...'
Write-Host ''
Invoke-InVaft @('python', (Join-Path $RepositoryRoot 'install\check_efit.py'), '--source', $source, '--prefix', $prefixPath)
exit $LASTEXITCODE

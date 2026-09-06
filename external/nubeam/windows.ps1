<#
.SYNOPSIS
    Build the serial NTCC NUBEAM distribution for native Windows and make VAFT
    able to run it.

.DESCRIPTION
    Builds an existing NUBEAM source tree with the MinGW-w64 toolchain from
    MSYS2, downloads and builds the three NTCC dependency modules it needs
    (PSPLINE, PREACT, XPLASMA), stages the PREACT and ADAS reaction databases,
    and sets NUBEAMHOME so that VAFT, JupyterLab and a plain terminal all find
    the result.

    The build recipe itself lives in external\nubeam\windows.sh and runs inside
    MSYS2. This script owns the Windows side: finding MSYS2, reporting which
    revision you built, colocating the runtime DLLs, and wiring the
    environment. external\nubeam\macos.sh is the same recipe for Apple Silicon.

    You obtain NUBEAM yourself and pass its path. This script never clones,
    fetches, pulls, or changes the revision of your source tree. What it does
    generate -- the install prefix, the object tree, the downloaded dependency
    sources and the generated share\Make.local files -- all lives inside that
    tree, and -Uninstall removes exactly those.

    Four things about a native Windows build differ from the Linux one, and
    all four are reported rather than hidden:

      * portlib's c_execsystem.c cannot be compiled at all: fork, execvp,
        clearenv, O_NONBLOCK and <sys/wait.h> are all absent on MinGW-w64. A
        Windows implementation over _spawnvpe is compiled in its place, and
        the shell-server optimisation it also provides reports itself
        unavailable, which its Fortran caller already handles.

      * portlib's trsocket.c uses close, read and write on sockets. A Winsock
        handle is not a file descriptor, so that file alone is compiled with
        those three redirected.

      * netCDF must be one built without the S3 and NCZarr backends. MSYS2's
        own package pulls in the AWS C++ SDK, whose atexit handler deadlocks
        after the program has finished -- the same defect that hung DCON.
        install\install_gpec_windows.ps1 -BuildDependencies produces a
        suitable prefix, and this script finds it by default.

      * NUBEAM composes every filename in a character*140 buffer. Nothing here
        can widen it, so install\check_nubeam.py reports the remaining budget
        and vaft.compat.short_temporary_directory picks a run directory that
        fits.

    Nothing is installed system-wide unless you pass -InstallToolchain.

.PARAMETER SourcePath
    Path to your existing NUBEAM source tree. Required; never guessed.

.PARAMETER AcceptNtccTerms
    You have read and accepted the NTCC agreement at
    https://w3.pppl.gov/NTCC/NUBEAM/downloads.shtml. Required before this
    script downloads PSPLINE, PREACT and XPLASMA. It is never implied.

.PARAMETER NetcdfHome
    A netCDF installation built without S3. Defaults to the one
    install\install_gpec_windows.ps1 -BuildDependencies leaves under
    %LOCALAPPDATA%\vaft\external\gpec\deps.

.PARAMETER Msys2Root
    MSYS2 installation root, when it is somewhere this script does not look.

.PARAMETER MinGWEnvironment
    Which MinGW-w64 environment to build with. ucrt64 by default, because
    CPython on Windows links the same UCRT.

.PARAMETER InstallToolchain
    Install MSYS2 with winget and the compiler packages with pacman. This is
    the only switch that changes anything outside your source tree.

.PARAMETER Resume
    Continue a build that stopped partway, reusing what it already produced.

.PARAMETER NoEnvironmentWiring
    Build and install, but do not set NUBEAMHOME.

.PARAMETER CheckOnly
    Run install\check_nubeam.py and change nothing.

.PARAMETER Uninstall
    Remove what this script generated inside your source tree -- local\,
    build\windows-x86_64\, vendor\ntcc\, the generated share\Make.local files
    and the manifest -- and, when it still points there, the NUBEAMHOME user
    variable. Nothing else in your source tree is touched.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File external\nubeam\windows.ps1 C:\git\NUBEAM -AcceptNtccTerms

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File external\nubeam\windows.ps1 C:\git\NUBEAM -CheckOnly
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)] [string] $SourcePath,
    [switch] $AcceptNtccTerms,
    [string] $NetcdfHome,
    [string] $Msys2Root,
    [ValidateSet('ucrt64', 'mingw64')] [string] $MinGWEnvironment = 'ucrt64',
    [switch] $InstallToolchain,
    [switch] $Resume,
    [switch] $NoEnvironmentWiring,
    [switch] $CheckOnly,
    [switch] $Uninstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $RepositoryRoot 'install\_external_code_common.ps1')

$CodeName = 'nubeam'
$HomeVariable = 'NUBEAMHOME'
$Title = 'NUBEAM (Windows native)'

# What a NUBEAM case drives, in the order it uses them.
$Executables = @('nubeam_comp_exec')

# The generated names, relative to the source tree. -Uninstall removes exactly
# these, which is also the list macos.sh's uninstall.sh removes.
$GeneratedPaths = @('local', 'build\windows-x86_64', 'vendor\ntcc', '.nubeam-install-manifest')

$prefixToken = Get-Msys2PackagePrefix -MinGWEnvironment $MinGWEnvironment
$Packages = @(
    'make', 'git', 'curl', 'diffutils', 'tar',
    ($prefixToken + 'gcc'),
    ($prefixToken + 'gcc-fortran'),
    ($prefixToken + 'openblas'),
    ($prefixToken + 'python')
)

Write-Host "VAFT external code: $Title"
Write-Host "Repository: $RepositoryRoot"
Write-Host ''

function Get-NubeamPrefix {
    param([Parameter(Mandatory)] [string] $Source)
    # windows.sh keeps everything it generates inside the source tree, exactly
    # as macos.sh does, so the prefix follows from the source path rather than
    # being a separate choice the two platforms could disagree about.
    Join-Path $Source 'local'
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

if ($Uninstall) {
    if (-not $SourcePath) {
        Stop-WithGuidance @'
-Uninstall needs the NUBEAM source tree, because everything this script
generates lives inside it.

    powershell -ExecutionPolicy Bypass -File external\nubeam\windows.ps1 C:\git\NUBEAM -Uninstall
'@
    }
    $source = (Resolve-Path -LiteralPath $SourcePath).Path
    $prefix = Get-NubeamPrefix -Source $source
    if (Test-Path -LiteralPath $prefix) {
        Remove-ExternalCodeEnvironment -Name $HomeVariable -ExpectedValue (Resolve-Path -LiteralPath $prefix).Path
    }
    foreach ($relative in $GeneratedPaths) {
        $target = Join-Path $source $relative
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
            Write-Result -Status PASS -Name $relative -Detail 'removed'
        }
        else {
            Write-Result -Status SKIP -Name $relative -Detail 'not present'
        }
    }
    # Only the Make.local files this script generated; a hand-written one is
    # left alone, which is the same rule the build applies before overwriting.
    foreach ($makeLocal in @(Get-ChildItem -Path $source -Recurse -Filter 'Make.local' -File -ErrorAction SilentlyContinue)) {
        if (Select-String -LiteralPath $makeLocal.FullName -Pattern 'Generated by VAFT external/nubeam' -Quiet) {
            Remove-Item -LiteralPath $makeLocal.FullName -Force
            Write-Result -Status PASS -Name 'Generated Make.local' -Detail $makeLocal.FullName
        }
    }
    Write-ExternalSummary -Title $Title
    Write-Host ''
    Write-Host 'Your NUBEAM source tree is otherwise untouched.'
    exit 0
}

# ---------------------------------------------------------------------------
# Check only
# ---------------------------------------------------------------------------

if ($CheckOnly) {
    foreach ($pair in @(@('InstallToolchain', $InstallToolchain), @('AcceptNtccTerms', $AcceptNtccTerms))) {
        if ($pair[1]) {
            Stop-WithGuidance "-CheckOnly changes nothing, so it cannot be combined with -$($pair[0])."
        }
    }
    $arguments = @('python', (Join-Path $RepositoryRoot 'install\check_nubeam.py'))
    if ($SourcePath) {
        $arguments += @('--source', $SourcePath)
        $arguments += @('--prefix', (Get-NubeamPrefix -Source (Resolve-Path -LiteralPath $SourcePath).Path))
    }
    Invoke-InVaft @($arguments)
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Source tree
# ---------------------------------------------------------------------------

if (-not $SourcePath) {
    Stop-WithGuidance @'
The path to your NUBEAM source tree is required.

    powershell -ExecutionPolicy Bypass -File external\nubeam\windows.ps1 C:\git\NUBEAM -AcceptNtccTerms

VAFT does not vendor NUBEAM: NTCC requires each user to accept its licence
before downloading it. Obtain the source from
https://w3.pppl.gov/NTCC/NUBEAM/ and pass its path.
'@
}

if (-not $AcceptNtccTerms) {
    Stop-WithGuidance @'
NUBEAM needs three NTCC dependency modules -- PSPLINE, PREACT and XPLASMA --
which this script downloads from PPPL.

Read the agreement at https://w3.pppl.gov/NTCC/NUBEAM/downloads.shtml, and if
you accept it, rerun with -AcceptNtccTerms. This script never accepts it for
you, and never downloads anything without that switch.
'@
}

$source = Assert-SourceCheckout -SourcePath $SourcePath -Project 'NUBEAM' `
    -ExpectedFiles @('Makefile', 'nubeam_comp_exec')
Write-RevisionResult -Project 'NUBEAM' -SourcePath $source

$prefix = Get-NubeamPrefix -Source $source
$binDirectory = Join-Path $prefix 'bin'

# ---------------------------------------------------------------------------
# Toolchain and dependencies
# ---------------------------------------------------------------------------

$root = Resolve-Toolchain -Explicit $Msys2Root -MinGWEnvironment $MinGWEnvironment `
    -Packages $Packages -InstallToolchain:$InstallToolchain

if (-not $NetcdfHome) {
    $candidate = Join-Path $env:LOCALAPPDATA 'vaft\external\gpec\deps'
    if (Test-Path -LiteralPath (Join-Path $candidate 'lib\libnetcdff.a')) {
        $NetcdfHome = $candidate
        Write-Result -Status PASS -Name 'netCDF prefix' -Detail $NetcdfHome
    }
    else {
        Stop-WithGuidance @"
No netCDF built without S3 was found at $candidate.

MSYS2's own netcdf package links the AWS C++ SDK, whose atexit handler
deadlocks after the program has finished, so NUBEAM would compute correctly
and then never exit. Build a usable one once:

    powershell -ExecutionPolicy Bypass -File install\install_gpec_windows.ps1 <gpec-source> -BuildDependencies

or pass -NetcdfHome pointing at a static netCDF-Fortran prefix of your own.
"@
    }
}
else {
    $NetcdfHome = (Resolve-Path -LiteralPath $NetcdfHome).Path
    if (-not (Test-Path -LiteralPath (Join-Path $NetcdfHome 'lib\libnetcdff.a'))) {
        Stop-WithGuidance "-NetcdfHome has no lib\libnetcdff.a: $NetcdfHome"
    }
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

Write-Step 'Building NUBEAM and its NTCC dependencies'
Write-Host 'This takes roughly an hour on a first run, most of it in PREACT.'
Write-Host ''

$logDirectory = Join-Path $source 'build\windows-x86_64'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory 'windows.ps1.log'

$steps = @(
    'set -e',
    'recipe="$(cygpath -u "$VAFT_RECIPE")"',
    'nubeam="$(cygpath -u "$VAFT_SRC")"',
    'netcdf="$(cygpath -u "$VAFT_NETCDF")"',
    'bash "$recipe" --nubeam-root "$nubeam" --netcdf-home "$netcdf" --accept-ntcc-terms $VAFT_RESUME'
)

$exitCode = Invoke-Msys2 -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -Command ($steps -join "`n") `
    -Variables @{
        VAFT_RECIPE = (Join-Path $PSScriptRoot 'windows.sh')
        VAFT_SRC    = $source
        VAFT_NETCDF = $NetcdfHome
        VAFT_RESUME = $(if ($Resume) { '--resume' } else { '' })
    } -LogPath $logPath -AllowFailure

if ($exitCode -ne 0) {
    Stop-WithGuidance "The NUBEAM build failed (exit $exitCode). The full log is at $logPath and at $logDirectory\install.log."
}

$missing = @()
foreach ($name in $Executables) {
    if (-not (Test-Path -LiteralPath (Join-Path $binDirectory "$name.exe"))) { $missing += $name }
}
if ($missing.Count -gt 0) {
    Stop-WithGuidance "The build reported success but did not produce: $($missing -join ', '). See $logDirectory\install.log."
}
Write-Result -Status PASS -Name 'Executables' -Detail (($Executables | ForEach-Object { "$_.exe" }) -join ', ')

# ---------------------------------------------------------------------------
# Make the prefix self-contained
# ---------------------------------------------------------------------------

Copy-RuntimeDependencies -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -BinDirectory $binDirectory
Test-ExecutableLoads -Executables (@($Executables | ForEach-Object { Join-Path $binDirectory "$_.exe" }))

Write-InstallManifest -Prefix $prefix -Record @{
    code        = $CodeName
    source      = $source
    revision    = (Get-SourceRevision -SourcePath $source)
    netcdf      = $NetcdfHome
    msys2       = $root
    environment = $MinGWEnvironment
    recipe      = 'external/nubeam/windows.sh'
}

if (-not $NoEnvironmentWiring) {
    Set-ExternalCodeEnvironment -Name $HomeVariable -Value $prefix
}
else {
    Write-Result -Status SKIP -Name $HomeVariable -Detail '-NoEnvironmentWiring was given'
}

Write-ExternalSummary -Title $Title -NextSteps @(
    "Verify the installation:  powershell -File external\nubeam\windows.ps1 $source -CheckOnly",
    'Reproduce the reference cases:  bash external/nubeam/run-local-validation.sh --case d3d',
    'NUBEAM results are returned as a native container; nothing maps them into IMAS yet (issue #490).'
)

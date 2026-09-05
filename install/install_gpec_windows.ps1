<#
.SYNOPSIS
    Build the DCON/GPEC suite for native Windows and make VAFT able to run it.

.DESCRIPTION
    Builds an existing GPEC checkout with the MinGW-w64 gfortran toolchain from
    MSYS2, installs dcon, rdcon, stride, gpec, match and rmatch together with
    the libraries they need into a self-contained prefix, and sets GPECHOME so
    that VAFT, JupyterLab and a plain terminal all find them.

    You obtain GPEC yourself and pass its path. This script never clones,
    fetches, pulls, or changes the revision of your source tree, and never
    initialises a submodule: which revision was built is a fact the operator
    states, not one a script infers. The only files it writes inside your
    source tree are the object files and binaries the upstream Makefile itself
    produces, all of which upstream already ignores.

    Two things about a native Windows build differ from the upstream Linux one,
    and both are reported rather than hidden:

      * OpenMP is off. LSODE and ZVODE mark a COMMON block threadprivate, and
        gfortran expresses that with a directive the PE object format has no
        equivalent for, so an OpenMP build cannot assemble at all. The result is
        correct and serial.

      * HDF5 and netCDF are built here rather than taken from MSYS2, when you
        pass -BuildDependencies. Both MSYS2 packages link the AWS C++ S3 SDK,
        whose shutdown handler waits on a condition variable that is never
        signalled: a solver built against them writes every output, prints its
        normal termination message, and then never exits, and cannot be killed.
        Cutting S3 out of netCDF alone does not help, because HDF5 pulls the
        same SDK in through its ROS3 driver, so both are built.

    Nothing is installed system-wide unless you pass -InstallToolchain.

.PARAMETER SourcePath
    Path to your existing GPEC checkout. Required; never guessed.

.PARAMETER Prefix
    Where to install. Defaults to %LOCALAPPDATA%\vaft\external\gpec. Must be
    outside both the VAFT checkout and the GPEC source tree.

.PARAMETER NetcdfHome
    An existing netCDF installation to build against. Defaults to the one
    -BuildDependencies puts under the prefix, and falls back to the MinGW
    environment's -- which cannot terminate, for the reason above.

.PARAMETER BuildDependencies
    Compile HDF5 and netCDF without S3 into the prefix. Needed once per prefix;
    without it the suite computes correctly and then never exits.

.PARAMETER Msys2Root
    MSYS2 installation root, when it is somewhere this script does not look.

.PARAMETER MinGWEnvironment
    Which MinGW-w64 environment to build with. ucrt64 by default, because
    CPython on Windows links the same UCRT.

.PARAMETER InstallToolchain
    Install MSYS2 with winget and the compiler packages with pacman. This is
    the only switch that changes anything outside the prefix.

.PARAMETER Jobs
    Parallel compiler jobs. Defaults to the processor count, capped at 8.

.PARAMETER StackReserveMB
    Stack reserved for each executable. DCON and STRIDE need far more than the
    1 MB Windows gives by default; upstream's own Linux runs lift the limit too.

.PARAMETER Clean
    Run the upstream clean target first, and remove stale binaries the clean
    target does not know about.

.PARAMETER NoEnvironmentWiring
    Build and install, but do not set GPECHOME.

.PARAMETER CheckOnly
    Run install\check_gpec.py and change nothing.

.PARAMETER Uninstall
    Remove what this script installed: the prefix and, when it still points
    there, the GPECHOME user variable. Your source tree is left alone.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\install_gpec_windows.ps1 C:\git\GPEC -BuildDependencies

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\install_gpec_windows.ps1 C:\git\GPEC -CheckOnly
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)] [string] $SourcePath,
    [string] $Prefix,
    [string] $NetcdfHome,
    [switch] $BuildDependencies,
    [string] $Msys2Root,
    [ValidateSet('ucrt64', 'mingw64')] [string] $MinGWEnvironment = 'ucrt64',
    [switch] $InstallToolchain,
    [ValidateRange(0, 64)] [int] $Jobs = 0,
    [ValidateRange(1, 2048)] [int] $StackReserveMB = 512,
    [switch] $Clean,
    [switch] $NoEnvironmentWiring,
    [switch] $CheckOnly,
    [switch] $Uninstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot '_external_code_common.ps1')

$CodeName = 'gpec'
$HomeVariable = 'GPECHOME'
$Title = 'DCON/GPEC (Windows native)'

# The six VAFT drives, in the order the suite depends on them.
$Executables = @('dcon', 'match', 'rdcon', 'rmatch', 'stride', 'gpec')

# `mkbin`, never `all`: `all` also builds xdraw, an X11 viewer the computational
# workflow does not use and which has no place in a Windows install.
$MakeTarget = 'mkbin'

$prefixToken = Get-Msys2PackagePrefix -MinGWEnvironment $MinGWEnvironment
$Packages = @(
    'make', 'git', 'm4', 'diffutils',
    ($prefixToken + 'gcc'),
    ($prefixToken + 'gcc-fortran'),
    ($prefixToken + 'openblas'),
    ($prefixToken + 'hdf5')
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
    Write-Host 'Your GPEC source tree, including anything built inside it, was not touched.'
    exit 0
}

# ---------------------------------------------------------------------------
# Check only
# ---------------------------------------------------------------------------

if ($CheckOnly) {
    foreach ($pair in @(@('InstallToolchain', $InstallToolchain), @('BuildDependencies', $BuildDependencies), @('Clean', $Clean))) {
        if ($pair[1]) {
            Stop-WithGuidance "-CheckOnly changes nothing, so it cannot be combined with -$($pair[0])."
        }
    }
    $arguments = @('python', (Join-Path $RepositoryRoot 'install\check_gpec.py'))
    if ($SourcePath) { $arguments += @('--source', $SourcePath) }
    if ($Prefix) { $arguments += @('--prefix', $Prefix) }
    Invoke-InVaft @($arguments)
    exit $LASTEXITCODE
}

if (-not $SourcePath) {
    Stop-WithGuidance @'
The path to your GPEC checkout is required.

    powershell -ExecutionPolicy Bypass -File install\install_gpec_windows.ps1 C:\git\GPEC

This script never obtains GPEC for you and never guesses where it lives.
See install\README.md.
'@
}

# ---------------------------------------------------------------------------
# Source, toolchain, prefix
# ---------------------------------------------------------------------------

$source = Assert-SourceCheckout -SourcePath $SourcePath -Project 'GPEC' `
    -ExpectedFiles @('install\makefile', 'install\DEFAULTS.inc', 'install\TARGETS.inc', 'dcon', 'gpec')
$revision = Get-SourceRevision -SourcePath $source
Write-RevisionResult -Project 'GPEC' -Revision $revision

$prefixPath = Resolve-InstallPrefix -Prefix $Prefix -CodeName $CodeName -RepositoryRoot $RepositoryRoot -SourcePath $source
$binDirectory = Join-Path $prefixPath 'bin'
New-Item -ItemType Directory -Path $binDirectory -Force | Out-Null

$root = Resolve-Toolchain -Explicit $Msys2Root -MinGWEnvironment $MinGWEnvironment -Packages $Packages -InstallToolchain:$InstallToolchain

$timestamp = (Get-Date).ToString('yyyyMMdd-HHmmss')

# ---------------------------------------------------------------------------
# netCDF and HDF5
#
# MSYS2 ships both linked against the AWS C++ S3 SDK, whose shutdown handler
# waits on a condition variable that is never signalled. A solver built
# against either writes every output, prints normal termination, and then
# never exits -- and cannot be killed. Six lines of Fortran that create and
# close a netCDF file reproduce it with no GPEC code involved.
#
# -BuildDependencies compiles both without it. Cutting S3 out of netCDF alone
# is not enough: HDF5 pulls the same SDK in through its ROS3 virtual file
# driver, so both have to be built.
# ---------------------------------------------------------------------------

$depsDirectory = Join-Path $prefixPath 'deps'

if ($BuildDependencies) {
    $dependencyLog = Join-Path $prefixPath ("logs" + [System.IO.Path]::DirectorySeparatorChar + "dependencies-$timestamp.log")
    Write-Step 'Building HDF5 and netCDF without S3 (20 to 40 minutes, once per prefix) ...'
    $steps = @(
        'set -euo pipefail',
        'deps="$(cygpath -u "$VAFT_DEPS")"',
        'work="$(cygpath -u "$VAFT_WORK")"',
        'mkdir -p "$deps" "$work"',
        'cd "$work"',
        'CF="-O2 -Wno-incompatible-pointer-types -Wno-implicit-function-declaration -Wno-int-conversion"',
        '',
        '# 1. HDF5 without the S3 virtual file driver.',
        '#    ROS3 is what drags the AWS SDK in, and the AWS SDK is what never lets',
        '#    the process exit. Float16 is off because HDF5 1.14.6 assumes a macro',
        '#    gcc 16 does not define.',
        '[ -f hdf5.tar.gz ] || curl -fsSL -o hdf5.tar.gz https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.6/hdf5-1.14.6.tar.gz',
        'rm -rf hdf5-1.14.6 && tar xf hdf5.tar.gz',
        'cd "$work/hdf5-1.14.6"',
        './configure --prefix="$deps" --disable-ros3-vfd --disable-libcurl --disable-hdfs --disable-mirror-vfd --disable-nonstandard-feature-float16 --disable-fortran --disable-cxx --disable-java --disable-tests --disable-tools --enable-hl --enable-static --disable-shared CFLAGS="$CF"',
        'make -j"$VAFT_JOBS"',
        'make install',
        '',
        '# 2. netCDF-C on that HDF5, with its own S3 and NCZarr paths off.',
        '[ -f netcdf-c.tar.gz ] || curl -fsSL -o netcdf-c.tar.gz https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.9.3.tar.gz',
        'rm -rf netcdf-c-4.9.3 && tar xf netcdf-c.tar.gz',
        'cd "$work/netcdf-c-4.9.3"',
        './configure --prefix="$deps" --disable-s3 --disable-nczarr --disable-dap --disable-byterange --disable-libxml2 --disable-plugins --disable-testsets --disable-examples --disable-utilities --enable-hdf5 --enable-static --disable-shared CFLAGS="$CF" CPPFLAGS="-I$deps/include" LDFLAGS="-L$deps/lib" LIBS="-lhdf5_hl -lhdf5 -lsz -lz"',
        'make -j"$VAFT_JOBS"',
        'make install',
        '',
        '# 3. netCDF-Fortran. The link needs the whole static chain spelled out:',
        '#    nc-config reports only -lnetcdf, and HDF5 carries an szip filter.',
        '[ -f netcdf-fortran.tar.gz ] || curl -fsSL -o netcdf-fortran.tar.gz https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.6.1.tar.gz',
        'rm -rf netcdf-fortran-4.6.1 && tar xf netcdf-fortran.tar.gz',
        'cd "$work/netcdf-fortran-4.6.1"',
        './configure --prefix="$deps" --enable-static --disable-shared CFLAGS="$CF" CPPFLAGS="-I$deps/include" LDFLAGS="-L$deps/lib" LIBS="-lnetcdf -lhdf5_hl -lhdf5 -lsz -lz -lm"',
        'make -j"$VAFT_JOBS"',
        'make install',
        '',
        '# 4. OpenBLAS is taken from the toolchain, but its import library has to sit',
        '#    beside these so that the one -L the makefile emits finds the static',
        '#    netCDF first. Without this the MinGW netCDF wins and the AWS SDK is',
        '#    back.',
        'cp -f "/$VAFT_ENV/lib/libopenblas.dll.a" "$deps/lib/"',
        ''
    )
    Invoke-Msys2 -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -Command ($steps -join "`n") `
        -Variables @{
            VAFT_DEPS = $depsDirectory
            VAFT_WORK = (Join-Path $prefixPath 'build')
            VAFT_ENV  = $MinGWEnvironment
            VAFT_JOBS = $Jobs
        } -LogPath $dependencyLog | Out-Null
    Write-Result -Status PASS -Name 'HDF5 and netCDF without S3' -Detail $depsDirectory
}

if (-not $NetcdfHome) {
    if (Test-Path -LiteralPath (Join-Path $depsDirectory 'include\netcdf.mod')) {
        $NetcdfHome = $depsDirectory
    }
    else {
        $NetcdfHome = Join-Path $root $MinGWEnvironment
        Write-Result -Status SKIP -Name 'HDF5 and netCDF without S3' -Detail 'using the MinGW build; the suite will not terminate -- see -BuildDependencies'
    }
}
$netcdfUnix = ConvertTo-Msys2Path -WindowsPath $NetcdfHome
Write-Result -Status PASS -Name 'netCDF' -Detail $NetcdfHome
# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

$logPath = Join-Path $prefixPath "logs\gpec-build-$timestamp.log"

if ($Clean) {
    # `make clean` removes the extension-less names the Linux build produces, so
    # on Windows it leaves every .exe behind, and make then treats a stale
    # binary as up to date.
    foreach ($name in $Executables) {
        foreach ($candidate in @((Join-Path $source "bin\$name.exe"), (Join-Path $source "$name\$name.exe"))) {
            if (Test-Path -LiteralPath $candidate) { Remove-Item -LiteralPath $candidate -Force }
        }
    }
}

$steps = New-Object System.Collections.Generic.List[string]
$steps.Add('set -euo pipefail')
# A stray MKLROOT or LAPACKHOME from an unrelated toolkit silently changes
# which math library the suite is linked against.
$steps.Add('unset MKLROOT LAPACKHOME LAPACK_HOME ACML_HOME NETCDFHOME NETCDF_DIR F90HOME X11_HOME')
$steps.Add('cd "$(cygpath -u "$VAFT_SRC")/install"')
$steps.Add('export FC=gfortran')
# CC must be set explicitly. GNU make predefines CC=cc, so the makefile's
# `ifndef CC` guard never fires and its compiler test then rejects `cc`.
$steps.Add('export CC=gcc')
$steps.Add('export OPENBLASHOME="$VAFT_MATH"')
$steps.Add('export NETCDF_FORTRAN_HOME="$VAFT_NETCDF"')
$steps.Add('export NETCDF_C_HOME="$VAFT_NETCDF"')
$steps.Add('export NETCDFINC="$VAFT_NETCDF/include"')
$steps.Add('export FFLAGS="-fallow-argument-mismatch -O2"')
# Empty on purpose: see the note in the description about threadprivate COMMON
# blocks and the PE object format.
$steps.Add('export OMPFLAG=')
$steps.Add('export RECURSFLAG=-frecursive')
# The static netCDF and HDF5 need their own dependencies named: the makefile
# emits only -lnetcdff -lnetcdf, and nc-config does not report the rest.
$steps.Add('export LDFLAGS="$VAFT_NCLIBS -Wl,--stack,$VAFT_STACK"')
# `make v` prints the configuration it derived. Its "Compiling supporting
# modules" line lists the dependencies it intends to build from submodules; if
# it is not empty, our library paths did not take and the next step would start
# cloning into the operator's checkout. Stop instead.
$steps.Add('make v | tee ../.vaft-make-v.txt')
$steps.Add('if grep -Eq "Compiling supporting modules[[:space:]]+[^[:space:]]" ../.vaft-make-v.txt; then echo "VAFT: GPEC wants to build its own dependencies, which would modify your checkout. Check OPENBLASHOME and NETCDF_FORTRAN_HOME." >&2; rm -f ../.vaft-make-v.txt; exit 3; fi')
$steps.Add('rm -f ../.vaft-make-v.txt')
if ($Clean) { $steps.Add('make clean || true') }
$steps.Add("make -j`$VAFT_JOBS $MakeTarget")

Write-Step 'Building the DCON/GPEC suite (10 to 30 minutes) ...'
Invoke-Msys2 -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -Command ($steps -join '; ') `
    -Variables @{
        VAFT_SRC    = $source
        VAFT_ENV    = $MinGWEnvironment
        VAFT_NETCDF = $netcdfUnix
        VAFT_MATH   = $(if ($NetcdfHome -eq $depsDirectory) { $netcdfUnix } else { '/' + $MinGWEnvironment })
        VAFT_JOBS   = $Jobs
        VAFT_STACK  = ('0x' + ('{0:X}' -f ($StackReserveMB * 1MB)))
        VAFT_NCLIBS = $(if ($NetcdfHome -eq $depsDirectory) { '-lhdf5_hl -lhdf5 -lsz -lz' } else { '' })
    } -LogPath $logPath | Out-Null

# The upstream rules judge themselves by a `cp` of an extension-less name, so
# what proves a target succeeded on Windows is the artifact, not make's word.
$sourceBin = Join-Path $source 'bin'
$missing = @()
foreach ($name in $Executables) {
    $produced = Join-Path $sourceBin "$name.exe"
    if ((Test-Path -LiteralPath $produced) -and ((Get-Item -LiteralPath $produced).Length -gt 0)) {
        Copy-Item -LiteralPath $produced -Destination (Join-Path $binDirectory "$name.exe") -Force
    }
    else {
        $missing += $name
    }
}
if ($missing.Count -gt 0) {
    Stop-WithGuidance @"
The build did not produce: $($missing -join ', ')

The full log is in $logPath. Look for the first line containing "Error:";
a compiler diagnostic in one module is an upstream portability problem, not
something this installer can work around.
"@
}
Write-Result -Status PASS -Name 'GPEC build' -Detail "$($Executables.Count) executables"

Copy-RuntimeDependencies -Msys2Root $root -MinGWEnvironment $MinGWEnvironment `
    -BinDirectory $binDirectory | Out-Null

$installed = $Executables | ForEach-Object { Join-Path $binDirectory "$_.exe" }
if (-not (Test-ExecutableLoads -Executables $installed)) {
    Stop-WithGuidance 'The installed executables could not load their runtime libraries. See install\README.md.'
}

# ---------------------------------------------------------------------------
# Record and wire up
# ---------------------------------------------------------------------------

$record = @{
    code             = 'gpec'
    prefix           = $prefixPath
    source           = $source
    source_revision  = if ($revision) { $revision.Revision } else { $null }
    source_described = if ($revision) { $revision.Described } else { $null }
    source_dirty     = if ($revision) { $revision.Dirty } else { $null }
    msys2_root       = $root
    mingw_env        = $MinGWEnvironment
    netcdf_home      = $NetcdfHome
    openmp           = $false
    dependencies_built = ($NetcdfHome -eq $depsDirectory)
    make_command     = "FC=gfortran CC=gcc OMPFLAG= make -j$Jobs $MakeTarget"
    executables      = ($Executables | ForEach-Object { "bin\$_.exe" })
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
    "powershell -ExecutionPolicy Bypass -File install\install_gpec_windows.ps1 $source -CheckOnly",
    'conda activate vaft; jupyter lab'
)

Write-Host ''
Write-Host 'This build is serial: OpenMP cannot be used on Windows, for the reason'
Write-Host 'given at the top of this script, so long runs take longer than the same'
Write-Host 'case on Linux.'
if ($NetcdfHome -ne $depsDirectory) {
    Write-Host ''
    Write-Host 'This suite is linked against the MSYS2 netCDF and HDF5, which carry the'
    Write-Host 'AWS S3 SDK: it will compute correctly and then never exit. Rerun with'
    Write-Host '-BuildDependencies to build both without it.'
}

if ($script:Failed) { exit 1 }

Write-Host ''
Write-Step 'Verifying the installation ...'
Write-Host ''
Invoke-InVaft @('python', (Join-Path $RepositoryRoot 'install\check_gpec.py'), '--source', $source, '--prefix', $prefixPath)
exit $LASTEXITCODE

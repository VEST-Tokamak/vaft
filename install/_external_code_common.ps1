<#
.SYNOPSIS
    Shared helpers for the native-Windows external-code installers.

.DESCRIPTION
    Dot-sourced by install\install_chease_windows.ps1 and
    install\install_gpec_windows.ps1. It holds what the two have in common:
    MSYS2 discovery, the opt-in toolchain install, running a build inside the
    MinGW-w64 environment, collecting an executable's runtime libraries,
    proving the result starts, and recording what was installed.

    The promises the VAFT bootstrap makes apply here too. These scripts never
    change the external source tree's branch, revision or working tree, never
    ask for or store credentials, and install nothing system-wide unless
    -InstallToolchain says so.
#>

Set-StrictMode -Version Latest

# StrictMode treats an unset automatic variable as an error, and LASTEXITCODE
# does not exist until the first native command runs.
$global:LASTEXITCODE = 0

$script:SummaryLines = New-Object System.Collections.Generic.List[string]
$script:Failed = $false
$script:Started = Get-Date

function Write-Result {
    param(
        [Parameter(Mandatory)] [ValidateSet('PASS', 'FAIL', 'SKIP')] [string] $Status,
        [Parameter(Mandatory)] [string] $Name,
        [string] $Detail = ''
    )
    $script:SummaryLines.Add("[$Status] $Name")
    if ($Status -eq 'FAIL') { $script:Failed = $true }
    if ($Detail) { Write-Host "[$Status] $Name : $Detail" } else { Write-Host "[$Status] $Name" }
}

function Write-Step {
    param([Parameter(Mandatory)] [string] $Message)
    $elapsed = (Get-Date) - $script:Started
    Write-Host ("[{0:mm\:ss}] {1}" -f $elapsed, $Message)
}

function Stop-WithGuidance {
    param([Parameter(Mandatory)] [string] $Message)
    Write-Error $Message
    exit 1
}

function Invoke-InVaft {
    <#
        Always call with one array literal: Invoke-InVaft @('python', $path).
        A bare flag at the call site binds to this function rather than to the
        command being run, because PowerShell resolves parameter names first.
    #>
    param([Parameter(Mandatory)] [string[]] $Arguments)
    & conda run --name vaft --no-capture-output @Arguments
}

function ConvertTo-Msys2Path {
    <#
        MSYS2 mounts each drive under a *lower-case* letter, so C: is /c and
        /C does not exist at all. Getting that wrong makes a glob match
        nothing rather than fail, and the caller sees an empty result with no
        error to explain it.
    #>
    param([Parameter(Mandatory)] [string] $WindowsPath)
    $forward = $WindowsPath -replace '\\', '/'
    if ($forward -match '^([A-Za-z]):(.*)$') {
        return '/' + $matches[1].ToLowerInvariant() + $matches[2]
    }
    return $forward
}

# ---------------------------------------------------------------------------
# MSYS2 discovery and the opt-in toolchain install
# ---------------------------------------------------------------------------

function Get-Msys2Candidate {
    param(
        [string] $Explicit,
        [Parameter(Mandatory)] [string] $MinGWEnvironment
    )

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($Explicit) { $candidates.Add($Explicit) }
    if ($env:MSYS2_ROOT) { $candidates.Add($env:MSYS2_ROOT) }

    # winget registers MSYS2 under a GUID, not a readable key name, so match on
    # the display name rather than guessing the key.
    $uninstallRoots = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall'
    )
    foreach ($root in $uninstallRoots) {
        foreach ($key in (Get-ChildItem -Path $root -ErrorAction SilentlyContinue)) {
            $properties = Get-ItemProperty -Path $key.PSPath -ErrorAction SilentlyContinue
            if ($null -eq $properties) { continue }
            $name = $properties.PSObject.Properties['DisplayName']
            $location = $properties.PSObject.Properties['InstallLocation']
            if ($name -and $location -and $name.Value -like 'MSYS2*' -and $location.Value) {
                $candidates.Add([string] $location.Value)
            }
        }
    }

    $candidates.Add('C:\msys64')
    if ($env:SystemDrive) { $candidates.Add((Join-Path $env:SystemDrive 'msys64')) }
    if ($env:LOCALAPPDATA) { $candidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\msys64')) }
    return $candidates
}

function Find-Msys2Root {
    <#
        Returns the MSYS2 installation root, or $null. A candidate counts only
        when it holds both the MSYS2 shell and the requested MinGW environment,
        so a half-removed install reads as absent rather than as usable.
    #>
    param(
        [string] $Explicit,
        [Parameter(Mandatory)] [string] $MinGWEnvironment
    )

    foreach ($candidate in (Get-Msys2Candidate -Explicit $Explicit -MinGWEnvironment $MinGWEnvironment)) {
        if (-not $candidate) { continue }
        $bash = Join-Path $candidate 'usr\bin\bash.exe'
        $environmentRoot = Join-Path $candidate $MinGWEnvironment
        if ((Test-Path -LiteralPath $bash) -and (Test-Path -LiteralPath $environmentRoot)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Get-Msys2PackagePrefix {
    param([Parameter(Mandatory)] [string] $MinGWEnvironment)
    if ($MinGWEnvironment -eq 'ucrt64') { return 'mingw-w64-ucrt-x86_64-' }
    return 'mingw-w64-x86_64-'
}

function Get-ToolchainGuidance {
    param(
        [Parameter(Mandatory)] [string] $MinGWEnvironment,
        [Parameter(Mandatory)] [string[]] $Packages,
        [string[]] $Searched = @()
    )
    $packageLine = ($Packages -join ' ')
    $searchedText = ''
    if ($Searched.Count -gt 0) {
        $searchedText = "

Looked for MSYS2 in:
  " + ($Searched -join "
  ")
    }
    return @"
MSYS2 with a MinGW-w64 Fortran toolchain was not found.

VAFT does not install compilers for you unless you ask. Either rerun this
script with -InstallToolchain, or set the toolchain up yourself once:

    winget install --id MSYS2.MSYS2 --exact --accept-package-agreements --accept-source-agreements

then open "MSYS2 $($MinGWEnvironment.ToUpper())" from the Start menu and run:

    pacman -Syu --noconfirm
    pacman -S --needed --noconfirm $packageLine

If MSYS2 is installed somewhere this script did not look, pass -Msys2Root.$searchedText
"@
}

function Invoke-Msys2 {
    <#
        Runs one command inside the MinGW-w64 login shell and returns its exit
        code.

        bash.exe rather than msys2_shell.cmd: one process instead of a cmd.exe
        wrapper, and a reliable exit code. The script itself is written to the
        shell's stdin -- see the note at the call below.

        MSYS2_PATH_TYPE is deliberately left at its default rather than set to
        inherit. Inheriting the Windows PATH puts Anaconda's Library\bin --
        with its own zlib, libssl and sometimes make -- ahead of MSYS2's, and
        the resulting failures look like compiler bugs.

        Windows paths cross the boundary through the environment and are
        converted with cygpath inside the shell, never interpolated into the
        command string, so drive letters, separators and spaces never need
        quoting.
    #>
    param(
        [Parameter(Mandatory)] [string] $Msys2Root,
        [Parameter(Mandatory)] [string] $MinGWEnvironment,
        [Parameter(Mandatory)] [string] $Command,
        [hashtable] $Variables = @{},
        [string] $LogPath,
        [switch] $AllowFailure
    )

    $bash = Join-Path $Msys2Root 'usr\bin\bash.exe'
    if (-not (Test-Path -LiteralPath $bash)) {
        Stop-WithGuidance "MSYS2 shell not found at $bash."
    }

    $assignments = @{
        MSYSTEM        = $MinGWEnvironment.ToUpper()
        CHERE_INVOKING = '1'
    }
    foreach ($key in $Variables.Keys) { $assignments[$key] = [string] $Variables[$key] }

    $saved = @{}
    foreach ($key in @($assignments.Keys)) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $assignments[$key], 'Process')
    }

    # A native command writing to stderr is a terminating error while
    # $ErrorActionPreference is Stop. A compiler warning is not a reason to
    # abort an install; the exit code is.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $code = 0
    try {
        # The script goes in on stdin rather than as a -c argument. Windows
        # PowerShell rewrites embedded double quotes when it builds a native
        # command line, so a shell command containing `cygpath -u "$VAR"`
        # arrives at bash as a syntax error. Nothing is quoted on the way in
        # this way, so nothing can be mangled.
        if ($LogPath) {
            $directory = Split-Path -Parent $LogPath
            if ($directory -and -not (Test-Path -LiteralPath $directory)) {
                New-Item -ItemType Directory -Path $directory -Force | Out-Null
            }
            # Tee-Object passes what it writes on down the pipeline, so it needs a
            # terminating Write-Host: without one the caller's Out-Null swallows
            # every line of a build that runs for minutes.
            $Command | & $bash -l -s 2>&1 | Tee-Object -FilePath $LogPath -Append | Write-Host
        }
        else {
            $Command | & $bash -l -s 2>&1 | Write-Host
        }
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
        foreach ($key in $saved.Keys) {
            [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process')
        }
    }

    if ($code -ne 0 -and -not $AllowFailure) {
        $where = ''
        if ($LogPath) { $where = " The full output is in $LogPath." }
        Stop-WithGuidance ("The MSYS2 command failed with exit code " + $code + "." + $where)
    }
    return $code
}

function Install-Msys2Toolchain {
    <#
        The only path in these scripts that installs anything system-wide, and
        it runs only when the caller passes -InstallToolchain.
    #>
    param(
        [Parameter(Mandatory)] [string] $MinGWEnvironment,
        [Parameter(Mandatory)] [string[]] $Packages,
        [string] $Explicit
    )

    $root = Find-Msys2Root -Explicit $Explicit -MinGWEnvironment $MinGWEnvironment
    if (-not $root) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            Stop-WithGuidance @'
-InstallToolchain needs winget, which is not on PATH.

Install MSYS2 by hand from https://www.msys2.org/ and rerun this script, or
pass -Msys2Root if it is already installed somewhere unusual.
'@
        }
        Write-Step 'Installing MSYS2 with winget ...'
        & winget install --id MSYS2.MSYS2 --exact --accept-package-agreements --accept-source-agreements --disable-interactivity
        # winget does not refresh this process's environment, so discovery has
        # to run again rather than reuse anything from before.
        $root = Find-Msys2Root -Explicit $Explicit -MinGWEnvironment $MinGWEnvironment
        if (-not $root) {
            Stop-WithGuidance 'MSYS2 was installed but could not be located afterwards. Pass -Msys2Root explicitly.'
        }
        Write-Result -Status PASS -Name 'MSYS2 installed' -Detail $root
    }

    Write-Step 'Updating MSYS2 packages (roughly 1 GB the first time) ...'
    # A first core update on a fresh install ends by terminating its own shell
    # by design, so a non-zero status here is expected rather than fatal.
    Invoke-Msys2 -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -Command 'pacman -Syu --noconfirm' -AllowFailure | Out-Null
    Invoke-Msys2 -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -Command 'pacman -Syu --noconfirm' -AllowFailure | Out-Null

    Write-Step 'Installing the MinGW-w64 toolchain ...'
    $install = 'pacman -S --needed --noconfirm ' + ($Packages -join ' ')
    Invoke-Msys2 -Msys2Root $root -MinGWEnvironment $MinGWEnvironment -Command $install | Out-Null
    Write-Result -Status PASS -Name 'MinGW-w64 toolchain' -Detail ($Packages -join ', ')
    return $root
}

function Resolve-Toolchain {
    <#
        Finds MSYS2, or installs it when the caller opted in, or stops with
        instructions naming both ways to fix it.
    #>
    param(
        [string] $Explicit,
        [Parameter(Mandatory)] [string] $MinGWEnvironment,
        [Parameter(Mandatory)] [string[]] $Packages,
        [switch] $InstallToolchain
    )

    if ($InstallToolchain) {
        return Install-Msys2Toolchain -MinGWEnvironment $MinGWEnvironment -Packages $Packages -Explicit $Explicit
    }
    $root = Find-Msys2Root -Explicit $Explicit -MinGWEnvironment $MinGWEnvironment
    if (-not $root) {
        $searched = Get-Msys2Candidate -Explicit $Explicit -MinGWEnvironment $MinGWEnvironment
        Stop-WithGuidance (Get-ToolchainGuidance -MinGWEnvironment $MinGWEnvironment -Packages $Packages -Searched $searched)
    }
    Write-Result -Status PASS -Name 'MSYS2' -Detail "$root ($MinGWEnvironment)"
    return $root
}

# ---------------------------------------------------------------------------
# Source validation -- read only, always
# ---------------------------------------------------------------------------

function Assert-SourceCheckout {
    <#
        Confirms the caller pointed at a plausible checkout of the expected
        project. The path is a required argument on both installers and is
        never guessed: with several checkouts on one machine, provenance has to
        be something the operator states rather than something a script infers.
    #>
    param(
        [Parameter(Mandatory)] [string] $SourcePath,
        [Parameter(Mandatory)] [string] $Project,
        [Parameter(Mandatory)] [string[]] $ExpectedFiles
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) {
        Stop-WithGuidance @"
$Project source path does not exist: $SourcePath

You obtain the external code yourself; this installer never does it for you.
See install\README.md for the command.
"@
    }
    $resolved = (Resolve-Path -LiteralPath $SourcePath).Path

    $missing = @()
    foreach ($relative in $ExpectedFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $resolved $relative))) { $missing += $relative }
    }
    if ($missing.Count -gt 0) {
        Stop-WithGuidance @"
$resolved does not look like a $Project checkout.

Expected to find: $($missing -join ', ')

Point -SourcePath at the top of your $Project tree.
"@
    }
    Write-Result -Status PASS -Name "$Project source path" -Detail $resolved
    return $resolved
}

function Get-SourceRevision {
    <#
        Reports the revision for provenance. Read-only by construction: it only
        asks Git what the tree already is.
    #>
    param([Parameter(Mandatory)] [string] $SourcePath)

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { return $null }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $revision = (& git -C $SourcePath rev-parse --short HEAD 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $revision) { return $null }
        $described = (& git -C $SourcePath describe --tags --always 2>$null | Out-String).Trim()
        # Untracked files excluded: a build writes its own products into the
        # source tree, and those say nothing about the revision compiled.
        $dirty = (& git -C $SourcePath status --porcelain --untracked-files=no 2>$null | Out-String).Trim()
        return [pscustomobject]@{
            Revision  = $revision
            Described = $described
            Dirty     = [bool] $dirty
        }
    }
    finally { $ErrorActionPreference = $previous }
}

function Write-RevisionResult {
    param(
        [Parameter(Mandatory)] [string] $Project,
        $Revision
    )
    if ($null -eq $Revision) {
        Write-Result -Status SKIP -Name "$Project revision" -Detail 'not a Git checkout, so provenance cannot be recorded'
        return
    }
    $detail = $Revision.Revision
    if ($Revision.Described -and $Revision.Described -ne $Revision.Revision) {
        $detail = "$($Revision.Described) ($($Revision.Revision))"
    }
    if ($Revision.Dirty) { $detail += ' [uncommitted changes present]' }
    Write-Result -Status PASS -Name "$Project revision" -Detail $detail
}

# ---------------------------------------------------------------------------
# Install prefix
# ---------------------------------------------------------------------------

function Resolve-InstallPrefix {
    param(
        [string] $Prefix,
        [Parameter(Mandatory)] [string] $CodeName,
        [Parameter(Mandatory)] [string] $RepositoryRoot,
        [Parameter(Mandatory)] [string] $SourcePath
    )

    if (-not $Prefix) {
        $Prefix = Join-Path $env:LOCALAPPDATA "vaft\external\$CodeName"
    }
    if (-not (Test-Path -LiteralPath $Prefix)) {
        New-Item -ItemType Directory -Path $Prefix -Force | Out-Null
    }
    $resolved = (Resolve-Path -LiteralPath $Prefix).Path

    # Anything inside the VAFT checkout would survive as an untracked file and
    # fail the bootstrap's own "the checkout is left clean" check.
    foreach ($forbidden in @($RepositoryRoot, $SourcePath)) {
        $normalized = (Resolve-Path -LiteralPath $forbidden).Path
        $separator = [System.IO.Path]::DirectorySeparatorChar
        if ($resolved -eq $normalized -or $resolved.StartsWith($normalized + $separator)) {
            Stop-WithGuidance "The install prefix must be outside $normalized. Pass a different -Prefix."
        }
    }
    return $resolved
}

# ---------------------------------------------------------------------------
# Runtime libraries and the load probe
# ---------------------------------------------------------------------------

function Copy-RuntimeDependencies {
    <#
        Copies the MinGW-w64 libraries the installed executables need into the
        same directory as the executables.

        Windows searches the directory holding the running .exe before anything
        on PATH, so this makes the prefix self-contained for every launcher --
        a plain terminal, conda run, or a Jupyter kernel started from a
        different environment -- with no PATH change anywhere.

        Prepending MSYS2's bin directory to PATH is the alternative and is
        deliberately not done: it puts a second libcrypto, libssl and zlib
        ahead of Anaconda's for every process in the session, which breaks
        unrelated things in ways that are very hard to trace back to here.
    #>
    param(
        [Parameter(Mandatory)] [string] $Msys2Root,
        [Parameter(Mandatory)] [string] $MinGWEnvironment,
        [Parameter(Mandatory)] [string] $BinDirectory
    )

    # Walk the import tables to a closure, statically.
    #
    # objdump reads the PE header; ldd runs the program to find out. That
    # difference decides the job: the moment this is most needed is when the
    # executable cannot start yet, which is exactly when ldd has nothing to
    # say. The walk is breadth-first because the interesting libraries are
    # not direct imports -- an executable names netCDF, and netCDF brings
    # HDF5, curl and the rest in behind it.
    #
    # Both directories cross into the shell through the environment and are
    # converted with cygpath there, for the same reason the build steps do
    # it: a path computed on this side can disagree with the one the shell
    # sees, and a glob that matches nothing fails silently.
    $toolchainBin = Join-Path $Msys2Root ($MinGWEnvironment + '\bin')
    $command = @'
set -u
bin="$(cygpath -u "$VAFT_BIN")"
tc="$(cygpath -u "$VAFT_TOOLCHAIN")"
queue=$(ls "$bin"/*.exe 2>/dev/null)
seen=""
while [ -n "$queue" ]; do
  next=""
  for f in $queue; do
    for d in $(objdump -p "$f" 2>/dev/null | sed -n "s/^[[:space:]]*DLL Name:[[:space:]]*//p"); do
      case " $seen " in *" $d "*) continue;; esac
      seen="$seen $d"
      if [ -f "$tc/$d" ]; then echo "$d"; next="$next $tc/$d"; fi
    done
  done
  queue="$next"
done
'@

    $bash = Join-Path $Msys2Root 'usr\bin\bash.exe'
    $assignments = @{
        MSYSTEM        = $MinGWEnvironment.ToUpper()
        VAFT_BIN       = $BinDirectory
        VAFT_TOOLCHAIN = $toolchainBin
    }
    $saved = @{}
    foreach ($key in @($assignments.Keys)) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $assignments[$key], 'Process')
    }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = $command | & $bash -l -s 2>$null
    }
    finally {
        $ErrorActionPreference = $previous
        foreach ($key in $saved.Keys) {
            [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process')
        }
    }

    # Every executable gfortran produces needs these, and a missing-library
    # failure is nearly always about one of them. They are copied whether or
    # not the walk above returned anything, so a toolchain that cannot be
    # inspected still yields a working install rather than one that reports
    # nothing was needed and then fails its own load probe.
    $baseline = @(
        'libgcc_s_seh-1.dll',
        'libwinpthread-1.dll',
        'libgfortran-5.dll',
        'libquadmath-0.dll',
        'libstdc++-6.dll',
        'libgomp-1.dll'
    )
    $wanted = New-Object System.Collections.Generic.List[string]
    foreach ($name in $baseline) { $wanted.Add($name) }
    foreach ($line in @($lines)) {
        $name = ([string] $line).Trim()
        if ($name -match '^[A-Za-z0-9_.+-]+[.]dll$' -and -not $wanted.Contains($name)) {
            $wanted.Add($name)
        }
    }

    $copied = 0
    $present = 0
    foreach ($name in $wanted) {
        # Only what the toolchain owns. A system library -- kernel32,
        # ucrtbase -- is not ours to ship.
        $source = Join-Path $toolchainBin $name
        if (-not (Test-Path -LiteralPath $source)) { continue }
        $destination = Join-Path $BinDirectory $name
        if (Test-Path -LiteralPath $destination) { $present++; continue }
        Copy-Item -LiteralPath $source -Destination $destination -Force
        $copied++
    }

    if ($copied -eq 0 -and $present -eq 0) {
        Write-Result -Status FAIL -Name 'Runtime libraries' -Detail (
            'none could be copied from ' + $toolchainBin)
        return 0
    }
    Write-Result -Status PASS -Name 'Runtime libraries' -Detail (
        "$copied copied next to the executables, $present already there")
    return $copied + $present
}


function Test-ExecutableLoads {
    <#
        Starts each executable with PATH stripped to the system directory. A
        process that starts and then exits for its own reasons -- usually
        "no input file" -- has proved its runtime libraries resolve, which is
        the most common way a successful build is still unusable.
    #>
    param(
        [Parameter(Mandatory)] [string[]] $Executables,
        [int] $TimeoutSeconds = 30
    )

    $scratch = Join-Path ([System.IO.Path]::GetTempPath()) ('vaft-loadprobe-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $scratch -Force | Out-Null
    $system32 = Join-Path $env:SystemRoot 'System32'
    $ok = $true
    try {
        foreach ($executable in $Executables) {
            $label = 'Loads: ' + (Split-Path -Leaf $executable)
            $info = New-Object System.Diagnostics.ProcessStartInfo
            $info.FileName = $executable
            $info.WorkingDirectory = $scratch
            $info.UseShellExecute = $false
            $info.RedirectStandardOutput = $true
            $info.RedirectStandardError = $true
            $info.EnvironmentVariables['PATH'] = $system32
            $process = [System.Diagnostics.Process]::Start($info)
            # Drain both pipes while it runs. A solver that greets the console
            # with more than one buffer's worth would otherwise block on a full
            # pipe and be reported as a timeout rather than as loading cleanly.
            $null = $process.StandardOutput.ReadToEndAsync()
            $null = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
                try { $process.Kill() } catch { }
                # Not a load failure: it started and kept running.
                Write-Result -Status PASS -Name $label -Detail 'started'
                continue
            }
            $code = $process.ExitCode
            if ($code -eq -1073741515 -or $code -eq -1073741701) {
                $hex = '{0:X8}' -f $code
                Write-Result -Status FAIL -Name $label -Detail "missing runtime libraries (exit 0x$hex)"
                $ok = $false
            }
            else {
                Write-Result -Status PASS -Name $label -Detail "started (exit $code)"
            }
        }
    }
    finally {
        Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
    }
    return $ok
}

# ---------------------------------------------------------------------------
# Manifest and environment wiring
# ---------------------------------------------------------------------------

function Get-ManifestPath {
    param([Parameter(Mandatory)] [string] $Prefix)
    return (Join-Path $Prefix 'vaft-external-install.json')
}

function Write-InstallManifest {
    param(
        [Parameter(Mandatory)] [string] $Prefix,
        [Parameter(Mandatory)] [hashtable] $Record
    )
    $Record['written'] = (Get-Date).ToString('o')
    $path = Get-ManifestPath -Prefix $Prefix
    $Record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding utf8
    Write-Result -Status PASS -Name 'Install manifest' -Detail $path
}

function Set-ExternalCodeEnvironment {
    <#
        Writes the installation root as a *user* environment variable.

        The registered "Python (vaft)" kernel launches the environment's
        python.exe directly rather than through Conda activation, so an
        activate.d script or `conda env config vars` would never reach a
        notebook. A user variable reaches every newly started process --
        terminal, JupyterLab, editor -- which is what the workflow needs.
        Nothing machine-wide is touched.
    #>
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [string] $Value
    )
    [Environment]::SetEnvironmentVariable($Name, $Value, 'User')
    Set-Item -Path ('Env:' + $Name) -Value $Value
    Write-Result -Status PASS -Name "$Name (user environment)" -Detail $Value
}

function Remove-ExternalCodeEnvironment {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [string] $ExpectedValue
    )
    $current = [Environment]::GetEnvironmentVariable($Name, 'User')
    if (-not $current) {
        Write-Result -Status SKIP -Name "$Name (user environment)" -Detail 'not set'
        return
    }
    if ($current -ne $ExpectedValue) {
        Write-Result -Status SKIP -Name "$Name (user environment)" -Detail "points at $current, which this script did not install"
        return
    }
    [Environment]::SetEnvironmentVariable($Name, $null, 'User')
    Write-Result -Status PASS -Name "$Name (user environment)" -Detail 'removed'
}

function Write-ExternalSummary {
    param(
        [Parameter(Mandatory)] [string] $Title,
        [string[]] $NextSteps = @()
    )
    Write-Host ''
    Write-Host "$Title : what changed"
    Write-Host '-------------------------------------------------------------'
    foreach ($line in $script:SummaryLines) { Write-Host "  $line" }
    Write-Host ''
    Write-Host 'Left alone: your source tree''s branch and revision, Conda, MSYS2'
    Write-Host 'itself, and any environment variable this script did not set.'
    if ($NextSteps.Count -gt 0) {
        Write-Host ''
        Write-Host 'NEXT'
        foreach ($line in $NextSteps) { Write-Host "  $line" }
    }
}

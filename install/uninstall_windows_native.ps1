<#
.SYNOPSIS
    VAFT uninstall for native Windows.

.DESCRIPTION
    Reverses install\windows_native.ps1 and nothing else: removes the user-level
    "Python (vaft)" Jupyter kernelspec, the `vaft` Conda environment (and with it
    the editable VAFT installation), and the gitignored build artifacts an
    editable install leaves in the checkout.

    It never removes `~\.hscfg`. That file holds your HSDS credentials and the
    bootstrap never created it -- `hsconfigure` did, run by you.

    It never touches a Conda environment whose name is not exactly `vaft`, your
    repository checkout, or Conda and Git themselves. Like the bootstrap, it
    runs no destructive Git command.

    Running it twice is a no-op: with nothing installed every step reports SKIP
    and the script exits 0.

.PARAMETER Yes
    Skip the confirmation prompt. Intended for CI and scripting.

.PARAMETER DryRun
    Print what would be removed and change nothing.

.PARAMETER KeepBuildArtifacts
    Leave vaft.egg-info\ in the checkout.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\uninstall_windows_native.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\uninstall_windows_native.ps1 -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\uninstall_windows_native.ps1 -Yes
#>
[CmdletBinding()]
param(
    [switch] $Yes,
    [switch] $DryRun,
    [switch] $KeepBuildArtifacts
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$EnvironmentName = 'vaft'
$KernelName = 'vaft'
$KernelDisplayName = 'Python (vaft)'
$PlatformLabel = 'Windows (native)'
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

# What the editable install leaves in the checkout, and the only thing here a
# reinstall would inherit. Deliberately not `build\` or `dist\`: the bootstrap
# never creates those -- `python -m build` does, per RELEASING.md -- and
# deleting a maintainer's release artifacts is not this script's business.
#
# Explicit names, never `git clean`: the tooling in install\ promises it runs
# no destructive Git command, and that promise still applies here.
$BuildArtifacts = @('vaft.egg-info')

$script:SummaryLines = New-Object System.Collections.Generic.List[string]
$script:Failed = $false

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

function Stop-WithGuidance {
    param([Parameter(Mandatory)] [string] $Message)
    Write-Error $Message
    exit 1
}

function Assert-Conda {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        Stop-WithGuidance @'
Conda was not found on PATH.

Reopen PowerShell, use the "Anaconda PowerShell Prompt", or run
`conda init powershell` once and reopen PowerShell, then rerun this script.
'@
    }
    Write-Result -Status PASS -Name 'Conda' -Detail (conda --version)
}

function Test-VaftEnvironment {
    # Parsed from the plain listing so the check never needs a second interpreter.
    $names = @(conda env list | ForEach-Object { ($_ -split '\s+')[0] })
    return $names -contains $EnvironmentName
}

function Invoke-InVaft {
    # Run a command inside the `vaft` environment without mutating this shell.
    #
    # Always call this with a single array literal: PowerShell would otherwise
    # try to bind a bare `-f`-style argument to a parameter of this function.
    param([Parameter(Mandatory)] [string[]] $Arguments)
    & conda run --name $EnvironmentName --no-capture-output @Arguments
}

function Get-KernelspecDirectory {
    # Where Jupyter keeps user-level kernelspecs, without needing Jupyter to
    # tell us -- once the environment is gone there is no interpreter to ask.
    # Only a fallback sweep; the supported path is `jupyter kernelspec remove`.
    $roots = New-Object System.Collections.Generic.List[string]
    if ($env:JUPYTER_DATA_DIR) { $roots.Add((Join-Path $env:JUPYTER_DATA_DIR 'kernels')) }
    if ($env:APPDATA) { $roots.Add((Join-Path $env:APPDATA 'jupyter\kernels')) }
    return $roots
}

function Test-VaftKernel {
    foreach ($root in Get-KernelspecDirectory) {
        if (Test-Path -LiteralPath (Join-Path $root $KernelName)) { return $true }
    }
    return $false
}

function Assert-EnvironmentIsNotActive {
    # Removal is ordered kernel-first, because the kernelspec can only be
    # removed through the environment's own interpreter. That ordering means a
    # `conda env remove` which refuses part-way would leave a working
    # environment with no kernel -- strictly worse than not having started.
    #
    # Conda refuses to remove the environment the current shell has activated,
    # and that is the one case we can see coming. Stop before touching anything.
    $active = $env:CONDA_DEFAULT_ENV
    if (-not $active -and $env:CONDA_PREFIX) {
        $active = Split-Path -Leaf $env:CONDA_PREFIX
    }
    if ($active -eq $EnvironmentName) {
        Stop-WithGuidance @"
The '$EnvironmentName' environment is active in this shell, and Conda refuses to
remove an environment you are standing in.

Run ``conda deactivate`` first, then rerun this script. Nothing has been removed.
"@
    }
}

function Remove-VaftKernel {
    $removed = $false

    # The supported route, while there is still an interpreter to run it. A
    # broken or Jupyter-less environment must not abort the uninstall, so the
    # failure is tolerated and the directory sweep below picks up the slack.
    if (Test-VaftEnvironment) {
        try {
            Invoke-InVaft @('python', '-m', 'jupyter', 'kernelspec', 'remove', '-f', $KernelName) | Out-Null
            if ($LASTEXITCODE -eq 0) { $removed = $true }
        }
        catch {
            # Windows PowerShell turns native stderr into a terminating error
            # when $ErrorActionPreference is Stop. Nothing here is worth
            # aborting the uninstall for -- the sweep below is the fallback.
        }
    }

    foreach ($root in Get-KernelspecDirectory) {
        $spec = Join-Path $root $KernelName
        if (Test-Path -LiteralPath $spec) {
            Remove-Item -LiteralPath $spec -Recurse -Force
            $removed = $true
        }
    }

    if ($removed) {
        Write-Result -Status PASS -Name "$KernelDisplayName kernel" -Detail 'removed'
    }
    else {
        Write-Result -Status SKIP -Name "$KernelDisplayName kernel" -Detail 'not registered'
    }
}

function Remove-VaftEnvironment {
    if (-not (Test-VaftEnvironment)) {
        Write-Result -Status SKIP -Name 'vaft environment' -Detail 'not present'
        return
    }
    Write-Host "Removing the '$EnvironmentName' environment ..."
    # --name pins the removal to the exact name. No prefix or pattern match, so
    # environments like `vaft-np2-test` are never in scope.
    conda env remove --name $EnvironmentName --yes
    if ($LASTEXITCODE -ne 0) {
        Write-Result -Status FAIL -Name 'vaft environment' -Detail 'conda env remove failed'
        return
    }
    Write-Result -Status PASS -Name 'vaft environment' -Detail 'removed'
}

function Get-PresentBuildArtifact {
    $present = New-Object System.Collections.Generic.List[string]
    foreach ($artifact in $BuildArtifacts) {
        if (Test-Path -LiteralPath (Join-Path $RepositoryRoot $artifact)) { $present.Add($artifact) }
    }
    return $present
}

function Remove-BuildArtifact {
    if ($KeepBuildArtifacts) {
        Write-Result -Status SKIP -Name 'build artifacts' -Detail 'kept (-KeepBuildArtifacts)'
        return
    }
    $present = @(Get-PresentBuildArtifact)
    if ($present.Count -eq 0) {
        Write-Result -Status SKIP -Name 'build artifacts' -Detail 'none in the checkout'
        return
    }
    foreach ($artifact in $present) {
        Remove-Item -LiteralPath (Join-Path $RepositoryRoot $artifact) -Recurse -Force
    }
    Write-Result -Status PASS -Name 'build artifacts' -Detail ("removed " + ($present -join ' '))
}

function Write-RemovalPlan {
    Write-Host 'This will remove:'
    if (Test-VaftEnvironment) {
        Write-Host "  - the '$EnvironmentName' Conda environment, and the editable VAFT installation in it"
    }
    else {
        Write-Host "  - (the '$EnvironmentName' Conda environment is already absent)"
    }
    if (Test-VaftKernel) {
        Write-Host "  - the user-level `"$KernelDisplayName`" Jupyter kernelspec"
    }
    else {
        Write-Host "  - (the `"$KernelDisplayName`" kernelspec is already absent)"
    }
    if ($KeepBuildArtifacts) {
        Write-Host '  - (build artifacts kept: -KeepBuildArtifacts)'
    }
    else {
        $present = @(Get-PresentBuildArtifact)
        if ($present.Count -gt 0) {
            Write-Host ("  - build artifacts in " + $RepositoryRoot + ": " + ($present -join ' '))
        }
        else {
            Write-Host ("  - (no build artifacts in " + $RepositoryRoot + ")")
        }
    }
    Write-Host @'

It will not touch: your repository checkout, any other Conda environment, or
`~\.hscfg` -- that file holds your HSDS credentials and this script never
created it.
'@
}

function Confirm-Removal {
    if ($Yes) { return $true }
    if (-not [Environment]::UserInteractive) {
        Stop-WithGuidance @'
Refusing to remove anything without confirmation.

This session is not interactive, so the script cannot ask. Rerun with -Yes to
confirm non-interactively, or -DryRun to see what would be removed.
'@
    }
    Write-Host ''
    $reply = Read-Host 'Remove these? [y/N]'
    return @('y', 'Y', 'yes', 'YES', 'Yes') -contains $reply
}

function Write-Summary {
    Write-Host ''
    Write-Host "VAFT uninstall ($PlatformLabel): what changed"
    Write-Host '--------------'
    $script:SummaryLines | ForEach-Object { Write-Host $_ }
    Write-Host @'

Your checkout, every other Conda environment and `~\.hscfg` are untouched.
Reinstall at any time with install\windows_native.ps1; it will create the
environment from scratch, exactly as it did the first time.
'@
}

Write-Host "VAFT uninstall ($PlatformLabel)"
Write-Host "Repository: $RepositoryRoot"
Write-Host ''

Assert-Conda
Assert-EnvironmentIsNotActive
Write-RemovalPlan

if ($DryRun) {
    Write-Host ''
    Write-Host 'Dry run: nothing was removed.'
    exit 0
}

if (-not (Confirm-Removal)) {
    Write-Host 'Aborted. Nothing was removed.'
    exit 0
}

Write-Host ''
# Kernel first: removing it goes through the environment's own interpreter.
Remove-VaftKernel
Remove-VaftEnvironment
Remove-BuildArtifact
Write-Summary

if ($script:Failed) { exit 1 }
exit 0

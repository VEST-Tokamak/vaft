<#
.SYNOPSIS
    VAFT bootstrap for native Windows.

.DESCRIPTION
    Creates or updates the `vaft` Conda environment, installs VAFT in editable
    mode from this checkout, and registers the "Python (vaft)" Jupyter kernel.

    VAFT is fully supported on native Windows; WSL2 is never required.

    The script is idempotent and strictly non-destructive. It never runs
    `git stash`, `git reset`, `git clean` or `git checkout`, never asks for or
    stores credentials, and never modifies a Conda environment other than
    `vaft`.

    Prerequisites (install these yourself first): Git for Windows and Miniconda.

.PARAMETER CheckOnly
    Run install\check_vaft_environment.py and change nothing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\windows_native.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\windows_native.ps1 -CheckOnly
#>
[CmdletBinding()]
param(
    [switch] $CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$EnvironmentName = 'vaft'
$KernelName = 'vaft'
$KernelDisplayName = 'Python (vaft)'
$PlatformLabel = 'Windows (native)'
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

$script:SummaryLines = New-Object System.Collections.Generic.List[string]
$script:Failed = $false

function Write-Result {
    param(
        [Parameter(Mandatory)] [ValidateSet('PASS', 'FAIL')] [string] $Status,
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

Install Miniconda first, then reopen PowerShell and rerun this script. If
Miniconda is already installed, use the "Anaconda PowerShell Prompt", or run
`conda init powershell` once and reopen PowerShell.

See install\README.md for the download link. This script deliberately does not
install Conda for you.
'@
    }
    Write-Result -Status PASS -Name 'Conda' -Detail (conda --version)
}

function Test-VaftEnvironment {
    # Parsed from the plain listing so the check never needs a second interpreter.
    $names = @(conda env list | ForEach-Object { ($_ -split '\s+')[0] })
    return $names -contains $EnvironmentName
}

function Initialize-VaftEnvironment {
    $specification = Join-Path $RepositoryRoot 'environment.yml'
    if (-not (Test-Path -LiteralPath $specification)) {
        Stop-WithGuidance "Missing $specification."
    }
    if (Test-VaftEnvironment) {
        Write-Host "Updating the existing '$EnvironmentName' environment ..."
        # Deliberately not --prune: it would remove packages a student
        # installed into this environment themselves.
        conda env update --name $EnvironmentName --file $specification
        if ($LASTEXITCODE -ne 0) { Stop-WithGuidance 'conda env update failed.' }
        Write-Result -Status PASS -Name 'vaft environment' -Detail 'updated in place'
    }
    else {
        Write-Host "Creating the '$EnvironmentName' environment ..."
        conda env create --name $EnvironmentName --file $specification
        if ($LASTEXITCODE -ne 0) { Stop-WithGuidance 'conda env create failed.' }
        Write-Result -Status PASS -Name 'vaft environment' -Detail 'created'
    }
}

function Invoke-InVaft {
    # Run a command inside the `vaft` environment without mutating this shell.
    #
    # Always call this with a single array literal, never as
    # `Invoke-InVaft python -m pip install -e .`: PowerShell would try to bind
    # the bare `-e` to a parameter of this function and fail before conda ever
    # sees it.
    param([Parameter(Mandatory)] [string[]] $Arguments)
    & conda run --name $EnvironmentName --no-capture-output @Arguments
}

function Write-PythonReport {
    $description = Invoke-InVaft @('python', '-c', 'import platform, sys; print(platform.python_version(), sys.executable)')
    Write-Result -Status PASS -Name 'Python' -Detail $description
}

function Install-VaftEditable {
    Write-Host "Installing VAFT in editable mode from $RepositoryRoot ..."
    Push-Location $RepositoryRoot
    try {
        Invoke-InVaft @('python', '-m', 'pip', 'install', '-e', '.')
        if ($LASTEXITCODE -ne 0) { Stop-WithGuidance 'Editable installation failed.' }
    }
    finally {
        Pop-Location
    }
    Write-Result -Status PASS -Name 'editable VAFT installation' -Detail $RepositoryRoot
}

function Register-VaftKernel {
    # `--name vaft` overwrites any existing spec of the same name, so repeated
    # runs replace the kernel in place instead of accumulating duplicates.
    # Whether exactly one survives is then confirmed by the checker.
    Invoke-InVaft @(
        'python', '-m', 'ipykernel', 'install', '--user',
        '--name', $KernelName, '--display-name', $KernelDisplayName
    ) | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Result -Status FAIL -Name "$KernelDisplayName kernel" -Detail 'ipykernel install failed'
        return
    }
    Write-Result -Status PASS -Name "$KernelDisplayName kernel" -Detail 'registered'
}

function Write-Summary {
    Write-Host ''
    Write-Host "VAFT bootstrap ($PlatformLabel): what changed"
    Write-Host '--------------'
    $script:SummaryLines | ForEach-Object { Write-Host $_ }
    Write-Host @'

This script changed only: the `vaft` Conda environment, an editable VAFT
installation inside it, and the user-level "Python (vaft)" Jupyter kernelspec.
It did not modify your repository checkout or any other Conda environment.

Next:
  1. Run `hsconfigure` if your HSDS credentials are not configured yet.
     This script never asks for, stores, or transmits your credentials.
  2. Run `conda activate vaft; jupyter lab`, and choose the "Python (vaft)" kernel.
'@
}

Write-Host "VAFT bootstrap ($PlatformLabel)"
Write-Host "Repository: $RepositoryRoot"
Write-Host ''

Assert-Conda

if ($CheckOnly) {
    if (-not (Test-VaftEnvironment)) {
        Stop-WithGuidance @"
The '$EnvironmentName' environment does not exist yet.
Run this script without -CheckOnly to create it.
"@
    }
    Invoke-InVaft @('python', (Join-Path $RepositoryRoot 'install\check_vaft_environment.py'))
    exit $LASTEXITCODE
}

Initialize-VaftEnvironment
Write-PythonReport
Install-VaftEditable
Register-VaftKernel
Write-Summary

if ($script:Failed) { exit 1 }

# Verification lives in one place. The checker reports every environment
# property with its own remediation, so the bootstrap does not reimplement those
# probes -- and its exit status becomes the bootstrap's.
Write-Host ''
Write-Host 'Verifying the environment ...'
Write-Host ''
Invoke-InVaft @('python', (Join-Path $RepositoryRoot 'install\check_vaft_environment.py'))
exit $LASTEXITCODE

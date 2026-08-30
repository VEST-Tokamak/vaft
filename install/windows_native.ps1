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
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

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
    $names = conda env list | ForEach-Object { ($_ -split '\s+')[0] }
    return $names -contains $EnvironmentName
}

function Initialize-VaftEnvironment {
    $specification = Join-Path $RepositoryRoot 'environment.yml'
    if (-not (Test-Path -LiteralPath $specification)) {
        Stop-WithGuidance "Missing $specification."
    }
    if (Test-VaftEnvironment) {
        Write-Host "Updating the existing '$EnvironmentName' environment ..."
        conda env update --name $EnvironmentName --file $specification --prune
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
    param([Parameter(Mandatory, ValueFromRemainingArguments)] [string[]] $Arguments)
    & conda run --name $EnvironmentName --no-capture-output @Arguments
}

function Write-PythonReport {
    $description = Invoke-InVaft python -c 'import platform, sys; print(platform.python_version(), sys.executable)'
    Write-Result -Status PASS -Name 'Python' -Detail $description
}

function Install-VaftEditable {
    Write-Host "Installing VAFT in editable mode from $RepositoryRoot ..."
    Push-Location $RepositoryRoot
    try {
        Invoke-InVaft python -m pip install -e .
        if ($LASTEXITCODE -ne 0) { Stop-WithGuidance 'Editable installation failed.' }
    }
    finally {
        Pop-Location
    }
    Write-Result -Status PASS -Name 'editable VAFT installation' -Detail $RepositoryRoot
}

function Test-VaftImportLocation {
    $probe = @'
import sys
from pathlib import Path
import vaft
root = Path(sys.argv[1]).resolve()
located = Path(vaft.__file__).resolve()
if root not in located.parents:
    sys.stderr.write(f"vaft resolves to {located}, outside {root}\n")
    raise SystemExit(1)
print(located)
'@
    Invoke-InVaft python -c $probe $RepositoryRoot | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Result -Status PASS -Name 'VAFT resolves to this checkout'
    }
    else {
        Write-Result -Status FAIL -Name 'VAFT resolves to this checkout' `
            -Detail "an unrelated installed copy is shadowing $RepositoryRoot"
    }
}

function Test-Importable {
    param(
        [Parameter(Mandatory)] [string] $Module,
        [Parameter(Mandatory)] [string] $Label
    )
    Invoke-InVaft python -c "import $Module" | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Result -Status PASS -Name $Label
    }
    else {
        Write-Result -Status FAIL -Name $Label -Detail "$Module did not import"
    }
}

function Register-VaftKernel {
    # `--name vaft` overwrites any existing spec of the same name, so repeated
    # runs replace the kernel in place instead of accumulating duplicates.
    Invoke-InVaft python -m ipykernel install --user --name $KernelName --display-name $KernelDisplayName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Result -Status FAIL -Name "$KernelDisplayName kernel" -Detail 'ipykernel install failed'
        return
    }
    $probe = @'
import json, subprocess, sys
payload = subprocess.run(
    [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
    capture_output=True, text=True, check=True,
).stdout
names = list(json.loads(payload).get("kernelspecs", {}))
sys.exit(0 if names.count(sys.argv[1]) == 1 else 1)
'@
    Invoke-InVaft python -c $probe $KernelName | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Result -Status PASS -Name "$KernelDisplayName kernel"
    }
    else {
        Write-Result -Status FAIL -Name "$KernelDisplayName kernel" `
            -Detail "expected exactly one kernelspec named $KernelName"
    }
}

function Write-Summary {
    Write-Host ''
    Write-Host "VAFT bootstrap ($PlatformLabel)"
    Write-Host '--------------'
    $script:SummaryLines | ForEach-Object { Write-Host $_ }
    Write-Host @'

Next:
  1. Run `hsconfigure` if your HSDS credentials are not configured yet.
     This script never asks for, stores, or transmits your credentials.
  2. Run `conda run -n vaft python install\check_vaft_environment.py`.
  3. Run `conda activate vaft; jupyter lab`, and choose the "Python (vaft)" kernel.

This script changed only: the `vaft` Conda environment, an editable VAFT
installation inside it, and the user-level "Python (vaft)" Jupyter kernelspec.
It did not modify your repository checkout or any other Conda environment.
'@
}

Write-Host "VAFT bootstrap ($PlatformLabel)"
Write-Host "Repository: $RepositoryRoot"
Write-Host ''

Assert-Conda

if ($CheckOnly) {
    Invoke-InVaft python (Join-Path $RepositoryRoot 'install\check_vaft_environment.py')
    exit $LASTEXITCODE
}

Initialize-VaftEnvironment
Write-PythonReport
Install-VaftEditable
Test-VaftImportLocation
Test-Importable -Module 'h5pyd' -Label 'HSDS client'
Test-Importable -Module 'jupyterlab' -Label 'JupyterLab'
Register-VaftKernel
Write-Summary

if ($script:Failed) { exit 1 }
exit 0

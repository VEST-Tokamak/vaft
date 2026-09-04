<#
.SYNOPSIS
    VAFT bootstrap for native Windows.

.DESCRIPTION
    Creates or updates the `vaft` Conda environment, installs VAFT in editable
    mode from this checkout, and registers the "Python (vaft)" Jupyter kernel.

    VAFT is fully supported on native Windows; WSL2 is never required.

    The script is idempotent and strictly non-destructive by default. It never
    runs `git stash`, `git reset`, `git clean` or `git checkout`, never asks for
    or stores credentials, and never modifies a Conda environment other than
    `vaft`. The single exception is opt-in: `-Recreate` removes and rebuilds
    the `vaft` environment, and nothing else.

    Prerequisites (install these yourself first): Git for Windows and Miniconda.

.PARAMETER CheckOnly
    Run install\check_vaft_environment.py and change nothing.

.PARAMETER Recreate
    Remove the existing `vaft` environment and build it again from
    environment.yml. Use this when the environment's Python version differs
    from the version pinned by the repository.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\windows_native.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\windows_native.ps1 -CheckOnly

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install\windows_native.ps1 -Recreate
#>
[CmdletBinding()]
param(
    [switch] $CheckOnly,
    [switch] $Recreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# StrictMode treats an unset automatic variable as an error, and LASTEXITCODE
# does not exist until the first native command runs.
$global:LASTEXITCODE = 0

$EnvironmentName = 'vaft'
$KernelName = 'vaft'
$KernelDisplayName = 'Python (vaft)'
$PlatformLabel = 'Windows (native)'
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

$script:SummaryLines = New-Object System.Collections.Generic.List[string]
$script:Failed = $false
$script:Started = Get-Date

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

function Write-SolverAdvice {
    # Advisory only: this script does not install anything into `base`.
    $configured = ''
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $configured = (& conda config --show solver | Out-String).Trim() }
    catch { $configured = '' }
    finally { $ErrorActionPreference = $previous }
    if ($configured -match 'classic') {
        Write-Host @'

Note: Conda is using the `classic` solver. On Windows it can take many minutes
to solve an environment of this size. The libmamba solver is much faster; if
you want it, install and select it yourself:

    conda install -n base -c conda-forge conda-libmamba-solver
    conda config --set solver libmamba

This script will not change your Conda configuration for you.

'@
    }
}

function Test-VaftEnvironment {
    # Parsed from the plain listing so the check never needs a second interpreter.
    $names = @(
        conda env list |
            Where-Object { $_ -and -not $_.TrimStart().StartsWith('#') } |
            ForEach-Object { ($_.Trim() -split '\s+')[0] }
    )
    return $names -contains $EnvironmentName
}

function Get-PinnedPython {
    $specification = Join-Path $RepositoryRoot 'environment.yml'
    foreach ($line in Get-Content -LiteralPath $specification) {
        if ($line -match '^\s*-\s*python\s*=+\s*(\d+)\.(\d+)') {
            return "$($Matches[1]).$($Matches[2])"
        }
    }
    return ''
}

function Get-EnvironmentPython {
    # Deliberately does not import vaft: the editable install may not exist yet.
    # Use single quotes in the payload because Windows PowerShell can mangle
    # embedded double quotes passed to native executables.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & conda run --name $EnvironmentName python -c "import sys; print('%d.%d' % sys.version_info[:2])"
    }
    catch {
        return ''
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) { return '' }
    $lines = @($output | Where-Object { $_ -match '^\d+\.\d+\s*$' })
    if ($lines.Count -eq 0) { return '' }
    return $lines[-1].Trim()
}

function Remove-VaftEnvironment {
    Write-Step "Removing the existing '$EnvironmentName' environment (-Recreate) ..."
    conda env remove --name $EnvironmentName --yes
    if ($LASTEXITCODE -ne 0) { Stop-WithGuidance 'conda env remove failed.' }
}

function New-VaftEnvironment {
    param([Parameter(Mandatory)] [string] $Specification)
    Write-Step "Creating the '$EnvironmentName' environment (this can take several minutes) ..."
    conda env create --name $EnvironmentName --file $Specification
    if ($LASTEXITCODE -ne 0) { Stop-WithGuidance 'conda env create failed.' }
}

function Initialize-VaftEnvironment {
    $specification = Join-Path $RepositoryRoot 'environment.yml'
    if (-not (Test-Path -LiteralPath $specification)) {
        Stop-WithGuidance "Missing $specification."
    }
    if (-not (Test-VaftEnvironment)) {
        New-VaftEnvironment -Specification $specification
        Write-Result -Status PASS -Name 'vaft environment' -Detail 'created'
        return
    }

    if ($Recreate) {
        Remove-VaftEnvironment
        New-VaftEnvironment -Specification $specification
        Write-Result -Status PASS -Name 'vaft environment' -Detail 'recreated'
        return
    }

    $pinned = Get-PinnedPython
    $current = Get-EnvironmentPython
    if ($pinned -and $current -and $pinned -ne $current) {
        Stop-WithGuidance @"
The '$EnvironmentName' environment is on Python $current, but environment.yml
pins Python $pinned.

Updating in place can spend a long time in "Solving environment" and is likely
to fail when changing the interpreter. Rebuild the environment instead:

    powershell -ExecutionPolicy Bypass -File install\windows_native.ps1 -Recreate

That removes and recreates '$EnvironmentName' only. No other Conda environment
or checkout file is touched. Record extra packages first, if needed, with:

    conda list --name $EnvironmentName --export
"@
    }

    Write-Step "Updating the existing '$EnvironmentName' environment (this can take several minutes) ..."
    # Deliberately not --prune: preserve packages added by the student.
    conda env update --name $EnvironmentName --file $specification
    if ($LASTEXITCODE -ne 0) { Stop-WithGuidance 'conda env update failed.' }
    Write-Result -Status PASS -Name 'vaft environment' -Detail 'updated in place'
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
    Write-Result -Status PASS -Name 'Python' -Detail (($description | Select-Object -Last 1) -join ' ')
}

function Install-VaftEditable {
    Write-Step "Installing VAFT in editable mode from $RepositoryRoot ..."
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
    Write-Step "Registering the `"$KernelDisplayName`" Jupyter kernel ..."
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
    if ($Recreate) {
        Stop-WithGuidance '-CheckOnly and -Recreate are mutually exclusive: one changes nothing, the other rebuilds the environment.'
    }
    if (-not (Test-VaftEnvironment)) {
        Stop-WithGuidance @"
The '$EnvironmentName' environment does not exist yet.
Run this script without -CheckOnly to create it.
"@
    }
    Invoke-InVaft @('python', (Join-Path $RepositoryRoot 'install\check_vaft_environment.py'))
    exit $LASTEXITCODE
}

Write-SolverAdvice
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
Write-Step 'Verifying the environment ...'
Write-Host ''
Invoke-InVaft @('python', (Join-Path $RepositoryRoot 'install\check_vaft_environment.py'))
exit $LASTEXITCODE

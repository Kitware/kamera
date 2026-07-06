# Set up the native post-processing environment (Windows).
#
# GDAL comes from conda-forge (environment.yml); everything else is installed
# by uv into .venv, created on top of the conda env with
# --system-site-packages so the conda GDAL is importable. These are the same
# steps as `make install` (make usually isn't available on Windows).
#
# Requires conda, mamba, or micromamba on PATH. Usage (PowerShell):
#   .\scripts\setup_postproc.ps1
# The env name defaults to 'kamera'; override with $env:KAMERA_CONDA_ENV.
# Afterwards (conda activation is required on Windows so GDAL's DLLs resolve):
#   conda activate kamera; .\.venv\Scripts\Activate.ps1
$ErrorActionPreference = "Stop"

$PythonVersion = "3.10"  # must match PYTHON_VERSION in the Makefile

Set-Location (Join-Path $PSScriptRoot "..")

$EnvName = if ($env:KAMERA_CONDA_ENV) { $env:KAMERA_CONDA_ENV } else { "kamera" }

$CondaTool = $null
foreach ($tool in @("conda", "mamba", "micromamba")) {
    if (Get-Command $tool -ErrorAction SilentlyContinue) {
        $CondaTool = $tool
        break
    }
}
if (-not $CondaTool) {
    throw "conda, mamba, or micromamba is required on PATH"
}

$Yes = @()
if ($CondaTool -eq "micromamba") { $Yes = @("-y") }

$envExists = (& $CondaTool env list) | Where-Object { ($_.Trim() -split '\s+')[0] -eq $EnvName }
if ($envExists) {
    # No --prune: the env may hold other tools (e.g. colmap) we shouldn't remove.
    Write-Host "Updating existing env '$EnvName' with $CondaTool..."
    & $CondaTool env update -n $EnvName -f environment.yml @Yes
} else {
    Write-Host "Creating env '$EnvName' with $CondaTool..."
    & $CondaTool env create -n $EnvName -f environment.yml @Yes
}
if ($LASTEXITCODE -ne 0) { throw "$CondaTool env setup failed" }

# Same steps as `make install`
Write-Host "🚀 Creating virtual environment using uv"
& $CondaTool run -n $EnvName uv venv --system-site-packages --python=$PythonVersion
if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
& $CondaTool run -n $EnvName uv sync --frozen --no-cache
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

& $CondaTool run -n $EnvName .venv\Scripts\python.exe -c "from osgeo import gdal; print(f'GDAL {gdal.__version__} OK')"
if ($LASTEXITCODE -ne 0) { throw "GDAL import check failed" }
Write-Host ""
Write-Host "Done. To use:"
Write-Host "  $CondaTool activate $EnvName"
Write-Host "  .\.venv\Scripts\Activate.ps1"

#!/usr/bin/env bash
# Set up the native post-processing environment (Linux/macOS).
#
# GDAL comes from conda-forge (environment.yml); everything else is installed
# by uv into .venv, which is created from the conda python with
# --system-site-packages so the conda GDAL is importable.
#
# Requires conda, mamba, or micromamba on PATH. Usage:
#   ./scripts/setup_postproc.sh
# The env name defaults to 'kamera'; override with KAMERA_CONDA_ENV.
# Afterwards:
#   conda activate kamera && source .venv/bin/activate
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_NAME="${KAMERA_CONDA_ENV:-kamera}"

CONDA_TOOL=""
for tool in conda mamba micromamba; do
    if command -v "$tool" >/dev/null 2>&1; then
        CONDA_TOOL="$tool"
        break
    fi
done
if [ -z "$CONDA_TOOL" ]; then
    echo "error: conda, mamba, or micromamba is required on PATH" >&2
    exit 1
fi

YES=""
[ "$CONDA_TOOL" = "micromamba" ] && YES="-y"

if "$CONDA_TOOL" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    # No --prune: the env may hold other tools (e.g. colmap) we shouldn't remove.
    echo "Updating existing env '$ENV_NAME' with $CONDA_TOOL..."
    "$CONDA_TOOL" env update -n "$ENV_NAME" -f environment.yml $YES
else
    echo "Creating env '$ENV_NAME' with $CONDA_TOOL..."
    "$CONDA_TOOL" env create -n "$ENV_NAME" -f environment.yml $YES
fi

CONDA_PY=$("$CONDA_TOOL" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')
echo "Conda python: $CONDA_PY"

"$CONDA_TOOL" run -n "$ENV_NAME" uv venv --clear --python "$CONDA_PY" --system-site-packages .venv
"$CONDA_TOOL" run -n "$ENV_NAME" uv sync --python "$CONDA_PY"

.venv/bin/python -c 'from osgeo import gdal; print(f"GDAL {gdal.__version__} OK")'
echo
echo "Done. To use:"
echo "  $CONDA_TOOL activate $ENV_NAME"
echo "  source .venv/bin/activate"

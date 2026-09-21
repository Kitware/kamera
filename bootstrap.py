"""Build the post-processing environment on Linux, macOS or Windows.

Creates (or updates) the conda env from environment.yml, then builds .venv on top
of it with uv from the lockfile. Run it with any Python, e.g. the conda base one:

    python bootstrap.py              # env named as in environment.yml
    python bootstrap.py --name test  # a second env beside it

Afterwards activate the conda env and then .venv (the script prints the commands).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(ROOT, "environment.yml")


def read_environment_yml() -> tuple[str, str]:
    """Return (env name, python version) without needing pyyaml."""
    text = open(ENV_FILE).read()
    name = re.search(r"^name:\s*(\S+)", text, re.M)
    python = re.search(r"^\s*-\s*python\s*=\s*([\d.]+)", text, re.M)
    if not name or not python:
        sys.exit(f"could not read name and python version from {ENV_FILE}")
    return name.group(1), python.group(1)


def find_conda() -> str:
    """CONDA_EXE if it still points at a real conda (a shell can carry a stale one
    after an uninstall), else whatever conda is on PATH, else micromamba (the docker
    image has nothing else)."""
    conda = os.environ.get("CONDA_EXE", "")
    if not os.path.isfile(conda):
        conda = shutil.which("conda")
    if not conda:
        conda = os.environ.get("MAMBA_EXE", "")
        if not os.path.isfile(conda):
            conda = shutil.which("micromamba")
    if not conda:
        sys.exit(
            "conda not found; install Miniforge from https://conda-forge.org/download/"
        )
    return conda


def is_micromamba(conda: str) -> bool:
    return "micromamba" in os.path.basename(conda).lower()


def run(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def conda_env_exists(conda: str, name: str) -> bool:
    """Ask this conda whether it can resolve the env by name (a path match is not
    enough when several conda installs share a machine)."""
    probe = [conda, "run", "-n", name, "python", "--version"]
    return subprocess.run(probe, capture_output=True).returncode == 0


def main() -> None:
    default_name, python_version = read_environment_yml()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default=default_name, help="conda env name")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands without running"
    )
    args = parser.parse_args()

    conda = find_conda()
    # micromamba prompts before installing unless told not to; conda env does not.
    yes = ["-y"] if is_micromamba(conda) else []
    verb = "update" if conda_env_exists(conda, args.name) else "create"
    run([conda, "env", verb, "-n", args.name, "-f", ENV_FILE] + yes, args.dry_run)

    # uv runs inside the conda env so .venv is built on the conda python and sees
    # the conda GDAL and pycolmap through --system-site-packages. micromamba run
    # never captures output and rejects conda's flag for that.
    stream = [] if is_micromamba(conda) else ["--no-capture-output"]
    uv = [conda, "run", "-n", args.name] + stream + ["uv"]
    run(
        uv
        + ["venv", "--clear", "--system-site-packages", f"--python={python_version}"],
        args.dry_run,
    )
    run(uv + ["sync", "--frozen", "--no-cache"], args.dry_run)

    activate = (
        r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    )
    tool = "micromamba" if is_micromamba(conda) else "conda"
    print(
        "\nInstallation finished. To use kamera:"
        f"\n    {tool} activate {args.name}\n    {activate}"
    )


if __name__ == "__main__":
    main()

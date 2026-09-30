"""``kamera``: the KAMERA command.

``calibrate`` runs on every platform. ``system``, ``gui`` and ``cfg`` run on the
flight hosts from the repo checkout (the editable install): ``system`` and ``gui``
are the bash scripts in ``scripts/``. Keep imports lazy so the flight-host commands
start without loading the post-processing stack.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import scriptconfig as scfg

from kamera.calibration.config import CalibrateConfig

# Present in a repo checkout (the editable install), not in a built wheel.
REPO_DIR = Path(__file__).resolve().parents[1]


def _exec_script(name: str, *args: str) -> None:
    script = REPO_DIR / "scripts" / name
    if os.name == "nt":
        sys.exit("kamera: this command runs only on the flight hosts")
    if not script.exists():
        sys.exit(f"kamera: {script} not found; install kamera from a repo checkout")
    os.environ["KAM_REPO_DIR"] = str(REPO_DIR)
    os.execvp("bash", ["bash", str(script), *args])


def config_path() -> Path:
    """This system's config.json: ``$KAMERA_CFG``, else ``src/cfg/<SYSTEM_NAME>``."""
    if os.environ.get("KAMERA_CFG"):
        return Path(os.environ["KAMERA_CFG"])
    system = os.environ.get("SYSTEM_NAME")
    if not system:
        name_file = Path.home() / "kw" / "SYSTEM_NAME"
        if not name_file.exists():
            sys.exit(f"kamera: set SYSTEM_NAME or write it to {name_file}")
        system = name_file.read_text().strip()
    return REPO_DIR / "src" / "cfg" / system / "config.json"


class SystemCLI(scfg.DataConfig):
    """Bring the whole system up or down, or show its state."""

    __command__ = "system"
    action = scfg.Value(
        None,
        position=1,
        choices=["start", "stop", "restart", "status"],
        help="start: bring the system up (does nothing if it is up). "
        "stop: stop every host's processes and remove the GUI container. "
        "restart: stop, then start; reopens the GUI if it was open. "
        "status: ROS master reachability and every host's process states",
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        if config.action is None:
            sys.exit("usage: kamera system {start,stop,restart,status}")
        _exec_script("system.sh", config.action)


class GuiCLI(scfg.DataConfig):
    """Open the control panel on a running system."""

    __command__ = "gui"

    @classmethod
    def main(cls, argv=1, **kwargs):
        cls.cli(argv=argv, data=kwargs, strict=True)
        _exec_script("gui.sh")


class CfgCLI(scfg.DataConfig):
    """Query this system's config.json with jq, e.g. `kamera cfg .master_host`."""

    __command__ = "cfg"
    query = scfg.Value(None, position=1, help="jq query")

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        if config.query is None:
            sys.exit("usage: kamera cfg <jq query>")
        if shutil.which("jq") is None:
            sys.exit("kamera cfg: jq is not installed")
        path = config_path()
        os.execvp("jq", ["jq", "-r", config.query, str(path)])


class KameraCLI(scfg.ModalCLI):
    """KAMERA: Kitware's Image Acquisition ManagER and Archiver."""

    __prog__ = "kamera"
    system = SystemCLI
    gui = GuiCLI
    cfg = CfgCLI
    calibrate = CalibrateConfig


def _set_conda_data_paths() -> None:
    """Point PROJ and GDAL at the conda env's data, as ``conda activate`` does. The
    flight hosts run ``.venv/bin/kamera`` without activating anything."""
    prefix = Path(sys.base_prefix)
    for var, sub in (("PROJ_DATA", "share/proj"), ("GDAL_DATA", "share/gdal")):
        for path in (prefix / sub, prefix / "Library" / sub):  # Library/: Windows
            if path.is_dir():
                os.environ.setdefault(var, str(path))
                break


def main(argv: list[str] | None = None) -> None:
    _set_conda_data_paths()
    KameraCLI.main(argv=argv)


if __name__ == "__main__":
    main()

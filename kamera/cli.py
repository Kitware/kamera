"""``kamera``: the KAMERA command. ``calibrate`` runs anywhere; ``system`` and ``gui``
run ``scripts/system.sh`` on the flight hosts."""

import os
import sys
from pathlib import Path

import scriptconfig as scfg

from kamera.calibration.config import CalibrateConfig

REPO_DIR = Path(__file__).resolve().parents[1]


def _system_sh(action: str) -> None:
    if os.name == "nt":
        sys.exit("kamera: system and gui run only on the flight hosts")
    os.execvp("bash", ["bash", str(REPO_DIR / "scripts/system.sh"), action])


class SystemCLI(scfg.DataConfig):
    """Start, stop or restart the whole system, or show its status."""

    __command__ = "system"
    action = scfg.Value(
        None, position=1, choices=["start", "stop", "restart", "status"]
    )

    @classmethod
    def main(cls, argv=1, **kwargs):
        action = cls.cli(argv=argv, data=kwargs, strict=True).action
        if action is None:
            sys.exit("usage: kamera system {start,stop,restart,status}")
        _system_sh(action)


class GuiCLI(scfg.DataConfig):
    """Open the control panel on a running system."""

    __command__ = "gui"

    @classmethod
    def main(cls, argv=1, **kwargs):
        cls.cli(argv=argv, data=kwargs, strict=True)
        _system_sh("gui")


class CfgCLI(scfg.DataConfig):
    """Query this system's config.json with jq, e.g. `kamera cfg .master_host`."""

    __command__ = "cfg"
    query = scfg.Value(None, position=1, help="jq query")

    @classmethod
    def main(cls, argv=1, **kwargs):
        query = cls.cli(argv=argv, data=kwargs, strict=True).query
        if query is None:
            sys.exit("usage: kamera cfg <jq query>")
        system = os.environ.get("SYSTEM_NAME") or (
            (Path.home() / "kw/SYSTEM_NAME").read_text().strip()
        )
        config = REPO_DIR / "src/cfg" / system / "config.json"
        os.execvp("jq", ["jq", "-r", query, str(config)])


class KameraCLI(scfg.ModalCLI):
    """KAMERA: Kitware's Image Acquisition ManagER and Archiver."""

    __prog__ = "kamera"
    system = SystemCLI
    gui = GuiCLI
    cfg = CfgCLI
    calibrate = CalibrateConfig


main = KameraCLI.main

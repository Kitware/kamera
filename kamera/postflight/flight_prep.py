"""Stage a raw KAMERA flight into the images0 layout the calibration
pipeline consumes.

Raw flights store one folder per camera view (``center_view`` /
``left_view`` / ``right_view``), each holding every modality's images
plus ``*_meta.json``. The calibration pipeline instead wants one folder
per physical camera named ``<prefix>_<channel>_<modality>`` under
``<colmap_dir>/images0``, split by modality group (rgb+uv -> colmap_rgb,
ir -> colmap_ir) so SIFT never tries to match across the EO/IR gap.

Every trigger event with a nav time is staged, as symlinks by default
so nothing is duplicated on disk; frame density is a flight-planning
concern, not something this pipeline second-guesses.
"""

import os
import pathlib
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from rich import print

from kamera.postflight.naming import KameraImageName

__all__ = [
    "TriggerFrame",
    "discover_frames",
    "find_raw_dir",
    "modality_group",
    "stage_flight",
]

# EO modalities share a reconstruction; IR is on its own.
_MODALITY_GROUP = {"rgb": "rgb", "uv": "rgb", "ir": "ir"}


def modality_group(modality: str) -> str:
    return _MODALITY_GROUP.get(modality, modality)


@dataclass
class TriggerFrame:
    """All images captured at one synchronized trigger event."""

    key: str  # date_time
    time: float  # exposure time (s since epoch)
    # camera folder name -> source image path
    images: Dict[str, str] = field(default_factory=dict)


def _channel_word(view_dir: str) -> str:
    """center_view -> center; also tolerates bare center/left/right."""
    base = os.path.basename(view_dir.rstrip("/"))
    return base[: -len("_view")] if base.endswith("_view") else base


def _has_view_dirs(path: pathlib.Path) -> bool:
    # both raw layouts occur in the wild: center_view/... and bare center/...
    return any(path.glob("*_view")) or any(
        (path / v).is_dir() for v in ("center", "left", "right")
    )


def find_raw_dir(flight_dir: str | os.PathLike) -> str:
    """The raw imagery directory of a flight: `flight_dir` itself if it
    holds the ``*_view`` folders, else the single immediate subdirectory
    that does. Raises SystemError when none or several qualify (pass the
    raw dir explicitly in that case)."""
    flight = pathlib.Path(flight_dir)
    if _has_view_dirs(flight):
        return str(flight)
    candidates = [d for d in flight.iterdir() if d.is_dir() and _has_view_dirs(d)]
    if len(candidates) == 1:
        return str(candidates[0])
    detail = (
        "none found"
        if not candidates
        else "several found: " + ", ".join(d.name for d in candidates)
    )
    raise SystemError(
        f"Could not identify the raw imagery dir under {flight_dir} "
        f"(a folder containing *_view subdirectories): {detail}."
    )


def discover_frames(
    raw_dir: str | os.PathLike,
    prefix: str,
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> List[TriggerFrame]:
    """Group every image under `raw_dir`'s view folders into synchronized
    trigger frames; frames without a meta-json exposure time are dropped
    (no INS state means no prior or boresight downstream).

    The camera folder for each image is ``<prefix>_<channel>_<modality>``,
    where channel comes from the containing view folder.
    """
    frames: Dict[str, TriggerFrame] = {}
    for path in pathlib.Path(raw_dir).rglob("*"):
        if path.suffix.lower() not in image_exts:
            continue
        try:
            name = KameraImageName.parse(path.name)
        except ValueError:
            continue
        channel = _channel_word(str(path.parent))
        camera_folder = f"{prefix}_{channel}_{name.modality}"
        key = f"{name.date}_{name.time}"
        if key not in frames:
            frames[key] = TriggerFrame(key=key, time=0.0)
        frames[key].images[camera_folder] = str(path)

    times = _basename_times(raw_dir)
    out: List[TriggerFrame] = []
    for key, frame in frames.items():
        # any image's base name resolves the trigger's exposure time
        sample = next(iter(frame.images.values()))
        base = KameraImageName.parse(os.path.basename(sample)).base_name
        t = times.get(base)
        if t is None:
            continue
        frame.time = t
        out.append(frame)
    out.sort(key=lambda f: f.time)
    return out


def _basename_times(flight_dir: str | os.PathLike) -> Dict[str, float]:
    import json

    out: Dict[str, float] = {}
    for meta in pathlib.Path(flight_dir).rglob("*_meta.json"):
        try:
            with open(meta) as f:
                d = json.load(f)
            out[KameraImageName.parse(meta).base_name] = float(d["evt"]["time"])
        except (OSError, ValueError, KeyError):
            continue
    return out


def stage_flight(
    raw_dir: str | os.PathLike,
    flight_dir: str | os.PathLike,
    prefix: str,
    copy: bool = False,
) -> Dict[str, int]:
    """Discover and stage every trigger frame into per-modality-group
    ``<flight_dir>/colmap_<group>/images0/<camera>/`` trees.

    Returns a map of camera folder -> number of images staged.
    """
    frames = discover_frames(raw_dir, prefix)
    if not frames:
        raise SystemError(f"No parseable images found under {raw_dir}.")
    print(f"Staging {len(frames)} triggers.")

    counts: Dict[str, int] = defaultdict(int)
    for frame in frames:
        for camera_folder, src in frame.images.items():
            modality = camera_folder.rsplit("_", 1)[-1]
            group = modality_group(modality)
            dst_dir = os.path.join(
                str(flight_dir), f"colmap_{group}", "images0", camera_folder
            )
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            src_abs = os.path.abspath(src)
            if not _staged_ok(src_abs, dst, copy):
                if os.path.lexists(dst):
                    os.remove(dst)
                if copy:
                    shutil.copyfile(src, dst)
                else:
                    os.symlink(src_abs, dst)
            counts[camera_folder] += 1
    return dict(counts)


def _staged_ok(src: str, dst: str, copy: bool) -> bool:
    """Whether dst already stages src -- a symlink pointing at it, or for
    copies a regular file of the same size (a mid-copy crash leaves a
    shorter one). Lets restaging skip completed entries."""
    if copy:
        return (
            os.path.isfile(dst)
            and not os.path.islink(dst)
            and os.path.getsize(dst) == os.path.getsize(src)
        )
    return os.path.islink(dst) and os.readlink(dst) == src

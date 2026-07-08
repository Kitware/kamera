"""Stage a raw KAMERA flight into the images0 layout the calibration
pipeline consumes, optionally paring the frame set down.

Raw flights store one folder per camera view (``center_view`` /
``left_view`` / ``right_view``), each holding every modality's images
plus ``*_meta.json``. The calibration pipeline instead wants one folder
per physical camera named ``<prefix>_<channel>_<modality>`` under
``<colmap_dir>/images0``, split by modality group (rgb+uv -> colmap_rgb,
ir -> colmap_ir) so SIFT never tries to match across the EO/IR gap.

Frames are selected per trigger event (all synchronized cameras kept
together, which the rig/frame logic requires) and staged as symlinks by
default so nothing is duplicated on disk.
"""

import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich import print

from kamera.postflight.naming import KameraImageName
from kamera.sensor_models.nav_state import NavStateINSJson

__all__ = [
    "TriggerFrame",
    "discover_frames",
    "find_raw_dir",
    "select_by_spacing",
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
    position: np.ndarray  # INS ENU position (3,)
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
    nav_state_provider: NavStateINSJson,
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> List[TriggerFrame]:
    """Group every image under `raw_dir`'s view folders into synchronized
    trigger frames, tagged with the INS position at exposure time.

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
            frames[key] = TriggerFrame(key=key, time=0.0, position=np.zeros(3))
        frames[key].images[camera_folder] = str(path)

    # attach times + positions from the nav provider (meta json exposure time)
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
        frame.position = np.asarray(nav_state_provider.pose(t)[0], dtype=float)
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


def select_by_spacing(
    frames: List[TriggerFrame],
    spacing_m: float = 0.0,
    every: int = 1,
    max_frames: Optional[int] = None,
) -> List[TriggerFrame]:
    """Pare a time-ordered frame list.

    `spacing_m` keeps a trigger only once the aircraft has moved that far
    (3-D) from the last kept one -- even coverage regardless of speed.
    `every` additionally keeps only every Nth survivor. `max_frames`
    caps the count by uniform decimation. Order is preserved.
    """
    kept: List[TriggerFrame] = []
    last: Optional[np.ndarray] = None
    for frame in frames:
        if spacing_m > 0 and last is not None:
            if np.linalg.norm(frame.position - last) < spacing_m:
                continue
        kept.append(frame)
        last = frame.position
    if every > 1:
        kept = kept[::every]
    if max_frames is not None and len(kept) > max_frames:
        idx = np.linspace(0, len(kept) - 1, max_frames).round().astype(int)
        kept = [kept[i] for i in idx]
    return kept


def _forward_overlap(
    frames: List[TriggerFrame], focal_px: float, height_px: int, ground_m: float = 0.0
) -> Tuple[float, float]:
    """Rough forward overlap fraction from INS spacing and the along-track
    ground footprint (height_px / focal_px * AGL). AGL is the ENU up
    coordinate minus `ground_m` (default 0: origin assumed near ground).

    Returns (median overlap, overlap at the lowest-altitude frame), the
    latter being the binding constraint for SfM connectivity.
    """
    if len(frames) < 2 or focal_px <= 0:
        return float("nan"), float("nan")
    pos = np.array([f.position for f in frames])
    agl = np.clip(pos[1:, 2] - ground_m, 1.0, None)
    footprint = height_px / focal_px * agl  # along-track ground length (m)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    overlap = np.clip(1.0 - step / footprint, -1, 1)
    lowest = overlap[np.argmin(agl)]
    return float(np.median(overlap)), float(lowest)


def stage_flight(
    raw_dir: str | os.PathLike,
    flight_dir: str | os.PathLike,
    prefix: str,
    spacing_m: float = 0.0,
    every: int = 1,
    max_frames: Optional[int] = None,
    copy: bool = False,
    focal_px: float = 0.0,
    height_px: int = 4384,
) -> Dict[str, int]:
    """Discover, select, and stage frames into per-modality-group
    ``<flight_dir>/colmap_<group>/images0/<camera>/`` trees.

    Returns a map of camera folder -> number of images staged.
    """
    nav = NavStateINSJson(pathlib.Path(flight_dir).rglob("*_meta.json"))
    frames = discover_frames(raw_dir, prefix, nav)
    if not frames:
        raise SystemError(f"No parseable images found under {raw_dir}.")
    kept = select_by_spacing(frames, spacing_m, every, max_frames)

    alts = np.array([f.position[2] for f in kept])
    print(
        f"Selected {len(kept)}/{len(frames)} triggers "
        f"(altitude {alts.min():.0f}-{alts.max():.0f} m)."
    )
    if focal_px:
        med, low = _forward_overlap(kept, focal_px, height_px)
        warn = " [red](too sparse for SfM at low altitude!)" if low < 0.4 else ""
        print(
            f"Estimated forward overlap: median {med:.0%}, "
            f"lowest-altitude {low:.0%}{warn}"
        )

    counts: Dict[str, int] = defaultdict(int)
    for frame in kept:
        for camera_folder, src in frame.images.items():
            modality = camera_folder.rsplit("_", 1)[-1]
            group = modality_group(modality)
            dst_dir = os.path.join(
                str(flight_dir), f"colmap_{group}", "images0", camera_folder
            )
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            if os.path.lexists(dst):
                os.remove(dst)
            if copy:
                import shutil

                shutil.copyfile(src, dst)
            else:
                os.symlink(os.path.abspath(src), dst)
            counts[camera_folder] += 1
    return dict(counts)

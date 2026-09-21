"""Flight discovery: synchronized frames, camera names, INS trajectory, image tree."""

from __future__ import annotations

import bisect
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from kamera.sensor_models.nav_conversions import llh_to_enu

# Image suffixes written by the KAMERA archiver, keyed by modality.
MODALITY_EXT = {"rgb": ".jpg", "uv": ".jpg", "ir": ".tif"}
# NED body attitude -> ENU: swap north/east and flip down (180 deg turn about (1,1,0)).
NED_TO_ENU = Rotation.from_quat([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])


@dataclass
class Frame:
    """All images captured on one trigger event, keyed by camera name (``C_rgb``)."""

    time: float
    images: dict[str, str] = field(default_factory=dict)


class InsTrajectory:
    """INS attitude and ENU position interpolated to any time.

    Quaternions follow the KAMERA convention: ``rotation`` maps body (forward, right,
    down) vectors into the local ENU frame. Any source with times, lat/lon/alt and
    heading/pitch/roll can build one, so a future high-rate or event-stamped log drops
    in via ``__init__``.
    """

    def __init__(self, times, llh_deg, hpr_deg, lat0=None, lon0=None, h0=0.0):
        order = np.argsort(times)
        self.times = np.asarray(times, float)[order]
        self.llh = np.asarray(llh_deg, float)[order]
        hpr = np.radians(np.asarray(hpr_deg, float)[order])
        self.lat0 = float(np.median(self.llh[:, 0]) if lat0 is None else lat0)
        self.lon0 = float(np.median(self.llh[:, 1]) if lon0 is None else lon0)
        self.h0 = float(h0)
        self.enu = np.array(
            [
                llh_to_enu(*r, self.lat0, self.lon0, self.h0, in_degrees=True)
                for r in self.llh
            ]
        )
        self.rotations = NED_TO_ENU * Rotation.from_euler("ZYX", hpr)

    @classmethod
    def from_meta(cls, samples: dict[float, tuple]) -> InsTrajectory:
        rows = np.array([samples[t] for t in sorted(samples)])
        return cls(sorted(samples), rows[:, :3], rows[:, 3:])

    def _segment(self, t: float) -> int:
        """Index ``i`` with ``t`` between samples ``i - 1`` and ``i`` (clamped)."""
        return int(np.clip(bisect.bisect(self.times, t), 1, len(self.times) - 1))

    def pose(self, t: float) -> tuple[np.ndarray, Rotation]:
        i = self._segment(t)
        w = float(
            np.clip(
                (t - self.times[i - 1]) / (self.times[i] - self.times[i - 1]), 0.0, 1.0
            )
        )
        pos = (1 - w) * self.enu[i - 1] + w * self.enu[i]
        rot = Slerp([0.0, 1.0], self.rotations[[i - 1, i]])([w])[0]
        return pos, rot

    def sample_gap(self, t: float) -> float:
        """Seconds from ``t`` to the nearest INS sample (how stale the attitude is)."""
        i = self._segment(t)
        return float(min(abs(t - self.times[i - 1]), abs(t - self.times[i])))

    # Duck-type the NavStateProvider interface used by camera_models.
    def pos(self, t):
        return self.pose(t)[0]

    def quat(self, t):
        return self.pose(t)[1].as_quat()


def discover_flight(flight_dir: str) -> tuple[list[Frame], InsTrajectory, str]:
    """Group every ``*_meta.json`` under ``flight_dir`` into synchronized frames.

    Camera names are ``<channel>_<modality>`` with the channel taken from the view
    directory (``center_view`` -> ``C``). Returns frames sorted by time, the INS
    trajectory assembled from the per-image INS samples, and the rig name from
    ``sys_cfg``.
    """
    frames: dict[float, Frame] = {}
    samples: dict[float, tuple] = {}
    rig_name = ""
    for meta in glob.glob(
        os.path.join(flight_dir, "**", "*_view", "*_meta.json"), recursive=True
    ):
        with open(meta) as f:
            d = json.load(f)
        view_dir = os.path.basename(os.path.dirname(meta))  # e.g. center_view
        channel = view_dir[0].upper()
        stem = meta[: -len("_meta.json")]
        t = float(d["evt"]["time"])
        frame = frames.setdefault(round(t, 3), Frame(time=t))
        for modality, ext in MODALITY_EXT.items():
            if os.path.exists(stem + f"_{modality}{ext}"):
                frame.images[f"{channel}_{modality}"] = stem + f"_{modality}{ext}"
        ins = d["ins"]
        samples[float(ins["time"])] = (
            ins["latitude"],
            ins["longitude"],
            ins["altitude"],
            ins["heading"],
            ins["pitch"],
            ins["roll"],
        )
        rig_name = rig_name or d.get("sys_cfg", "")
    if not frames:
        raise FileNotFoundError(f"No *_meta.json files found under {flight_dir}")
    return (
        [frames[k] for k in sorted(frames)],
        InsTrajectory.from_meta(samples),
        rig_name,
    )


def normalize(src: str, dst: str) -> None:
    """Percentile-stretch (0.1-99.9) a dim or 16-bit frame to 8 bits and apply CLAHE.

    Gives SIFT some contrast to work with on UV and IR.
    """
    im = cv2.imread(src, cv2.IMREAD_UNCHANGED).astype(np.float32)
    lo, hi = np.percentile(im, [0.1, 99.9])
    im = np.clip((im - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(
        dst,
        cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5)).apply(im),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )


def build_image_tree(
    frames: list[Frame], image_dir: str
) -> dict[str, tuple[str, float]]:
    """Lay frames out as ``image_dir/<camera>/<frame time>.jpg`` for COLMAP.

    COLMAP assigns one camera per folder and groups images into rig frames by identical
    file names across folders, hence the time-based names. RGB is symlinked; the dim UV
    and 16-bit IR frames are contrast-normalized.
    Returns ``{colmap image name: (camera name, frame time)}``.
    """
    names: dict[str, tuple[str, float]] = {}
    to_normalize: list[tuple[str, str]] = []
    for frame in frames:
        for camera, src in frame.images.items():
            name = f"{camera}/{frame.time:.3f}.jpg"
            dst = os.path.join(image_dir, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            names[name] = (camera, frame.time)
            if os.path.exists(dst):
                continue
            if camera.endswith("_rgb"):
                os.symlink(os.path.abspath(src), dst)
            else:
                to_normalize.append((src, dst))
    if to_normalize:
        with ProcessPoolExecutor() as pool:
            list(pool.map(normalize, *zip(*to_normalize)))
    return names

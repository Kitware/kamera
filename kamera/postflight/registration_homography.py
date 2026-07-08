"""Per-station registration homographies for DIVE.

DIVE visualizes co-located cameras by warping one modality onto another
with a plane homography. This module derives that homography for each
non-reference camera at a station straight from the calibrated camera
models -- no scene points or hand-clicked correspondences. It writes one
``calibration.json`` per station in the DIVE format:

    {"version": 1, "pairs": [{"left": "EO", "right": "IR",
                              "leftToRight": <3x3>, "rightToLeft": <3x3>,
                              "transformType": "homography"}, ...]}

``left`` is always the reference modality (EO); ``leftToRight`` maps a
reference-image pixel to the other camera's pixel, and ``rightToLeft`` is
its inverse. The cameras are rigidly co-located, so the pixel map is a
homography to within a fraction of a pixel; DIVE renders it, it is not an
accuracy metric (use the registration gifs for that).
"""

import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from rich import print

from kamera.colmap_processing.camera_models import StandardCamera
from kamera.postflight.naming import KameraCameraName

# Modality string -> the tag DIVE labels the camera with.
MODALITY_TAG = {"rgb": "EO", "ir": "IR", "uv": "UV"}


def _tag(modality: str) -> str:
    return MODALITY_TAG.get(modality, modality.upper())


def pixel_homography(
    src: StandardCamera,
    dst: StandardCamera,
    t_src: float = 0.0,
    t_dst: float = 0.0,
    ground_z: Optional[float] = None,
) -> np.ndarray:
    """3x3 homography mapping a ``src`` pixel to the ``dst`` pixel seeing
    the same scene point.

    Sample the ``dst`` image, unproject each pixel, place the rays on the
    plane z=``ground_z`` (ENU) — or at far field when ``ground_z`` is
    None, valid only for co-located same-time views — project into
    ``src``, and fit ``src -> dst`` over the pairs that land inside
    ``src``. The times let the platform move between the two exposures.
    """
    x = np.linspace(0, dst.width - 1, max(2, dst.width // 2))
    y = np.linspace(0, dst.height - 1, max(2, dst.height // 2))
    X, Y = np.meshgrid(x, y)
    dst_pts = np.vstack([X.ravel(), Y.ravel()])

    ray_pos, ray_dir = dst.unproject(dst_pts, t_dst)
    if ground_z is None:
        xyz = ray_dir * 1e6
        forward = np.ones(dst_pts.shape[1], dtype=bool)
    else:
        s = (ground_z - ray_pos[2]) / ray_dir[2]
        forward = s > 0
        xyz = ray_pos + s * ray_dir
    src_pts = src.project(xyz, t_src).astype(np.float64)

    inside = (
        forward
        & (src_pts[0] >= 0)
        & (src_pts[0] <= src.width)
        & (src_pts[1] >= 0)
        & (src_pts[1] <= src.height)
    )
    src_pts, dst_pts = src_pts[:, inside], dst_pts[:, inside]
    if src_pts.shape[1] < 4:
        raise ValueError("too few overlapping points to fit a homography")

    # findHomography normalizes h[2, 2] to 1; the reverse direction is its
    # exact inverse, matching DIVE's stored pairs.
    h, _ = cv2.findHomography(src_pts.T, dst_pts.T)
    return h


def write_registration_homographies(
    models: Dict[str, StandardCamera],
    save_dir: str,
    reference_modality: str = "rgb",
) -> List[str]:
    """Write one DIVE ``calibration.json`` per station under ``save_dir``.

    ``models`` maps camera folder (e.g. ``21deg_N56RF_center_rgb``) to its
    StandardCamera. Cameras are grouped by station; each station's file
    pairs its reference (EO) camera with every other co-located modality.
    Returns the paths written.
    """
    by_station: Dict[str, Dict[str, StandardCamera]] = {}
    for folder, model in models.items():
        cam = KameraCameraName.parse(folder)
        by_station.setdefault(cam.channel, {})[cam.modality] = model

    written: List[str] = []
    for station, mods in sorted(by_station.items()):
        ref = mods.get(reference_modality)
        if ref is None:
            print(
                f"[yellow]Station {station}: no {reference_modality} reference "
                "camera; skipping calibration.json."
            )
            continue

        pairs = []
        for modality, model in sorted(mods.items()):
            if modality == reference_modality:
                continue
            try:
                left_to_right = pixel_homography(ref, model)  # EO -> other
            except ValueError as e:
                print(f"[yellow]Station {station} {modality}: {e}; skipping pair.")
                continue
            pairs.append(
                {
                    "left": _tag(reference_modality),
                    "right": _tag(modality),
                    "leftToRight": left_to_right.tolist(),
                    "rightToLeft": np.linalg.inv(left_to_right).tolist(),
                    "transformType": "homography",
                }
            )

        if not pairs:
            continue
        out_dir = os.path.join(save_dir, station)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "calibration.json")
        with open(out_path, "w") as f:
            json.dump({"version": 1, "pairs": pairs}, f, indent=2)
        written.append(out_path)

    return written

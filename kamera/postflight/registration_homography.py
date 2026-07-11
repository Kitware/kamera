"""Per-camera registration homographies.

Co-located cameras are visualized (e.g. in DIVE) by warping one modality
onto another with a plane homography. This module derives that homography
for each non-reference camera at a station straight from the calibrated
camera models -- no scene points or hand-clicked correspondences -- and
writes one DIVE camera-registration file per station
(``<station>_registration.json``):

    {"type": "dive-camera-registration",
     "version": 1,
     "source": {"model": "kamera-v3", "flight": "<flight>"},
     "pairs": [{"left": "eo", "right": "ir",
                "points": [[xl, yl, xr, yr], ...],
                "leftToRight": <3x3>, "rightToLeft": <3x3>,
                "transformType": "homography"}, ...]}

``left`` is always the station's reference camera under its DIVE name
(rgb -> "eo"), and ``leftToRight`` maps a left pixel to the right
camera's pixel seeing the same scene point. ``points`` are exact
correspondences under the homography, sampled on a grid over the right
image, so a consumer can refit its own transform type. The cameras are
rigidly co-located, so the pixel map is a homography to within a
fraction of a pixel; it is for rendering, not an accuracy metric (use
the registration gifs for that).
"""

import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from rich import print

from kamera.colmap_processing.camera_models import StandardCamera
from kamera.postflight.naming import KameraCameraName


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


# DIVE's camera names where they differ from KAMERA modality names.
_DIVE_CAMERA_NAMES = {"rgb": "eo"}


def _grid_correspondences(
    cam_to_ref: np.ndarray, width: int, height: int, n: int = 3
) -> List[List[float]]:
    """Exact [x_left, y_left, x_right, y_right] correspondences under the
    homography, on an ``n x n`` grid over the right (non-reference) image.
    The right footprint sits inside the reference view for co-located
    KAMERA stations, so gridding the right image keeps both ends in frame.
    """
    x = np.linspace(0, width - 1, n)
    y = np.linspace(0, height - 1, n)
    X, Y = np.meshgrid(x, y)
    right = np.vstack([X.ravel(), Y.ravel(), np.ones(X.size)])
    left = cam_to_ref @ right
    left = left[:2] / left[2]
    return np.round(np.vstack([left, right[:2]]).T, 2).tolist()


def write_registration_homographies(
    models: Dict[str, StandardCamera],
    save_dir: str,
    reference_modality: str = "rgb",
    flight: str = "",
    source_model: str = "kamera-v3",
) -> List[str]:
    """Write one DIVE camera-registration json per station under
    ``save_dir`` (``<station>_registration.json``).

    ``models`` maps camera folder (e.g. ``21deg_N56RF_center_rgb``) to its
    StandardCamera. Cameras are grouped by station; each station's file
    holds one ``pairs`` entry per non-reference camera, left = the
    reference (EO) camera. Returns the paths written.
    """
    by_station: Dict[str, Dict[str, str]] = {}
    for folder in models:
        cam = KameraCameraName.parse(folder)
        by_station.setdefault(cam.channel, {})[cam.modality] = folder

    written: List[str] = []
    for station, folders in sorted(by_station.items()):
        ref_folder = folders.get(reference_modality)
        if ref_folder is None:
            print(
                f"[yellow]Station {station}: no {reference_modality} reference "
                "camera; skipping registration homographies."
            )
            continue
        ref = models[ref_folder]
        left = _DIVE_CAMERA_NAMES.get(reference_modality, reference_modality)

        pairs = []
        for modality, folder in sorted(folders.items()):
            if modality == reference_modality:
                continue
            try:
                cam_to_ref = pixel_homography(models[folder], ref)
            except ValueError as e:
                print(f"[yellow]Station {station} {modality}: {e}; skipping pair.")
                continue
            ref_to_cam = np.linalg.inv(cam_to_ref)
            ref_to_cam /= ref_to_cam[2, 2]
            cam = models[folder]
            pairs.append(
                {
                    "left": left,
                    "right": _DIVE_CAMERA_NAMES.get(modality, modality),
                    "points": _grid_correspondences(
                        cam_to_ref, cam.width, cam.height
                    ),
                    "leftToRight": ref_to_cam.tolist(),
                    "rightToLeft": cam_to_ref.tolist(),
                    "transformType": "homography",
                }
            )
        if not pairs:
            continue

        name = KameraCameraName.parse(ref_folder)
        station_slug = f"{name.prefix}_{name.channel}" if name.prefix else name.channel
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{station_slug}_registration.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "type": "dive-camera-registration",
                    "version": 1,
                    "source": {"model": source_model, "flight": flight},
                    "pairs": pairs,
                },
                f,
                indent=2,
            )
        written.append(out_path)

    return written

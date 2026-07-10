"""Per-camera registration homographies.

Co-located cameras are visualized (e.g. in DIVE) by warping one modality
onto another with a plane homography. This module derives that homography
for each non-reference camera at a station straight from the calibrated
camera models -- no scene points or hand-clicked correspondences. It
writes one ``<camera>_to_<reference_camera>_registration.json`` per
non-reference camera (matching the registration gif naming):

    {"camera": "<camera folder>",
     "reference_camera": "<reference camera folder>",
     "camera_to_reference": <3x3>,
     "reference_to_camera": <3x3>}

``camera_to_reference`` maps a pixel in the camera's image to the
reference (EO) camera's pixel seeing the same scene point, and
``reference_to_camera`` is its inverse. The cameras are rigidly
co-located, so the pixel map is a homography to within a fraction of a
pixel; it is for rendering, not an accuracy metric (use the registration
gifs for that).
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


def write_registration_homographies(
    models: Dict[str, StandardCamera],
    save_dir: str,
    reference_modality: str = "rgb",
) -> List[str]:
    """Write one ``<camera>_to_<reference_camera>_registration.json`` per
    non-reference camera under ``save_dir``.

    ``models`` maps camera folder (e.g. ``21deg_N56RF_center_rgb``) to its
    StandardCamera. Cameras are grouped by station; each non-reference
    camera gets its own file with the homography to its station's
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

        for modality, folder in sorted(folders.items()):
            if modality == reference_modality:
                continue
            try:
                cam_to_ref = pixel_homography(models[folder], ref)
            except ValueError as e:
                print(f"[yellow]Station {station} {modality}: {e}; skipping pair.")
                continue
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(
                save_dir, f"{folder}_to_{ref_folder}_registration.json"
            )
            with open(out_path, "w") as f:
                json.dump(
                    {
                        "camera": folder,
                        "reference_camera": ref_folder,
                        "camera_to_reference": cam_to_ref.tolist(),
                        "reference_to_camera": np.linalg.inv(cam_to_ref).tolist(),
                    },
                    f,
                    indent=2,
                )
            written.append(out_path)

    return written

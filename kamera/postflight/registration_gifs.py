"""Registration QC gifs for calibrated camera models.

For each non-reference camera, warp its colocated reference-modality
image (same station) into the camera's view and flip between the two.
Well-calibrated mounts make ground features hold still across the flip;
misregistration shows up as jitter. This is the rig pipeline's
equivalent of the legacy ColmapCalibration.write_gifs, driven purely by
exported StandardCamera models rather than a reconstruction.
"""

import os
import pathlib
from typing import Dict, List

import cv2
import numpy as np
import PIL.Image
from rich import print

from kamera.colmap_processing.camera_models import StandardCamera
from kamera.colmap_processing.image_renderer import render_view
from kamera.postflight.naming import KameraCameraName, KameraImageName  # noqa: F401

__all__ = ["write_registration_gifs"]

def _index_by_time(image_dir: str, times: Dict[str, float]) -> Dict[float, str]:
    out: Dict[float, str] = {}
    for path in pathlib.Path(image_dir).iterdir():
        try:
            name = KameraImageName.parse(path.name)
            t = times[name.base_name]
        except (ValueError, KeyError):
            continue
        out[t] = str(path)
    return out


def write_registration_gifs(
    models: Dict[str, StandardCamera],
    image_dirs: Dict[str, str],
    times: Dict[str, float],
    save_dir: str | os.PathLike,
    ref_modality: str = "rgb",
    num_gifs: int = 5,
    width: int = 1280,
) -> List[str]:
    """Write flip gifs pairing each non-reference camera with its
    colocated reference-modality camera.

    `models` and `image_dirs` are keyed by camera folder name. `times`
    maps image base names to exposure times (rig.basename_to_time).
    Returns the list of gif paths written.
    """
    os.makedirs(save_dir, exist_ok=True)
    # camera folder -> reference-modality folder at the same station
    by_station_ref: Dict[str, str] = {}
    for folder in models:
        cam = KameraCameraName.parse(folder)
        if cam.modality != ref_modality:
            continue
        by_station_ref[cam.channel] = folder

    written: List[str] = []
    for folder, model in sorted(models.items()):
        cam = KameraCameraName.parse(folder)
        if cam.modality == ref_modality:
            continue
        ref_folder = by_station_ref.get(cam.channel)
        if ref_folder is None or ref_folder not in models:
            print(f"[yellow]No {ref_modality} reference for {folder}; no gif.")
            continue
        if folder not in image_dirs or ref_folder not in image_dirs:
            print(f"[yellow]Missing image dir for {folder}/{ref_folder}; no gif.")
            continue

        cam_by_t = _index_by_time(image_dirs[folder], times)
        ref_by_t = _index_by_time(image_dirs[ref_folder], times)
        shared = sorted(set(cam_by_t) & set(ref_by_t))
        if not shared:
            print(f"[yellow]No synchronized {folder}/{ref_folder} pairs; no gif.")
            continue
        # spread the samples across the flight
        picks = shared[:: max(1, len(shared) // max(num_gifs, 1))][:num_gifs]

        ref_model = models[ref_folder]
        for k, t in enumerate(picks):
            img = cv2.imread(cam_by_t[t], cv2.IMREAD_COLOR)
            ref_img = cv2.imread(ref_by_t[t], cv2.IMREAD_COLOR)
            if img is None or ref_img is None:
                continue
            img = img[:, :, ::-1]
            ref_img = ref_img[:, :, ::-1]
            # warp the reference image into this camera's view
            warped, _ = render_view(ref_model, ref_img, t, model, t, block_size=10)
            h, w = img.shape[:2]
            size = (width, int(width * h / w))
            frame_a = PIL.Image.fromarray(np.ascontiguousarray(img)).resize(size)
            frame_b = PIL.Image.fromarray(np.ascontiguousarray(warped)).resize(size)
            out_path = os.path.join(
                save_dir, f"{folder}_vs_{ref_folder}_{k}.gif"
            )
            frame_a.save(
                out_path,
                save_all=True,
                append_images=[frame_b],
                duration=350,
                loop=0,
            )
            written.append(out_path)
            print(f"Wrote {out_path}")
    return written

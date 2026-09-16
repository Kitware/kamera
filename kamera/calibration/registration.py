"""Inter-camera homographies: DIVE camera-registration JSON (format v2) and GIF overlays.

Each ``<left>_to_<right>_registration.json`` holds one matrix-only pair whose
``leftToRight`` homography maps left-camera pixels onto right-camera pixels. The
matrix is fit to the calibrated models by casting a grid of left pixels to a nominal
ground range and projecting them into the right camera. The range matters: cameras
whose exposure lags the trigger sit an effective metre or so along track, and that
baseline only vanishes at infinity. The fit residual is reported.
"""

from __future__ import annotations

import datetime
import json
import os

import cv2
import numpy as np
import PIL.Image

DIVE_TYPE = "dive-camera-registration"
DIVE_VERSION = 2


def model_homography(src_cm, dst_cm, range_m: float, grid: int = 40) -> tuple[np.ndarray, dict]:
    """Least-squares homography from ``src_cm`` pixels to ``dst_cm`` pixels for ground ``range_m`` away, plus fit stats."""
    xg, yg = np.meshgrid(np.linspace(0, src_cm.width - 1, grid), np.linspace(0, src_cm.height - 1, grid))
    src = np.vstack([xg.ravel(), yg.ravel()])
    ray_pos, ray_dir = src_cm.unproject(src, -np.inf)
    dst = np.asarray(dst_cm.project(ray_pos + ray_dir * range_m, -np.inf), dtype=np.float64)
    inside = np.all(np.isfinite(dst), 0) & (dst[0] >= 0) & (dst[0] <= dst_cm.width) & (dst[1] >= 0) & (dst[1] <= dst_cm.height)
    if inside.sum() < 4:
        raise ValueError(f"only {inside.sum()} of {src.shape[1]} samples land in the destination image")
    h, _ = cv2.findHomography(src[:, inside].T, dst[:, inside].T, 0)
    err = np.linalg.norm(cv2.perspectiveTransform(src[:, inside].T.reshape(-1, 1, 2), h).reshape(-1, 2) - dst[:, inside].T, axis=1)
    stats = {"rmsPx": float(np.sqrt(np.mean(err**2))), "p95Px": float(np.percentile(err, 95)),
             "maxPx": float(np.max(err)), "coverage": float(inside.mean()), "rangeM": float(range_m)}
    return h, stats


def write_dive_registration(out_dir: str, left: str, right: str, h: np.ndarray, stats: dict, source: dict) -> str:
    """Write one matrix-only v2 pair file and return its path."""
    inv = np.linalg.inv(h)
    pair = {"left": left, "right": right, "transformType": "homography",
            "leftToRight": h.tolist(), "rightToLeft": (inv / inv[2, 2]).tolist(), "observations": [],
            "stats": {f"modelFit{k[0].upper()}{k[1:]}": v for k, v in stats.items()}}
    body = {"type": DIVE_TYPE, "version": DIVE_VERSION, "source": source, "pairs": [pair]}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{left}_to_{right}_registration.json")
    with open(path, "w") as f:
        json.dump(body, f, indent=2)
    return path


def source_stamp(flight_dir: str, extra: dict | None = None) -> dict:
    stamp = {"producer": "kamera-rig-calibration", "flight": os.path.basename(os.path.abspath(flight_dir)),
             "generated": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")}
    return {**stamp, **(extra or {})}


def warp_pair(left_img: np.ndarray, right_img: np.ndarray, h: np.ndarray, width: int = 1280) -> tuple[np.ndarray, np.ndarray]:
    """Warp the left image into the right image's pixels; both returned resized to ``width`` wide, RGB."""
    scale = width / right_img.shape[1]
    size = (width, round(right_img.shape[0] * scale))
    s = np.diag([scale, scale, 1.0])
    warped = cv2.warpPerspective(left_img, s @ h, size, flags=cv2.INTER_LINEAR)
    return _rgb(warped), _rgb(cv2.resize(right_img, size, interpolation=cv2.INTER_AREA))


def _rgb(im: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else im[:, :, ::-1]


def write_gif(path: str, a: np.ndarray, b: np.ndarray, duration_ms: int = 400) -> None:
    PIL.Image.fromarray(a).save(path, save_all=True, append_images=[PIL.Image.fromarray(b)], duration=duration_ms, loop=0)


def blend_overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Magenta/green false-colour blend: misregistration shows as coloured fringes."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return np.dstack([ga, gb, ga])

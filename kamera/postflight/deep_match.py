"""Deep cross-modal image matching for calibration fusion.

SIFT cannot match long-wave IR to visible imagery, which is why the
calibration pipeline reconstructs each modality group separately. This
module wraps MINIMA (modality-invariant matchers, CVPR 2025) via the
``vismatch`` package so IR images can be matched directly against EO
images and registered into the EO reconstruction (see ``fusion.py``).

``vismatch`` pulls in torch and is deliberately an optional dependency
(``uv sync --group fusion``); it is imported lazily inside
``DeepMatcher`` so the rest of postflight works without it. Model
weights are downloaded automatically on first use (network required).
"""

from typing import Tuple

import cv2
import numpy as np

__all__ = ["DeepMatcher", "to_uint8"]


def to_uint8(img: np.ndarray, clahe: bool = False) -> np.ndarray:
    """An image as 8-bit grayscale, contrast-stretched if necessary.

    uint8 input passes through (modulo an RGB->gray collapse); higher
    bit depths (16-bit IR TIFFs) are stretched over their 1-99
    percentile range. `clahe` additionally applies local equalization,
    which can help very low-contrast thermal scenes.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.float64)
        lo, hi = np.percentile(img, [1, 99])
        if hi <= lo:
            lo, hi = float(img.min()), float(max(img.max(), img.min() + 1))
        img = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    if clahe:
        img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
    return img


def _matcher_input(img: np.ndarray) -> np.ndarray:
    """A grayscale/BGR image as the (3, H, W) float32 [0, 1] array
    vismatch matchers consume."""
    img = to_uint8(img)
    chan = img.astype(np.float32) / 255.0
    return np.stack([chan] * 3)


class DeepMatcher:
    """Semi-dense cross-modal matcher (default MINIMA-LoFTR).

    Matched keypoints are returned in the pixel coordinates of the
    arrays passed to `match` -- any resizing the underlying model does
    internally is rescaled back by vismatch (covered by the identity and
    known-shift tests in test_deep_match.py).
    """

    def __init__(
        self,
        matcher_name: str = "minima-loftr",
        device: str = "auto",
        max_num_keypoints: int = 2048,
    ):
        try:
            from vismatch import get_default_device, get_matcher
        except ImportError as e:
            raise ImportError(
                "Cross-modal fusion needs the optional 'fusion' dependency "
                "group: uv sync --group fusion"
            ) from e
        if device == "auto":
            device = get_default_device()
        self.name = matcher_name
        self.device = device
        try:
            self._matcher = get_matcher(
                matcher_name, device=device, max_num_keypoints=max_num_keypoints
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize matcher '{matcher_name}' (weights are "
                "downloaded on first use; check network access)."
            ) from e

    def match(
        self, img0: np.ndarray, img1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match two images; returns (kpts0 [N,2], kpts1 [N,2]) float64
        pixel coordinates of the putative matches (no per-match scores --
        downstream RANSAC handles outliers)."""
        result = self._matcher(_matcher_input(img0), _matcher_input(img1))
        kpts0 = np.asarray(result["matched_kpts0"], dtype=np.float64).reshape(-1, 2)
        kpts1 = np.asarray(result["matched_kpts1"], dtype=np.float64).reshape(-1, 2)
        return kpts0, kpts1

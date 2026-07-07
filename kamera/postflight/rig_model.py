"""Assemble a complete, self-contained rig model JSON for a flight.

Where the per-camera ``*_v3.yaml`` files each describe one camera, the
rig JSON is the authoritative description of the whole mount: every
camera's intrinsics and INS mount, the rig extrinsics
(``sensor_from_rig``) and boresight (``ins_from_rig``) that relate them,
the ENU reference frame, and the provenance and quality of the
calibration. It is written once per flight as
``<flight>_<date>_<config>_rig.json``.
"""

import datetime
import json
import os
import subprocess
from typing import Dict, List, Optional

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from kamera.colmap_processing.camera_models import StandardCamera
from kamera.postflight.boresight import BoresightEstimate
from kamera.postflight.naming import KameraCameraName, KameraImageName

SCHEMA_VERSION = "1.0"

# The mount quaternion convention, documented in the file so a reader
# never has to guess (this is what StandardCamera.unproject implements).
QUATERNION_CONVENTION = (
    "unit quaternion [x, y, z, w]; camera_quaternion maps a ray from the "
    "camera frame into the INS body frame (ins_from_camera). "
    "sensor_from_rig maps rig -> sensor; ins_from_rig (boresight) maps rig "
    "-> INS body. Reference sensor defines the rig frame."
)


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _rigid_dict(rotation: Rotation, translation: np.ndarray) -> Dict:
    return {
        "quaternion_xyzw": [float(x) for x in rotation.as_quat()],
        "translation_m": [float(x) for x in np.asarray(translation)],
    }


def reprojection_error_px(
    reconstruction: "pycolmap.Reconstruction",
    model: StandardCamera,
    folder: str,
    times: Dict[str, float],
    max_images: int = 60,
) -> Optional[float]:
    """Median per-image reprojection error of `model` against the
    reconstruction's own 3D observations for that camera folder."""
    errs: List[float] = []
    for im in reconstruction.images.values():
        if not im.has_pose or im.name.rsplit("/", 1)[0] != folder:
            continue
        try:
            t = times[KameraImageName.parse(im.name).base_name]
        except (ValueError, KeyError):
            continue
        xy, xyz = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                xy.append(p2.xy)
                xyz.append(reconstruction.points3D[p2.point3D_id].xyz)
        if len(xy) < 5:
            continue
        proj = model.project(np.asarray(xyz).T, t).T
        errs.append(float(np.median(np.sqrt(np.sum((np.asarray(xy) - proj) ** 2, 1)))))
        if len(errs) >= max_images:
            break
    return float(np.median(errs)) if errs else None


def camera_record(
    folder: str,
    model: StandardCamera,
    sensor_from_rig: Optional["pycolmap.Rigid3d"],
    is_reference: bool,
    reprojection_px: Optional[float],
    num_images: int,
) -> Dict:
    name = KameraCameraName.parse(folder)
    rec = {
        "name": folder,
        "channel": name.channel,
        "modality": name.modality,
        "is_reference": is_reference,
        "image_width": int(model.width),
        "image_height": int(model.height),
        "model_type": "opencv",
        "fx": float(model.fx),
        "fy": float(model.fy),
        "cx": float(model.cx),
        "cy": float(model.cy),
        "distortion_opencv": [float(x) for x in np.asarray(model.dist).ravel()],
        # the mount: camera -> INS body
        "camera_quaternion_xyzw": [float(x) for x in np.asarray(model.cam_quat)],
        "camera_position_m": [float(x) for x in np.asarray(model.cam_pos)],
        "num_images": int(num_images),
        "reprojection_error_px": reprojection_px,
    }
    if is_reference:
        rec["sensor_from_rig"] = _rigid_dict(Rotation.identity(), np.zeros(3))
    elif sensor_from_rig is not None:
        rec["sensor_from_rig"] = _rigid_dict(
            Rotation.from_quat(sensor_from_rig.rotation.quat),
            np.asarray(sensor_from_rig.translation),
        )
    else:
        rec["sensor_from_rig"] = None
    return rec


def group_record(
    modality_group: str,
    reference_camera: str,
    estimate: BoresightEstimate,
    model_source: str,
    num_images: int,
) -> Dict:
    return {
        "modality_group": modality_group,
        "reference_camera": reference_camera,
        "model_source": model_source,
        "boresight_ins_from_rig_xyzw": [float(x) for x in estimate.ins_from_rig],
        "lever_arm_ins_m": [float(x) for x in estimate.lever_arm_ins],
        "num_frames_used": int(estimate.num_frames),
        "num_frames_rejected": int(estimate.num_rejected),
        "boresight_residual_deg": {
            "median": float(np.median(estimate.residuals_deg)),
            "p90": float(np.percentile(estimate.residuals_deg, 90)),
            "max": float(np.max(estimate.residuals_deg)),
        },
        "num_images_registered": int(num_images),
    }


def build_rig_model(
    reconstruction: "pycolmap.Reconstruction",
    nav_state_provider,
    groups: List[Dict],
    cameras: List[Dict],
    extra_provenance: Optional[Dict] = None,
) -> Dict:
    """Assemble the full rig dict. Flight identity is read from a sample
    image name; `groups` and `cameras` come from group_record /
    camera_record."""
    sample = next(iter(reconstruction.images.values())).name
    img = KameraImageName.parse(os.path.basename(sample))
    config = KameraCameraName.parse(cameras[0]["name"]).prefix

    provenance = {
        "flight": img.flight,
        "effort": img.prefix,
        "config": config,
        "date": img.date,
        "calibrated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "rig-boresight",
        "pycolmap_version": pycolmap.__version__,
        "git_commit": _git_commit(),
    }
    if extra_provenance:
        provenance.update(extra_provenance)

    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "reference_frame": {
            "platform": "INS body",
            "world": "ENU",
            "quaternion_convention": QUATERNION_CONVENTION,
            "enu_origin_llh": {
                "lat0_deg": getattr(nav_state_provider, "lat0", None),
                "lon0_deg": getattr(nav_state_provider, "lon0", None),
                "h0_m": getattr(nav_state_provider, "h0", None),
            },
        },
        "groups": groups,
        "cameras": cameras,
    }


def rig_json_path(save_dir: str | os.PathLike, rig: Dict) -> str:
    p = rig["provenance"]
    fname = f"{p['flight']}_{p['date']}_{p['config']}_rig.json"
    return os.path.join(str(save_dir), fname)


def write_rig_json(save_dir: str | os.PathLike, rig: Dict) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = rig_json_path(save_dir, rig)
    with open(path, "w") as f:
        json.dump(rig, f, indent=2)
    return path

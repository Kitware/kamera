"""Solve the rig-to-INS boresight from a rig-constrained reconstruction.

With rigs in the COLMAP model, bundle adjustment has already solved each
camera's pose relative to the rig (``sensor_from_rig``). The only
remaining unknown between the reconstruction and the aircraft is a
single rigid transform: the rotation of the rig relative to the INS
(the boresight) and, optionally, the lever arm. One robust estimate
over every registered frame replaces the per-camera quaternion searches
of the legacy pipeline, and makes all camera mounts mutually consistent
by construction:

    mount(cam) = sensor_from_rig(cam) o rig_from_ins

The reconstruction must be in the same ENU frame as the nav provider
(which prior-position mapping yields directly).
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycolmap
from rich import print
from scipy.spatial.transform import Rotation

from kamera.colmap_processing.camera_models import StandardCamera
from kamera.postflight.naming import KameraImageName
from kamera.sensor_models.nav_state import NavStateINSJson

__all__ = [
    "BoresightEstimate",
    "average_quaternions",
    "solve_rig_boresight",
    "export_rig_camera_models",
]


@dataclass
class BoresightEstimate:
    rig_from_ins: np.ndarray  # quaternion (x, y, z, w)
    lever_arm_ins: np.ndarray  # rig origin relative to INS, in the INS frame (m)
    num_frames: int  # frames used (inliers)
    num_rejected: int  # frames rejected as outliers
    residuals_deg: np.ndarray  # per-inlier-frame angular residual


def average_quaternions(
    quats: np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Chordal L2 mean of unit quaternions (largest eigenvector of the
    weighted outer-product matrix). Handles the q/-q sign ambiguity."""
    Q = np.asarray(quats, dtype=float).copy()
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    signs = np.sign(Q @ Q[0])
    signs[signs == 0] = 1
    Q *= signs[:, None]
    w = np.ones(len(Q)) if weights is None else np.asarray(weights, dtype=float)
    M = (Q.T * w) @ Q
    _, eigvecs = np.linalg.eigh(M)
    mean = eigvecs[:, -1]
    return mean / np.linalg.norm(mean)


def _frame_time(
    reconstruction: "pycolmap.Reconstruction",
    frame: "pycolmap.Frame",
    times: Dict[str, float],
) -> Optional[float]:
    for data_id in frame.data_ids:
        image = reconstruction.images.get(data_id.id)
        if image is None:
            continue
        try:
            return times[KameraImageName.parse(image.name).base_name]
        except (ValueError, KeyError):
            continue
    return None


def solve_rig_boresight(
    reconstruction: "pycolmap.Reconstruction",
    nav_state_provider: NavStateINSJson,
    times: Dict[str, float],
    outlier_threshold_deg: float = 2.0,
    max_iterations: int = 5,
) -> BoresightEstimate:
    """Estimate rig_from_ins by robust rotation averaging over all
    registered frames of an ENU-registered reconstruction.

    `times` maps image base names to exposure times (see
    `kamera.postflight.rig.basename_to_time`).
    """
    rel_quats: List[np.ndarray] = []
    lever_arms: List[np.ndarray] = []
    for frame in reconstruction.frames.values():
        if not frame.has_pose:
            continue
        t = _frame_time(reconstruction, frame, times)
        if t is None:
            continue
        ins_pos, ins_quat = nav_state_provider.pose(t)
        # World here is ENU; the INS quaternion is the body attitude in ENU.
        R_enu_from_ins = Rotation.from_quat(ins_quat)
        rig_from_world = frame.rig_from_world
        R_rig_from_enu = Rotation.from_quat(rig_from_world.rotation.quat)
        rel_quats.append((R_rig_from_enu * R_enu_from_ins).as_quat())
        # Rig origin in ENU, then expressed in the INS body frame.
        R = rig_from_world.rotation.matrix()
        rig_pos_enu = -R.T @ rig_from_world.translation
        lever_arms.append(R_enu_from_ins.inv().apply(rig_pos_enu - np.asarray(ins_pos)))

    if len(rel_quats) < 3:
        raise SystemError(
            f"Only {len(rel_quats)} frames with poses and nav times; "
            "cannot solve the boresight."
        )

    quats = np.asarray(rel_quats)
    levers = np.asarray(lever_arms)
    inliers = np.ones(len(quats), dtype=bool)
    mean = average_quaternions(quats)
    for _ in range(max_iterations):
        mean_rot = Rotation.from_quat(mean)
        residuals = np.degrees(
            (Rotation.from_quat(quats) * mean_rot.inv()).magnitude()
        )
        new_inliers = residuals < outlier_threshold_deg
        if new_inliers.sum() < 3 or np.array_equal(new_inliers, inliers):
            inliers = new_inliers if new_inliers.sum() >= 3 else inliers
            break
        inliers = new_inliers
        mean = average_quaternions(quats[inliers])

    mean_rot = Rotation.from_quat(mean)
    residuals = np.degrees(
        (Rotation.from_quat(quats[inliers]) * mean_rot.inv()).magnitude()
    )
    estimate = BoresightEstimate(
        rig_from_ins=mean,
        lever_arm_ins=np.median(levers[inliers], axis=0),
        num_frames=int(inliers.sum()),
        num_rejected=int((~inliers).sum()),
        residuals_deg=residuals,
    )
    print(
        f"Boresight from {estimate.num_frames} frames "
        f"({estimate.num_rejected} rejected): "
        f"median residual {np.median(residuals):.3f} deg, "
        f"p90 {np.percentile(residuals, 90):.3f} deg."
    )
    print(f"Lever arm (INS frame, m): {np.round(estimate.lever_arm_ins, 3)}")
    return estimate


def _camera_folder_names(
    reconstruction: "pycolmap.Reconstruction",
) -> Dict[int, str]:
    folders: Dict[int, str] = {}
    for image in reconstruction.images.values():
        if image.camera_id not in folders and "/" in image.name:
            folders[image.camera_id] = image.name.rsplit("/", 1)[0]
    return folders


def export_rig_camera_models(
    reconstruction: "pycolmap.Reconstruction",
    estimate: BoresightEstimate,
    nav_state_provider: NavStateINSJson,
    save_dir: str | os.PathLike,
    use_lever_arm: bool = False,
    suffix: str = "_v3",
) -> Dict[str, StandardCamera]:
    """Compose mount = sensor_from_rig o rig_from_ins for every rig
    sensor and write a StandardCamera yaml per camera."""
    os.makedirs(save_dir, exist_ok=True)
    folders = _camera_folder_names(reconstruction)
    rig_from_ins = Rotation.from_quat(estimate.rig_from_ins)
    position = estimate.lever_arm_ins if use_lever_arm else np.zeros(3)
    models: Dict[str, StandardCamera] = {}
    for rig in reconstruction.rigs.values():
        sensor_ids = [rig.ref_sensor_id] + list(rig.non_ref_sensors.keys())
        for sensor_id in sensor_ids:
            cam_id = sensor_id.id
            camera = reconstruction.cameras.get(cam_id)
            folder = folders.get(cam_id)
            if camera is None or folder is None:
                print(f"[yellow]Sensor {sensor_id} has no camera/images, skipping.")
                continue
            if rig.is_ref_sensor(sensor_id):
                cam_from_rig = Rotation.identity()
                cam_offset_rig = np.zeros(3)
            else:
                sensor_from_rig = rig.sensor_from_rig(sensor_id)
                cam_from_rig = Rotation.from_quat(sensor_from_rig.rotation.quat)
                R = sensor_from_rig.rotation.matrix()
                cam_offset_rig = -R.T @ sensor_from_rig.translation
            mount = cam_from_rig * rig_from_ins
            if camera.model.name == "OPENCV":
                fx, fy, cx, cy, d1, d2, d3, d4 = camera.params
                dist = np.array([d1, d2, d3, d4])
            elif camera.model.name == "PINHOLE":
                fx, fy, cx, cy = camera.params
                dist = np.zeros(4)
            else:
                raise SystemError(f"Unexpected camera model {camera.model.name}")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            cam_position = position + rig_from_ins.inv().apply(cam_offset_rig)
            model = StandardCamera(
                camera.width,
                camera.height,
                K,
                dist,
                cam_position if use_lever_arm else np.zeros(3),
                mount.as_quat(),
                platform_pose_provider=nav_state_provider,
            )
            out_path = os.path.join(save_dir, f"{folder}{suffix}.yaml")
            model.save_to_file(out_path)
            print(f"Wrote {out_path}")
            models[folder] = model
    return models

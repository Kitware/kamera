"""Solve the rig-to-INS boresight from a prior-mapped ENU reconstruction.

The reference sensor defines the rig frame, so its per-frame pose is the
rig pose. The only unknown between the reconstruction and the aircraft
is then a single rigid transform: the rotation of the rig relative to
the INS (the boresight) and, optionally, the lever arm. One robust
estimate over every synchronized frame replaces the per-camera
quaternion searches of the legacy pipeline, and -- combined with the
`sensor_from_rig` extrinsics from `rig.derive_sensor_from_rig` -- makes
all camera mounts mutually consistent by construction:

    mount(cam) = ins_from_rig o rig_from_sensor(cam)

This works directly on the single rigless ENU model that prior-position
mapping produces (grouping images into frames by trigger time); it does
NOT need a rig-constrained reconstruction, which fragments badly to
build. The reconstruction must share the nav provider's ENU frame.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pycolmap
from rich import print
from scipy.spatial.transform import Rotation

from kamera.colmap_processing.camera_models import StandardCamera
from kamera.postflight.naming import KameraImageName
from kamera.postflight.rig import _order_by_ref, derive_sensor_from_rig
from kamera.sensor_models.nav_state import NavStateINSJson

__all__ = [
    "BoresightEstimate",
    "average_quaternions",
    "solve_rig_boresight",
    "export_rig_camera_models",
]


@dataclass
class BoresightEstimate:
    # Quaternion (x, y, z, w) mapping rig coordinates into the INS body
    # frame. This is the reference sensor's StandardCamera mount directly:
    # StandardCamera's camera_quaternion maps camera->INS (see unproject in
    # camera_models.py), and the reference sensor is the rig frame.
    ins_from_rig: np.ndarray
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


def solve_rig_boresight(
    reconstruction: "pycolmap.Reconstruction",
    nav_state_provider: NavStateINSJson,
    times: Dict[str, float],
    ref_modality: str = "rgb",
    outlier_threshold_deg: float = 2.0,
    max_iterations: int = 5,
) -> BoresightEstimate:
    """Estimate ins_from_rig by robust rotation averaging over the
    reference sensor's images of an ENU-registered reconstruction.

    The reference sensor (first camera folder of `ref_modality`) defines
    the rig frame, so its per-image pose is the rig pose. Per frame the
    boresight is

        ins_from_rig = (enu_from_ins)^-1 . (rig_from_enu)^-1

    which is constant across frames because the rig is rigidly mounted to
    the INS. The reconstruction must share the nav provider's ENU frame
    (which prior-position mapping yields); an arbitrary-gauge model, e.g.
    from the global mapper which ignores priors, gives a non-constant
    per-frame estimate (large residual spread) and must not be used here.

    `times` maps image base names to exposure times (see
    `kamera.postflight.rig.basename_to_time`).
    """
    folders = {
        im.name.rsplit("/", 1)[0]
        for im in reconstruction.images.values()
        if im.has_pose and "/" in im.name
    }
    if not folders:
        raise SystemError("Reconstruction has no posed, folder-qualified images.")
    ref_folder = _order_by_ref(folders, ref_modality)[0]

    rel_quats: List[np.ndarray] = []
    lever_arms: List[np.ndarray] = []
    for im in reconstruction.images.values():
        if not im.has_pose or im.name.rsplit("/", 1)[0] != ref_folder:
            continue
        try:
            t = times[KameraImageName.parse(im.name).base_name]
        except (ValueError, KeyError):
            continue
        ins_pos, ins_quat = nav_state_provider.pose(t)
        # World here is ENU; the INS quaternion is the body attitude in ENU.
        R_enu_from_ins = Rotation.from_quat(ins_quat)
        rig_from_world = im.cam_from_world()
        R_rig_from_enu = Rotation.from_quat(rig_from_world.rotation.quat)
        # ins_from_rig = (enu_from_ins)^-1 . (rig_from_enu)^-1
        rel_quats.append((R_enu_from_ins.inv() * R_rig_from_enu.inv()).as_quat())
        # Rig origin in ENU, then expressed in the INS body frame.
        R = rig_from_world.rotation.matrix()
        rig_pos_enu = -R.T @ rig_from_world.translation
        lever_arms.append(R_enu_from_ins.inv().apply(rig_pos_enu - np.asarray(ins_pos)))

    if len(rel_quats) < 3:
        raise SystemError(
            f"Only {len(rel_quats)} reference-sensor frames with nav times; "
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
        ins_from_rig=mean,
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


def _folder_camera(
    reconstruction: "pycolmap.Reconstruction",
) -> Dict[str, "pycolmap.Camera"]:
    """Map each camera folder to its Camera, via a posed image."""
    out: Dict[str, "pycolmap.Camera"] = {}
    for image in reconstruction.images.values():
        if not image.has_pose or "/" not in image.name:
            continue
        folder = image.name.rsplit("/", 1)[0]
        if folder not in out:
            out[folder] = reconstruction.cameras[image.camera_id]
    return out


def export_rig_camera_models(
    reconstruction: "pycolmap.Reconstruction",
    estimate: BoresightEstimate,
    nav_state_provider: NavStateINSJson,
    save_dir: str | os.PathLike,
    ref_modality: str = "rgb",
    sensor_from_rig: Optional[Dict[str, "pycolmap.Rigid3d"]] = None,
    use_lever_arm: bool = False,
    suffix: str = "_v3",
) -> Dict[str, StandardCamera]:
    """Compose and write a StandardCamera yaml per camera folder.

    The mount (StandardCamera.camera_quaternion, camera->INS) is

        cam_quat(sensor) = ins_from_rig . rig_from_sensor

    where rig_from_sensor comes from `sensor_from_rig` (derived by
    `rig.derive_sensor_from_rig`; computed here if not supplied). For the
    reference sensor rig_from_sensor is identity, so its mount is the
    boresight itself.
    """
    os.makedirs(save_dir, exist_ok=True)
    if sensor_from_rig is None:
        sensor_from_rig = derive_sensor_from_rig(reconstruction, ref_modality)
    folder_camera = _folder_camera(reconstruction)
    ref_folder = _order_by_ref(folder_camera, ref_modality)[0]
    ins_from_rig = Rotation.from_quat(estimate.ins_from_rig)
    position = estimate.lever_arm_ins if use_lever_arm else np.zeros(3)

    models: Dict[str, StandardCamera] = {}
    for folder, camera in folder_camera.items():
        if folder == ref_folder:
            rig_from_sensor = Rotation.identity()
            cam_offset_rig = np.zeros(3)
        elif folder in sensor_from_rig:
            sfr = sensor_from_rig[folder]
            rig_from_sensor = Rotation.from_quat(sfr.rotation.quat).inv()
            cam_offset_rig = -sfr.rotation.matrix().T @ sfr.translation
        else:
            print(f"[yellow]No extrinsics for {folder}; skipping.")
            continue
        mount = ins_from_rig * rig_from_sensor
        if camera.model.name == "OPENCV":
            fx, fy, cx, cy, d1, d2, d3, d4 = camera.params
            dist = np.array([d1, d2, d3, d4])
        elif camera.model.name == "PINHOLE":
            fx, fy, cx, cy = camera.params
            dist = np.zeros(4)
        else:
            raise SystemError(f"Unexpected camera model {camera.model.name}")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        # cam_offset_rig is in rig coords; the mount position is in the INS
        # frame, so rotate rig->INS via ins_from_rig.
        cam_position = position + ins_from_rig.apply(cam_offset_rig)
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

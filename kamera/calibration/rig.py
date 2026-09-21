"""Turn the rigged reconstruction into the deliverables: per-camera models (in the INS frame),
the rig geometry, and the INS boresight with its per-frame residuals."""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field

import numpy as np
import pycolmap as pc
import yaml
from scipy.spatial.transform import Rotation

from kamera.calibration.flight import InsTrajectory
from kamera.calibration.sfm import robust_mean


@dataclass
class CameraCalibration:
    name: str
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    cam_from_rig: pc.Rigid3d
    colmap_params: dict
    frames: int
    reproj_rms_px: float

    @property
    def rig_from_cam(self) -> Rotation:
        return Rotation.from_quat(self.cam_from_rig.rotation.quat).inv()

    @property
    def center_in_rig(self) -> np.ndarray:
        return self.cam_from_rig.inverse().translation


@dataclass
class RigCalibration:
    rig: str
    flight: str
    reference: str
    cameras: dict[str, CameraCalibration]
    ins_from_rig: Rotation
    lever_arm_m: np.ndarray
    frame_times: np.ndarray
    rotation_residual_deg: (
        np.ndarray
    )  # (N, 3) rotvec of each frame's boresight about the mean, rig axes
    position_residual_m: (
        np.ndarray
    )  # (N, 3) rig origin relative to INS, body axes, minus the lever arm
    ins_gap_s: np.ndarray  # (N,) staleness of the INS sample behind each frame
    ground_speed_mps: float
    scene_range_m: float  # median distance from the reference camera to its 3D points
    inlier: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    def camera_quaternion(self, name: str) -> np.ndarray:
        """(x, y, z, w) rotating camera vectors into the INS body frame, the KAMERA yaml convention."""
        return (self.ins_from_rig * self.cameras[name].rig_from_cam).as_quat()

    def camera_position(self, name: str) -> np.ndarray:
        return self.lever_arm_m + self.ins_from_rig.apply(
            self.cameras[name].center_in_rig
        )

    def implied_delay_ms(self, name: str) -> float:
        """Exposure midpoint of a camera relative to the reference camera's, from its along-track offset.

        Positive means it exposes later than the reference. Only this relative timing is
        observable: the position priors absorb any delay common to the whole rig.
        """
        return (
            1000.0
            * float(self.ins_from_rig.apply(self.cameras[name].center_in_rig)[0])
            / self.ground_speed_mps
        )


def per_camera_reprojection(
    model: pc.Reconstruction, names: dict
) -> dict[str, list[float]]:
    errors: dict[str, list[float]] = {}
    for im in model.images.values():
        if not im.has_pose:
            continue
        cam = names[im.name][0]
        for p in im.points2D:
            if p.has_point3D():
                proj = im.project_point(model.points3D[p.point3D_id].xyz)
                if proj is not None:
                    errors.setdefault(cam, []).append(
                        float(np.linalg.norm(proj - p.xy))
                    )
    return errors


def calibrate_rig(
    model: pc.Reconstruction,
    names: dict,
    ins: InsTrajectory,
    reference: str,
    rig_name: str,
    flight: str,
) -> RigCalibration:
    """Read the rig geometry out of the reconstruction and solve the INS boresight over all frames."""
    rig = next(iter(model.rigs.values()))
    errors = per_camera_reprojection(model, names)
    cameras = {}
    frames_per_camera: dict[str, int] = {}
    for im in model.images.values():
        if im.has_pose:
            frames_per_camera[names[im.name][0]] = (
                frames_per_camera.get(names[im.name][0], 0) + 1
            )
    for im in model.images.values():
        name = names[im.name][0]
        if name in cameras or not im.has_pose:
            continue
        cam = model.cameras[im.camera_id]
        cam_from_rig = (
            pc.Rigid3d()
            if rig.is_ref_sensor(cam.sensor_id)
            else rig.sensor_from_rig(cam.sensor_id)
        )
        fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
        cameras[name] = CameraCalibration(
            name=name,
            width=cam.width,
            height=cam.height,
            K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            dist=np.array([k1, k2, p1, p2]),
            cam_from_rig=cam_from_rig,
            colmap_params={
                "model": cam.model_name,
                "params": [float(v) for v in cam.params],
            },
            frames=frames_per_camera[name],
            reproj_rms_px=float(
                np.sqrt(np.mean(np.square(errors.get(name, [np.nan]))))
            ),
        )

    times, ins_from_rig, lever, gaps = [], [], [], []
    for frame in model.frames.values():
        if not frame.has_pose():
            continue
        t = names[model.images[next(iter(frame.data_ids)).id].name][1]
        world_from_rig = frame.rig_from_world.inverse()
        pos, enu_from_body = ins.pose(t)
        times.append(t)
        ins_from_rig.append(
            (
                enu_from_body.inv() * Rotation.from_quat(world_from_rig.rotation.quat)
            ).as_quat()
        )
        lever.append(enu_from_body.inv().apply(world_from_rig.translation - pos))
        gaps.append(ins.sample_gap(t))
    order = np.argsort(times)
    rotations = Rotation.from_quat(np.array(ins_from_rig)[order])
    lever = np.array(lever)[order]
    mean_rot, mean_lever, _, keep = robust_mean(rotations, lever)
    positions = np.array([ins.pose(t)[0] for t in np.array(times)[order]])
    speed = float(
        np.median(
            np.linalg.norm(np.diff(positions, axis=0), axis=1)
            / np.diff(np.array(times)[order])
        )
    )
    ranges = [
        np.linalg.norm(model.points3D[p.point3D_id].xyz - im.projection_center())
        for im in model.images.values()
        if im.has_pose and names[im.name][0] == reference
        for p in im.points2D[::50]
        if p.has_point3D()
    ]
    return RigCalibration(
        rig=rig_name,
        flight=flight,
        reference=reference,
        cameras=cameras,
        ins_from_rig=mean_rot,
        lever_arm_m=mean_lever,
        frame_times=np.array(times)[order],
        rotation_residual_deg=(mean_rot.inv() * rotations).as_rotvec(degrees=True),
        position_residual_m=lever - mean_lever,
        ins_gap_s=np.array(gaps)[order],
        ground_speed_mps=speed,
        scene_range_m=float(np.median(ranges)),
        inlier=keep,
    )


def _floats(a) -> list[float]:
    return [float(v) for v in np.asarray(a).ravel()]


def write_camera_yaml(cal: RigCalibration, name: str, path: str) -> None:
    """KAMERA ``standard`` camera model plus rig and calibration provenance (loader ignores the extras)."""
    cam = cal.cameras[name]
    body = {
        "model_type": "standard",
        "image_width": int(cam.width),
        "image_height": int(cam.height),
        "fx": float(cam.K[0, 0]),
        "fy": float(cam.K[1, 1]),
        "cx": float(cam.K[0, 2]),
        "cy": float(cam.K[1, 2]),
        "distortion_coefficients": _floats(cam.dist),
        "camera_quaternion": _floats(cal.camera_quaternion(name)),
        "camera_position": _floats(cal.camera_position(name)),
        "camera_name": name,
        "channel": name.split("_")[0],
        "modality": name.split("_")[1],
        "rig": cal.rig,
        "reference_camera": cal.reference,
        "cam_from_rig": {
            "quaternion_xyzw": _floats(cam.cam_from_rig.rotation.quat),
            "translation_m": _floats(cam.cam_from_rig.translation),
        },
        "colmap_camera": cam.colmap_params,
        "calibration": {
            "flight": cal.flight,
            "generated": datetime.datetime.now(datetime.timezone.utc)
            .date()
            .isoformat(),
            "frames": cam.frames,
            "reprojection_rms_px": cam.reproj_rms_px,
            "ifov_deg": float(np.degrees(1.0 / cam.K[0, 0])),
        },
    }
    header = (
        "# KAMERA camera model. camera_quaternion (x, y, z, w) rotates camera vectors into the INS body\n"
        "# frame; camera_position is the camera centre in that frame (metres). distortion_coefficients\n"
        "# follow OpenCV (k1, k2, p1, p2). The extra keys record the rig calibration this came from.\n"
    )
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(body, f, sort_keys=False)


def write_rig_yaml(cal: RigCalibration, path: str) -> None:
    ref = cal.cameras[cal.reference]
    cams = {}
    for name, cam in cal.cameras.items():
        rel = ref.rig_from_cam.inv() * cam.rig_from_cam
        cams[name] = {
            "cam_from_rig": {
                "quaternion_xyzw": _floats(cam.cam_from_rig.rotation.quat),
                "translation_m": _floats(cam.cam_from_rig.translation),
            },
            "rotation_from_reference_deg": _floats(rel.as_rotvec(degrees=True)),
            "angle_from_reference_deg": float(np.degrees(rel.magnitude())),
            "centre_in_rig_m": _floats(cam.center_in_rig),
            "centre_in_ins_body_m": _floats(cal.ins_from_rig.apply(cam.center_in_rig)),
            "exposure_offset_from_reference_ms": cal.implied_delay_ms(name),
            "frames": cam.frames,
            "reprojection_rms_px": cam.reproj_rms_px,
        }
    keep = cal.inlier
    res = np.linalg.norm(cal.rotation_residual_deg[keep], axis=1)
    body = {
        "rig": cal.rig,
        "flight": cal.flight,
        "reference_camera": cal.reference,
        "generated": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        "ins_from_rig": {
            "quaternion_xyzw": _floats(cal.ins_from_rig.as_quat()),
            "rotvec_deg": _floats(cal.ins_from_rig.as_rotvec(degrees=True)),
            "euler_zyx_deg": _floats(cal.ins_from_rig.as_euler("ZYX", degrees=True)),
            "lever_arm_m": _floats(cal.lever_arm_m),
        },
        "flight_stats": {
            "ground_speed_mps": cal.ground_speed_mps,
            "scene_range_m": cal.scene_range_m,
        },
        "boresight_quality": {
            "frames": int(keep.sum()),
            "frames_rejected": int((~keep).sum()),
            "rotation_scatter_deg": {
                "median": float(np.median(res)),
                "p90": float(np.percentile(res, 90)),
                "max": float(res.max()),
            },
            "rotation_axis_std_deg": _floats(cal.rotation_residual_deg[keep].std(0)),
            "lever_arm_std_m": _floats(cal.position_residual_m[keep].std(0)),
            "ins_sample_gap_s": {
                "median": float(np.median(cal.ins_gap_s)),
                "max": float(cal.ins_gap_s.max()),
            },
        },
        "cameras": cams,
    }
    with open(path, "w") as f:
        f.write(
            "# Rig geometry (cam_from_rig maps rig -> camera, COLMAP convention) and INS boresight (ins_from_rig maps rig -> INS body).\n"
            "# A camera exposing later than the reference sits ahead along track by speed x delay; exposure_offset_from_reference_ms reads that off.\n"
        )
        yaml.safe_dump(body, f, sort_keys=False)


def write_outputs(cal: RigCalibration, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name in sorted(cal.cameras):
        paths.append(os.path.join(out_dir, f"{cal.rig}_{name}.yaml"))
        write_camera_yaml(cal, name, paths[-1])
    paths.append(os.path.join(out_dir, f"{cal.rig}_rig.yaml"))
    write_rig_yaml(cal, paths[-1])
    return paths

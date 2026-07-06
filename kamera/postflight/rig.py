"""Configure COLMAP rigs, frames, and pose priors for a KAMERA flight.

KAMERA cameras are hardware-synchronized and rigidly mounted, so a flight
maps directly onto COLMAP's (3.12+) rig data model:

- one ``Rig`` whose sensors are the per-camera image folders
  (``images0/<prefix>_<channel>_<modality>``),
- one ``Frame`` per trigger event, grouping the images that share the
  same ``<date>_<time>`` filename fields across all cameras,
- one position ``PosePrior`` per image, interpolated from the INS
  (``*_meta.json``) at the exposure time, in the same ENU frame the rest
  of postflight uses.

With these in the database, COLMAP's rig-aware bundle adjustment solves
``sensor_from_rig`` (the rig extrinsics) directly, and prior-position
mapping keeps the reconstruction unfragmented and geo-registered --
replacing the select-best-model + sim3-align + per-camera boresight
dance downstream.

COLMAP's own ``rig_configurator`` groups frames by identical basenames
after an image-prefix, which KAMERA names violate (channel and modality
are embedded in the basename), so this module writes rigs/frames
directly instead.
"""

import json
import os
import pathlib
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import pycolmap
from rich import print

from kamera.postflight.naming import KameraCameraName, KameraImageName
from kamera.sensor_models.nav_state import NavStateINSJson

__all__ = [
    "basename_to_time",
    "configure_rig_and_frames",
    "write_pose_priors",
    "run_rig_mapping",
]


def basename_to_time(flight_dir: str | os.PathLike) -> Dict[str, float]:
    """Map image base names (name sans modality/extension) to exposure
    times, from the ``*_meta.json`` files in a flight directory."""
    out = {}
    for json_fname in pathlib.Path(flight_dir).rglob("*_meta.json"):
        try:
            with open(json_fname) as f:
                d = json.load(f)
            out[KameraImageName.parse(json_fname).base_name] = float(d["evt"]["time"])
        except (OSError, ValueError, KeyError):
            continue
    return out


def _camera_sensor(camera_id: int) -> "pycolmap.sensor_t":
    return pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)


def _frame_key(image_name: str) -> str:
    n = KameraImageName.parse(image_name)
    return f"{n.date}_{n.time}"


def configure_rig_and_frames(
    db: "pycolmap.Database", ref_modality: str = "rgb"
) -> Tuple[int, int]:
    """Replace the database's (auto-generated, trivial) rigs and frames
    with a single rig of all cameras and one frame per trigger event.

    The reference sensor is the first (sorted) camera folder of
    `ref_modality`, falling back to the first folder overall.

    Returns (rig_id, num_frames).
    """
    images = db.read_all_images()
    if not images:
        raise SystemError("Database contains no images.")

    # Each images0 subfolder is one physical camera (single_camera_per_folder).
    folder_to_cam: Dict[str, int] = {}
    for im in images:
        folder = im.name.rsplit("/", 1)[0] if "/" in im.name else ""
        prev = folder_to_cam.setdefault(folder, im.camera_id)
        if prev != im.camera_id:
            raise SystemError(
                f"Folder {folder} maps to multiple cameras ({prev}, "
                f"{im.camera_id}); expected one camera per folder."
            )

    def _ref_rank(folder: str) -> Tuple[int, str]:
        try:
            is_ref = KameraCameraName.parse(folder).modality != ref_modality
        except ValueError:
            is_ref = True
        return (is_ref, folder)

    ordered = sorted(folder_to_cam, key=_ref_rank)
    print(f"Rig sensors ({len(ordered)}), ref = {ordered[0]}:")
    for f in ordered:
        print(f"  camera {folder_to_cam[f]}: {f}")

    db.clear_frames()
    db.clear_rigs()

    rig = pycolmap.Rig()
    rig.add_ref_sensor(_camera_sensor(folder_to_cam[ordered[0]]))
    for folder in ordered[1:]:
        # sensor_from_rig left unset; solved by rig-aware bundle adjustment
        rig.add_sensor(_camera_sensor(folder_to_cam[folder]), None)
    rig_id = db.write_rig(rig)

    groups = defaultdict(list)
    skipped = 0
    for im in images:
        try:
            groups[_frame_key(im.name)].append(im)
        except ValueError:
            skipped += 1
    if skipped:
        print(f"[yellow]{skipped} images had unparseable names; not in any frame.")

    for key in sorted(groups):
        frame = pycolmap.Frame()
        frame.rig_id = rig_id
        for im in groups[key]:
            frame.add_data_id(pycolmap.data_t(_camera_sensor(im.camera_id), im.image_id))
        db.write_frame(frame)

    sizes = [len(g) for g in groups.values()]
    print(
        f"Wrote rig {rig_id} and {len(groups)} frames "
        f"({min(sizes)}-{max(sizes)} images/frame)."
    )
    return rig_id, len(groups)


def write_pose_priors(
    db: "pycolmap.Database",
    flight_dir: str | os.PathLike,
    nav_state_provider: Optional[NavStateINSJson] = None,
    position_std: float = 2.0,
) -> int:
    """Write an ENU position prior for every image, interpolated from the
    INS at the exposure time. The ENU origin is the nav provider's
    (lat0, lon0), matching the frame postflight already works in.
    """
    if nav_state_provider is None:
        json_glob = pathlib.Path(flight_dir).rglob("*_meta.json")
        nav_state_provider = NavStateINSJson(json_glob)
    times = basename_to_time(flight_dir)
    covariance = np.eye(3) * position_std**2

    db.clear_pose_priors()
    written = skipped = 0
    for im in db.read_all_images():
        try:
            t = times[KameraImageName.parse(im.name).base_name]
        except (ValueError, KeyError):
            skipped += 1
            continue
        prior = pycolmap.PosePrior()
        prior.corr_data_id = pycolmap.data_t(_camera_sensor(im.camera_id), im.image_id)
        prior.position = np.asarray(nav_state_provider.pose(t)[0], dtype=float)
        prior.position_covariance = covariance
        prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        db.write_pose_prior(prior)
        written += 1
    print(f"Wrote {written} pose priors ({skipped} images without nav times).")
    return written


def run_rig_mapping(
    database_path: str | os.PathLike,
    image_path: str | os.PathLike,
    output_path: str | os.PathLike,
    use_priors: bool = True,
    mapper: str = "incremental",
) -> Dict[int, "pycolmap.Reconstruction"]:
    """Run mapping on a rig/prior-configured database.

    "incremental" honors both the rig constraints and the position
    priors. "global" (COLMAP's built-in GLOMAP-style mapper) is much
    faster but currently ignores priors -- use it for quick iteration,
    then refine with the incremental path or a prior-position bundle
    adjustment.
    """
    os.makedirs(output_path, exist_ok=True)
    if mapper == "global":
        options = pycolmap.GlobalPipelineOptions()
        return pycolmap.global_mapping(database_path, image_path, output_path, options)
    options = pycolmap.IncrementalPipelineOptions()
    options.use_prior_position = use_priors
    options.use_robust_loss_on_prior_position = True
    options.ba_refine_sensor_from_rig = True
    return pycolmap.incremental_mapping(database_path, image_path, output_path, options)

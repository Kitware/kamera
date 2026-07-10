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

import itertools
import json
import os
import pathlib
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycolmap
from rich import print
from scipy.spatial.transform import Rotation

from kamera.postflight.naming import KameraCameraName, KameraImageName
from kamera.sensor_models.nav_state import NavStateINSJson

__all__ = [
    "basename_to_time",
    "build_colmap_database",
    "configure_rig_and_frames",
    "derive_sensor_from_rig",
    "filter_same_trigger_matches",
    "write_frames",
    "write_pose_priors",
    "run_rig_mapping",
    "best_reconstruction",
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


def _folder_to_camera(images) -> Dict[str, int]:
    """Map each images0 subfolder to its (single) camera id."""
    folder_to_cam: Dict[str, int] = {}
    for im in images:
        folder = im.name.rsplit("/", 1)[0] if "/" in im.name else ""
        prev = folder_to_cam.setdefault(folder, im.camera_id)
        if prev != im.camera_id:
            raise SystemError(
                f"Folder {folder} maps to multiple cameras ({prev}, "
                f"{im.camera_id}); expected one camera per folder."
            )
    return folder_to_cam


def _order_by_ref(folders, ref_modality: str) -> List[str]:
    """Sorted camera folders with the reference-modality cameras first."""

    def rank(folder: str) -> Tuple[int, str]:
        try:
            is_ref = KameraCameraName.parse(folder).modality != ref_modality
        except ValueError:
            is_ref = True
        return (is_ref, folder)

    return sorted(folders, key=rank)


def write_frames(db: "pycolmap.Database", rig_id: int) -> int:
    """(Re)write one frame per trigger event, grouping the database's
    images by the date/time fields of their basenames. Returns the frame
    count."""
    groups = defaultdict(list)
    skipped = 0
    for im in db.read_all_images():
        try:
            groups[_frame_key(im.name)].append(im)
        except ValueError:
            skipped += 1
    if skipped:
        print(f"[yellow]{skipped} images had unparseable names; not in any frame.")

    db.clear_frames()
    for key in sorted(groups):
        frame = pycolmap.Frame()
        frame.rig_id = rig_id
        for im in groups[key]:
            frame.add_data_id(pycolmap.data_t(_camera_sensor(im.camera_id), im.image_id))
        db.write_frame(frame)
    sizes = [len(g) for g in groups.values()]
    print(
        f"Wrote {len(groups)} frames ({min(sizes)}-{max(sizes)} images/frame)."
    )
    return len(groups)


def configure_rig_and_frames(
    db: "pycolmap.Database",
    ref_modality: str = "rgb",
    sensor_from_rig: Optional[Dict[str, "pycolmap.Rigid3d"]] = None,
) -> Tuple[int, int]:
    """Replace the database's (auto-generated, trivial) rigs and frames
    with a single rig of all cameras and one frame per trigger event.

    The reference sensor is the first (sorted) camera folder of
    `ref_modality`, falling back to the first folder overall.

    `sensor_from_rig` optionally supplies the rig extrinsics per camera
    folder (from `derive_sensor_from_rig` on an initial reconstruction).
    When omitted the poses are left unset, which only the global mapper
    accepts; the incremental mapper requires them (run a first pass
    without a multi-sensor rig, derive, then re-map).

    Returns (rig_id, num_frames).
    """
    images = db.read_all_images()
    if not images:
        raise SystemError("Database contains no images.")

    folder_to_cam = _folder_to_camera(images)
    ordered = _order_by_ref(folder_to_cam, ref_modality)
    have_poses = sensor_from_rig is not None
    print(
        f"Rig sensors ({len(ordered)}), ref = {ordered[0]}, "
        f"extrinsics {'provided' if have_poses else 'unset'}:"
    )
    for f in ordered:
        print(f"  camera {folder_to_cam[f]}: {f}")

    db.clear_frames()
    db.clear_rigs()

    rig = pycolmap.Rig()
    rig.add_ref_sensor(_camera_sensor(folder_to_cam[ordered[0]]))
    for folder in ordered[1:]:
        pose = sensor_from_rig.get(folder) if sensor_from_rig else None
        rig.add_sensor(_camera_sensor(folder_to_cam[folder]), pose)
    rig_id = db.write_rig(rig)

    n_frames = write_frames(db, rig_id)
    print(f"Wrote rig {rig_id}.")
    return rig_id, n_frames


def derive_sensor_from_rig(
    reconstruction: "pycolmap.Reconstruction",
    ref_modality: str = "rgb",
    min_frames: int = 5,
) -> Dict[str, "pycolmap.Rigid3d"]:
    """Estimate each camera's sensor_from_rig from an initial (rigless)
    reconstruction, by robustly averaging, over synchronized frames,

        sensor_from_rig = sensor_cam_from_world . (ref_cam_from_world)^-1

    Rotations are averaged as unit quaternions; translations by median.
    Returns a map from camera folder to Rigid3d, excluding the reference.
    """
    from kamera.postflight.boresight import average_quaternions

    # group images by camera folder and by frame key
    by_folder: Dict[str, Dict[str, "pycolmap.Image"]] = defaultdict(dict)
    folders_present = set()
    for im in reconstruction.images.values():
        if not im.has_pose or "/" not in im.name:
            continue
        folder = im.name.rsplit("/", 1)[0]
        folders_present.add(folder)
        try:
            by_folder[folder][_frame_key(im.name)] = im
        except ValueError:
            continue

    ordered = _order_by_ref(folders_present, ref_modality)
    ref_folder = ordered[0]
    ref_by_key = by_folder[ref_folder]

    out: Dict[str, "pycolmap.Rigid3d"] = {}
    for folder in ordered[1:]:
        quats, trans = [], []
        for key, im in by_folder[folder].items():
            ref = ref_by_key.get(key)
            if ref is None:
                continue
            # sensor_from_rig = sensor_cam_from_world . world_from_ref
            rel = im.cam_from_world() * ref.cam_from_world().inverse()
            quats.append(rel.rotation.quat)
            trans.append(rel.translation)
        if len(quats) < min_frames:
            print(
                f"[yellow]Only {len(quats)} synchronized frames for {folder}; "
                "skipping (its extrinsics stay unset)."
            )
            continue
        q = average_quaternions(np.asarray(quats))
        t = np.median(np.asarray(trans), axis=0)
        spread = np.degrees(
            (
                Rotation.from_quat(np.asarray(quats)) * Rotation.from_quat(q).inv()
            ).magnitude()
        )
        print(
            f"  {folder}: sensor_from_rig from {len(quats)} frames, "
            f"rot spread {np.median(spread):.3f} deg"
        )
        out[folder] = pycolmap.Rigid3d(pycolmap.Rotation3d(q), t)
    return out


def filter_same_trigger_matches(database_path: str | os.PathLike) -> int:
    """Empty the matches between images captured at the same trigger.

    Co-located cameras fire simultaneously, so same-trigger pairs are
    the strongest matches in the database (same scene, same instant) but
    have ~zero baseline: their correspondences can never triangulate,
    and incremental mapping -- which ranks candidate pairs by match
    count -- keeps initializing and growing through them, fragmenting
    the reconstruction. The rig geometry these pairs describe is
    recovered downstream from the per-image poses instead
    (``derive_sensor_from_rig``), so mapping loses nothing.

    The pairs are overwritten with empty match/geometry entries rather
    than deleted, so a re-run of the matcher skips them instead of
    faithfully recreating them. Returns the number of pairs emptied.
    """
    empty = np.zeros((0, 2), dtype=np.uint32)
    emptied = 0
    db = pycolmap.Database.open(database_path)
    try:
        by_trigger: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for im in db.read_all_images():
            try:
                name = KameraImageName.parse(im.name)
            except ValueError:
                continue
            by_trigger[(name.date, name.time)].append(im.image_id)
        for ids in by_trigger.values():
            for id1, id2 in itertools.combinations(sorted(ids), 2):
                if not db.exists_matches(id1, id2):
                    continue
                if len(db.read_matches(id1, id2)) == 0:
                    continue  # already emptied by a previous run
                db.delete_matches(id1, id2)
                db.write_matches(id1, id2, empty)
                if db.exists_two_view_geometry(id1, id2):
                    db.delete_two_view_geometry(id1, id2)
                geom = pycolmap.TwoViewGeometry()
                geom.inlier_matches = empty
                db.write_two_view_geometry(id1, id2, geom)
                emptied += 1
    finally:
        db.close()
    if emptied:
        print(f"Emptied {emptied} same-trigger (zero-baseline) match pairs.")
    return emptied


def build_colmap_database(
    database_path: str | os.PathLike,
    image_path: str | os.PathLike,
    flight_dir: str | os.PathLike,
    nav_state_provider: Optional[NavStateINSJson] = None,
    position_std: float = 2.0,
    matcher: str = "spatial",
    filter_same_trigger: bool = True,
) -> None:
    """Extract SIFT features and match images0 into a COLMAP database,
    replacing the docker feature_extractor/matcher scripts.

    One OPENCV camera per image folder, KAMERA-tuned SIFT settings, and
    -- for the default spatial matcher -- INS position priors so pairs
    are restricted to spatial neighbors instead of the full O(N^2) set.
    Same-trigger pairs are then emptied (see
    ``filter_same_trigger_matches``) unless `filter_same_trigger` is
    False.
    """
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    reader.default_focal_length_factor = 1.2

    extraction = pycolmap.FeatureExtractionOptions()
    # The default (-1) spawns one worker per core, each holding a
    # full-resolution decode of these very large images -- tens of GB of
    # transient RAM on a many-core machine. A few feeder threads already
    # saturate the GPU.
    extraction.num_threads = min(8, os.cpu_count() or 8)
    extraction.max_image_size = 3200
    extraction.sift.max_num_features = 8192
    extraction.sift.first_octave = -1
    extraction.sift.num_octaves = 11
    extraction.sift.octave_resolution = 3
    extraction.sift.peak_threshold = 0.02 / 3
    extraction.sift.edge_threshold = 10
    extraction.sift.max_num_orientations = 2

    print(f"Extracting features from {image_path} (one OPENCV camera per folder).")
    pycolmap.extract_features(
        database_path,
        image_path,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        reader_options=reader,
        extraction_options=extraction,
    )

    if matcher == "spatial":
        # Priors give the spatial matcher the image locations it pairs on.
        db = pycolmap.Database.open(database_path)
        try:
            write_pose_priors(db, flight_dir, nav_state_provider, position_std)
        finally:
            db.close()
        pairing = pycolmap.SpatialPairingOptions()
        pairing.max_num_neighbors = 50
        pairing.ignore_z = False
        print("Spatial matching on INS priors.")
        pycolmap.match_spatial(database_path, pairing_options=pairing)
    elif matcher == "exhaustive":
        print("Exhaustive matching.")
        pycolmap.match_exhaustive(database_path)
    else:
        raise ValueError(f"Unknown matcher '{matcher}'.")

    if filter_same_trigger:
        filter_same_trigger_matches(database_path)


def write_pose_priors(
    db: "pycolmap.Database",
    flight_dir: str | os.PathLike,
    nav_state_provider: Optional[NavStateINSJson] = None,
    position_std: float = 2.0,
) -> int:
    """Write an ENU position prior for every image, interpolated from the
    INS at the exposure time. The ENU origin is the nav provider's
    (lat0, lon0), matching the frame postflight already works in.

    Every image MUST get a finite prior: COLMAP's spatial pair generator
    sizes its position matrix by image count but only fills rows for
    images with finite priors, then mean-centers over the whole matrix --
    one prior-less image poisons every position with uninitialized memory
    and the matcher aborts. Images whose ``*_meta.json`` is missing fall
    back to the exposure time embedded in the filename (the archiver
    renders ``<date>_<time>`` from the event time in UTC, so inverting it
    recovers the same timestamp).
    """
    if nav_state_provider is None:
        json_glob = pathlib.Path(flight_dir).rglob("*_meta.json")
        nav_state_provider = NavStateINSJson(json_glob)
    times = basename_to_time(flight_dir)
    covariance = np.eye(3) * position_std**2

    db.clear_pose_priors()
    written = fallback = 0
    for im in db.read_all_images():
        name = KameraImageName.parse(im.name)
        try:
            t = times[name.base_name]
        except KeyError:
            fmt = "%Y%m%d_%H%M%S.%f" if "." in name.time else "%Y%m%d_%H%M%S"
            dt = datetime.strptime(f"{name.date}_{name.time}", fmt)
            t = dt.replace(tzinfo=timezone.utc).timestamp()
            fallback += 1
        position = np.asarray(nav_state_provider.pose(t)[0], dtype=float)
        if not np.all(np.isfinite(position)):
            raise ValueError(
                f"Non-finite INS position {position} for image {im.name} "
                f"(t={t}); a non-finite prior would crash spatial matching."
            )
        prior = pycolmap.PosePrior()
        prior.corr_data_id = pycolmap.data_t(_camera_sensor(im.camera_id), im.image_id)
        prior.position = position
        prior.position_covariance = covariance
        prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        db.write_pose_prior(prior)
        written += 1
    print(
        f"Wrote {written} pose priors "
        f"({fallback} from filename times; no meta.json)."
    )
    return written


def _incremental_options(use_priors: bool) -> "pycolmap.IncrementalPipelineOptions":
    options = pycolmap.IncrementalPipelineOptions()
    options.use_prior_position = use_priors
    options.use_robust_loss_on_prior_position = True
    options.ba_refine_sensor_from_rig = True
    return options


def best_reconstruction(
    recs: Dict[int, "pycolmap.Reconstruction"],
) -> "pycolmap.Reconstruction":
    """The reconstruction with the most registered images. Incremental
    mapping always emits scrap sub-models alongside the main one."""
    return max(recs.values(), key=lambda r: r.num_reg_images())


def run_rig_mapping(
    database_path: str | os.PathLike,
    image_path: str | os.PathLike,
    output_path: str | os.PathLike,
    use_priors: bool = True,
    mapper: str = "incremental",
) -> Dict[int, "pycolmap.Reconstruction"]:
    """Prior-position map a database into an ENU model.

    The rig is deliberately NOT imposed on the mapper: rig-constrained
    incremental mapping from scratch fragments badly (initializing rigid
    multi-camera frames is far harder than individual images -- on fl09
    it split into 50 sub-models, largest ~100 images vs ~1450 rigless).
    The rig geometry is recovered afterward, in closed form, by
    `derive_sensor_from_rig` and the boresight solve, both of which only
    need this per-image ENU model. So mapping stays a single rigless
    pass.

    "global" (COLMAP's GLOMAP-style mapper) additionally ignores the
    priors, giving an arbitrary-gauge model; it is only a fast sanity
    check, never the ENU source for calibration.
    """
    # Mapping always starts over, so clear the numbered submodels of any
    # previous run -- a stale sparse/3 next to a fresh sparse/0-2 would
    # misrepresent this run's output.
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    if mapper == "global":
        options = pycolmap.GlobalPipelineOptions()
        return pycolmap.global_mapping(database_path, image_path, output_path, options)

    # Rigless so the mapper registers individual images; the priors fix
    # the ENU gauge.
    db = pycolmap.Database.open(database_path)
    try:
        db.clear_rigs()
        db.clear_frames()
    finally:
        db.close()
    print("[blue]Prior-position mapping (rigless).[/blue]")
    return pycolmap.incremental_mapping(
        database_path, image_path, output_path, _incremental_options(use_priors)
    )

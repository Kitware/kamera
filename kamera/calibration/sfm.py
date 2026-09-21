"""Structure from motion with pycolmap: features, INS position priors, matching, and the
two mapping passes (trivial rigs to bootstrap, then the full multi-sensor rig)."""

from __future__ import annotations

import os
import shutil

import cv2
import numpy as np
import PIL.Image
import pycolmap as pc
from scipy.spatial.transform import Rotation

CAMERA_MODEL = "OPENCV"
# Only image headers are read here; the 100 MP RGB frames trip PIL's default bomb limit.
PIL.Image.MAX_IMAGE_PIXELS = None


def device() -> pc.Device:
    return pc.Device.cuda if pc.has_cuda else pc.Device.cpu


def extract_features(
    db_path: str,
    image_dir: str,
    names: dict,
    focal_px: dict,
    max_image_size: int,
    num_features: int,
) -> None:
    """SIFT per camera folder, seeding each camera with its modality's focal length."""
    for camera in sorted({c for c, _ in names.values()}):
        image_names = sorted(n for n in names if n.startswith(camera + "/"))
        w, h = PIL.Image.open(os.path.join(image_dir, image_names[0])).size
        f = focal_px[camera.split("_")[1]]
        reader = pc.ImageReaderOptions(
            camera_model=CAMERA_MODEL, camera_params=f"{f},{f},{w / 2},{h / 2},0,0,0,0"
        )
        # Each thread decodes a full-resolution image; large sensors get fewer threads.
        opts = pc.FeatureExtractionOptions(
            max_image_size=max_image_size,
            use_gpu=pc.has_cuda,
            num_threads=4 if w * h > 40e6 else 16,
        )
        opts.sift.max_num_features = num_features
        pc.extract_features(
            db_path,
            image_dir,
            image_names=image_names,
            camera_mode=pc.CameraMode.PER_FOLDER,
            reader_options=reader,
            extraction_options=opts,
            device=device(),
        )


def write_pose_priors(db_path: str, names: dict, ins, std_m: float) -> None:
    """Attach the INS ENU position at each image's trigger time as a pose prior."""
    db = pc.Database.open(db_path)
    for image in db.read_all_images():
        prior = pc.PosePrior(
            position=ins.pose(names[image.name][1])[0],
            position_covariance=np.eye(3) * std_m**2,
            coordinate_system=pc.PosePriorCoordinateSystem.CARTESIAN,
        )
        prior.corr_data_id = pc.data_t(
            pc.sensor_t(pc.SensorType.CAMERA, image.camera_id), image.image_id
        )
        db.write_pose_prior(prior)
    db.close()


def match_features(db_path: str, max_distance_m: float, max_neighbors: int) -> None:
    """Match each image against its spatial neighbours (from the priors).

    Pairs are formed across all cameras.
    """
    pairing = pc.SpatialPairingOptions(
        max_num_neighbors=max_neighbors, max_distance=max_distance_m, ignore_z=True
    )
    pc.match_spatial(
        db_path,
        matching_options=pc.FeatureMatchingOptions(use_gpu=pc.has_cuda),
        pairing_options=pairing,
        device=device(),
    )
    prune_cross_spectral(db_path)


def prune_cross_spectral(db_path: str) -> int:
    """Drop thermal-to-visible pairs.

    SIFT cannot match them, so their few 'inliers' only mislead the mapper.
    """
    db = pc.Database.open(db_path)
    is_ir = {
        im.image_id: im.name.split("/")[0].endswith("_ir")
        for im in db.read_all_images()
    }
    pair_ids, _ = db.read_two_view_geometries()
    dropped = 0
    for pair_id in pair_ids:
        i, j = pc.pair_id_to_image_pair(pair_id)
        if is_ir[i] != is_ir[j]:
            db.delete_matches(i, j)
            db.delete_two_view_geometry(i, j)
            dropped += 1
    db.close()
    return dropped


def mapping_options(refine_rig: bool) -> pc.IncrementalPipelineOptions:
    # Colours are unused and extracting them re-decodes every 100 MP frame.
    opts = pc.IncrementalPipelineOptions(
        use_prior_position=True,
        ba_refine_sensor_from_rig=refine_rig,
        extract_colors=False,
        # Distortion cannot be recovered from two or three views of flat ground: on
        # the May 2025 flight, refining it from the initial pair drove L_ir to a 30%
        # focal error and k2 of -3, so no L_ir model ever grew past three images.
        # Pass 2 refines the full intrinsics once the whole rig is posed.
        ba_refine_extra_params=False,
    )
    # Nadir aerial pairs subtend small angles; the default 16 deg init threshold
    # rejects them.
    opts.mapper.init_min_tri_angle = 4.0
    # Global BA every 30% of growth instead of 10%: it dominates runtime on thousands
    # of frames.
    opts.ba_global_frames_ratio = opts.ba_global_points_ratio = 1.3
    opts.ba_global_max_refinements = 2
    return opts


def run_mapping(
    db_path: str, image_dir: str, out_dir: str
) -> dict[int, pc.Reconstruction]:
    """Incremental mapping from scratch with every camera independent (trivial rigs)."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    return pc.incremental_mapping(
        db_path, image_dir, out_dir, options=mapping_options(refine_rig=False)
    )


def rig_bundle_adjust(
    model: pc.Reconstruction, priors: list, refine_intrinsics: bool, max_iterations: int
) -> str:
    """Refine rig poses, sensor_from_rig and optionally intrinsics.

    Anchored to the INS position priors.
    """
    opts = pc.BundleAdjustmentOptions(
        refine_sensor_from_rig=True,
        refine_rig_from_world=True,
        refine_principal_point=False,
        refine_focal_length=refine_intrinsics,
        refine_extra_params=refine_intrinsics,
        print_summary=False,
    )
    opts.ceres.solver_options.max_num_iterations = max_iterations
    config = pc.BundleAdjustmentConfig()
    for image in model.images.values():
        if image.has_pose:
            config.add_image(image.image_id)
    prior_opts = pc.PosePriorBundleAdjustmentOptions()
    prior_opts.alignment_ransac.max_error = 5.0
    summary = pc.create_pose_prior_bundle_adjuster(
        opts, prior_opts, config, priors, model
    ).solve()
    model.update_point_3d_errors()
    return summary.brief_report()


def refine_rig(
    db_path: str, names: dict, init_dir: str, out_dir: str, max_iterations: int = 200
) -> pc.Reconstruction:
    """Pass 2: triangulate every image from the rig poses, bundle adjust, retriangulate,
    and bundle adjust again with the intrinsics free.

    Returns the final model, also written to ``out_dir``.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    # The triangulator always colours points from disk; 8x8 stand-ins spare it the
    # 100 MP frames.
    image_dir = os.path.join(os.path.dirname(out_dir), "placeholders")
    for name in names:
        os.makedirs(os.path.dirname(os.path.join(image_dir, name)), exist_ok=True)
        cv2.imwrite(os.path.join(image_dir, name), np.zeros((8, 8, 3), np.uint8))
    db = pc.Database.open(db_path)
    priors = db.read_all_pose_priors()
    db.close()
    opts = mapping_options(refine_rig=True)
    model = pc.Reconstruction(init_dir)
    for refine_intrinsics in (False, True):
        # Intrinsics are only ever refined in the rig bundle adjustment below, never by
        # the triangulator.
        model = pc.triangulate_points(
            model,
            db_path,
            image_dir,
            out_dir,
            clear_points=True,
            options=opts,
            refine_intrinsics=False,
        )
        print(
            f"  triangulated {model.num_points3D()} points, "
            f"{model.compute_mean_reprojection_error():.2f} px",
            flush=True,
        )
        print(
            f"  {rig_bundle_adjust(model, priors, refine_intrinsics, max_iterations)}",
            flush=True,
        )
    model.write(out_dir)
    return model


def load_models(out_dir: str) -> dict[int, pc.Reconstruction]:
    return {
        int(d): pc.Reconstruction(os.path.join(out_dir, d))
        for d in sorted(os.listdir(out_dir))
        if d.isdigit()
    }


def image_poses(
    models: dict[int, pc.Reconstruction], names: dict
) -> dict[tuple[str, float], pc.Rigid3d]:
    """``{(camera, time): cam_from_world}`` over every posed image in every model.

    All models are in INS ENU.
    """
    return {
        names[im.name]: im.cam_from_world()
        for r in models.values()
        for im in r.images.values()
        if im.has_pose
    }


def robust_mean(
    rotations: Rotation, translations: np.ndarray, cluster_deg: float = 1.0
) -> tuple[Rotation, np.ndarray, np.ndarray, np.ndarray]:
    """Mean rotation of the densest cluster and median translation over its members.

    Seeds from the sample with the most neighbours within ``cluster_deg``, so a wrongly
    registered majority (a folded sub-model) cannot drag the estimate; then keeps
    everything within 3x that cluster's median residual. Returns mean, translation,
    per-sample residual angles (deg) and the inlier mask.
    """
    q = rotations.as_quat()
    pairwise = np.degrees(2.0 * np.arccos(np.clip(np.abs(q @ q.T), 0.0, 1.0)))
    keep = pairwise[np.argmax((pairwise < cluster_deg).sum(1))] < cluster_deg
    mean = rotations[keep].mean()
    angles = np.degrees((mean.inv() * rotations).magnitude())
    keep = angles <= max(3.0 * np.median(angles[keep]), 0.05)
    mean = rotations[keep].mean()
    return (
        mean,
        np.median(translations[keep], 0),
        np.degrees((mean.inv() * rotations).magnitude()),
        keep,
    )


def derive_rig(
    models: dict[int, pc.Reconstruction], names: dict, reference: str
) -> dict[str, dict]:
    """Initial ``cam_from_rig`` per camera from frames shared with the reference."""
    poses = image_poses(models, names)
    rig = {}
    for camera in sorted({c for c, _ in names.values()}):
        rel = [
            poses[(camera, t)] * poses[(reference, t)].inverse()
            for (c, t) in poses
            if c == camera and (reference, t) in poses
        ]
        if len(rel) < 3:
            raise RuntimeError(
                f"{camera}: only {len(rel)} frames shared with {reference}; "
                "cannot initialise the rig"
            )
        rotations = Rotation.from_quat([x.rotation.quat for x in rel])
        translations = np.array([x.translation for x in rel])
        rot, trans, angles, keep = robust_mean(rotations, translations)
        rig[camera] = {
            "cam_from_rig": pc.Rigid3d(pc.Rotation3d(rot.as_quat()), trans),
            "frames": int(keep.sum()),
            "frames_total": len(rel),
            "rotation_scatter_deg": float(np.median(angles[keep])),
            "translation_std_m": translations[keep].std(0),
        }
    return rig


def model_cameras(
    models: dict[int, pc.Reconstruction], names: dict
) -> dict[str, pc.Camera]:
    """Refined intrinsics per camera name from whichever model registered it."""
    cams = {}
    for r in models.values():
        for im in r.images.values():
            if im.has_pose:
                cams.setdefault(names[im.name][0], r.cameras[im.camera_id])
    return cams


def rigged_model(
    db_path: str,
    models: dict[int, pc.Reconstruction],
    names: dict,
    reference: str,
    out_dir: str,
) -> pc.Reconstruction:
    """Put the rig onto the largest pass-1 model and fill its frames with every image.

    Writes the rig and frames into the database, copies each registered frame's pose
    from the model, and adds the images (IR, typically) that pass 1 never posed: they
    inherit their pose from the frame through the initial ``cam_from_rig``. The result
    is written to ``out_dir`` as the starting point for the rig-refining mapping pass.
    """
    rig = derive_rig(models, names, reference)
    cameras = model_cameras(models, names)
    config = pc.RigConfig(
        cameras=[
            pc.RigConfigCamera(
                ref_sensor=(name == reference),
                image_prefix=name + "/",
                camera=cameras[name],
                cam_from_rig=None if name == reference else rig[name]["cam_from_rig"],
            )
            for name in [reference] + sorted(set(rig) - {reference})
        ]
    )
    model = largest(models)
    for cam in cameras.values():
        if not model.exists_camera(cam.camera_id):
            model.add_camera(cam)
    db = pc.Database.open(db_path)
    db.clear_frames()
    db.clear_rigs()
    pc.apply_rig_config([config], db, model)
    poses = {
        f.frame_id: f.rig_from_world for f in model.frames.values() if f.has_pose()
    }
    frames = db.read_all_frames()
    for frame in frames:
        if frame.frame_id in poses:
            frame.rig_from_world = poses[frame.frame_id]
    model.set_rigs_and_frames(db.read_all_rigs(), frames)
    for image in db.read_all_images():
        if not model.exists_image(image.image_id):
            model.add_image(image)
    db.close()
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    model.write(out_dir)
    return model


def largest(models: dict[int, pc.Reconstruction]) -> pc.Reconstruction:
    return max(models.values(), key=lambda r: r.num_reg_images())

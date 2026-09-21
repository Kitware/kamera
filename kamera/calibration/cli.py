"""``kamera-calibrate``: run the calibration pipeline stage by stage, resumable."""

from __future__ import annotations

import json
import os
import shutil
import sys

import cv2
import numpy as np
import pycolmap as pc
from rich import print

from kamera.calibration import registration, rig, sfm
from kamera.calibration.config import CalibrateConfig
from kamera.calibration.flight import build_image_tree, discover_flight
from kamera.calibration.report import write_report
from kamera.colmap_processing.camera_models import StandardCamera

# Homography pairs per channel, left -> right (DIVE registers the left onto the right).
# Only the pairs DIVE uses; ir->uv follows from the other two and only adds noise.
PAIRS = [("ir", "rgb"), ("uv", "rgb")]

# A rig seed is trusted only when the per-frame estimates behind it agree. The rig
# bundle adjustment drops tracks over 4 px of reprojection error (about 0.13 deg for
# the IR cameras), so a seed a degree off stalls near the seed instead of converging.
SEED_MAX_SCATTER_DEG = 0.5
SEED_MIN_CLUSTER_FRACTION = 0.5


def write_gifs(frames, names, image_dir, left, right, h, gif_dir, count) -> dict:
    """Flip GIFs of the left image warped onto the right, for evenly spaced frames.

    Returns the report images (warped, right, overlay) from the middle frame, or an
    empty dict when no frame has both images or ``count`` is 0.
    """
    os.makedirs(gif_dir, exist_ok=True)
    # Both sides come from the normalized tree: the raw UV frames are nearly black.
    by_time = {(c, t): n for n, (c, t) in names.items() if c in (left, right)}
    usable = [f for f in frames if left in f.images and right in f.images]
    out = {}
    chosen = usable[:: max(1, len(usable) // max(count, 1))][:count]
    for k, frame in enumerate(chosen):
        left_img, right_img = (
            cv2.imread(
                os.path.join(image_dir, by_time[(camera, frame.time)]),
                cv2.IMREAD_COLOR,
            )
            for camera in (left, right)
        )
        warped, ref = registration.warp_pair(left_img, right_img, h)
        registration.write_gif(
            os.path.join(gif_dir, f"{left}_to_{right}_{k}.gif"), warped, ref
        )
        if k == len(chosen) // 2:
            out = {
                "warped_img": warped,
                "right_img": ref,
                "overlay_img": registration.blend_overlay(warped, ref),
            }
    return out


def main(argv=None) -> None:
    cfg = CalibrateConfig.cli(argv=argv, strict=True)
    work = cfg.work_dir or os.path.join(cfg.flight_dir, "calibration")
    image_dir, db_path = os.path.join(work, "images"), os.path.join(work, "database.db")
    pass1_dir, rig_dir, camera_model_dir = (
        os.path.join(work, "pass1"),
        os.path.join(work, "rig"),
        os.path.join(work, "camera_models"),
    )
    os.makedirs(work, exist_ok=True)

    def done(path: str) -> bool:
        return os.path.exists(path) and not cfg.force

    def staging(path: str) -> str:
        """A clean scratch path for a stage. Stages build there and ``publish`` moves
        the result into place, so ``path`` only ever exists once its stage finished
        and an interrupted run redoes the stage instead of skipping it."""
        tmp = path + ".partial"
        for stale in (tmp, tmp + "-wal", tmp + "-shm", tmp + "-journal"):
            if os.path.isdir(stale):
                shutil.rmtree(stale)
            elif os.path.exists(stale):
                os.remove(stale)
        return tmp

    def publish(tmp: str, path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.replace(tmp, path)

    print("[blue]Discovering frames[/blue]")
    frames, ins, rig_name = discover_flight(cfg.flight_dir)
    rig_name = cfg.rig_name or rig_name.replace("images_", "") or "rig"
    all_cameras = {camera for frame in frames for camera in frame.images}
    full = [f for f in frames if len(f.images) == len(all_cameras)]
    stop = (
        cfg.frame_start + cfg.max_frames * cfg.frame_stride if cfg.max_frames else None
    )
    frames = full[cfg.frame_start : stop : cfg.frame_stride]
    median_gap_ms = 1000 * np.median([ins.sample_gap(f.time) for f in frames])
    print(
        f"{len(full)} frames with every camera, using {len(frames)}; "
        f"INS median sample gap {median_gap_ms:.1f} ms"
    )
    names = build_image_tree(frames, image_dir)
    with open(os.path.join(work, "images.json"), "w") as f:
        json.dump(names, f)

    if not done(db_path):
        print("[blue]Extracting features and writing INS priors[/blue]")
        tmp = staging(db_path)
        sfm.extract_features(
            tmp,
            image_dir,
            names,
            cfg.focal_px,
            cfg.distortion,
            cfg.max_image_size,
            cfg.num_features,
        )
        sfm.write_pose_priors(tmp, names, ins, cfg.prior_std_m)
        print("[blue]Matching[/blue]")
        sfm.match_features(tmp, cfg.match_distance_m, cfg.match_neighbors)
        publish(tmp, db_path)

    if not done(pass1_dir):
        print("[blue]Pass 1: mapping with independent cameras[/blue]")
        tmp = staging(pass1_dir)
        sfm.run_mapping(db_path, image_dir, tmp)
        publish(tmp, pass1_dir)
    pass1 = sfm.load_models(pass1_dir)
    for k, r in pass1.items():
        print(
            f"  model {k}: {r.num_reg_images()} images, {r.num_points3D()} points, "
            f"{r.compute_mean_reprojection_error():.2f} px"
        )

    if not done(rig_dir):
        print("[blue]Pass 2: rig bundle adjustment[/blue]")
        for name, v in sfm.derive_rig(pass1, names, cfg.reference_camera).items():
            fraction = v["frames"] / v["frames_total"]
            print(
                f"  {name}: {v['frames']}/{v['frames_total']} frames in cluster, "
                f"rotation scatter {v['rotation_scatter_deg']:.3f} deg, "
                f"translation std {np.round(v['translation_std_m'], 2)} m"
            )
            if (
                v["rotation_scatter_deg"] > SEED_MAX_SCATTER_DEG
                or fraction < SEED_MIN_CLUSTER_FRACTION
            ):
                print(
                    f"  [yellow]{name}: rig seed is unreliable (scatter over "
                    f"{SEED_MAX_SCATTER_DEG} deg or under "
                    f"{SEED_MIN_CLUSTER_FRACTION:.0%} of frames in the cluster). "
                    "Pass 2 may stall near this seed: check its observation count "
                    "below and its registration GIFs.[/yellow]"
                )
        rig_in = os.path.join(work, "rig_init")
        sfm.rigged_model(db_path, pass1, names, cfg.reference_camera, rig_in)
        tmp = staging(rig_dir)
        sfm.refine_rig(db_path, names, rig_in, tmp)
        publish(tmp, rig_dir)
    model = pc.Reconstruction(rig_dir)
    print(
        f"  rig model: {model.num_reg_frames()} frames, "
        f"{model.num_reg_images()} images, "
        f"{model.compute_mean_reprojection_error():.2f} px"
    )

    print("[blue]Extracting camera models and boresight[/blue]")
    cal = rig.calibrate_rig(
        model,
        names,
        ins,
        cfg.reference_camera,
        rig_name,
        os.path.basename(os.path.abspath(cfg.flight_dir)),
    )
    for name in sorted(cal.cameras):
        c = cal.cameras[name]
        print(
            f"  {name}: {c.frames} frames, {c.observations} observations, "
            f"{c.reproj_rms_px:.2f} px rms"
        )
    for p in rig.write_outputs(cal, camera_model_dir):
        print(f"  wrote {p}")
    # <sys_cfg>/<view>_view/<image>: postflight reads <sys_cfg>/sys_config.json.
    config_dirs = {
        os.path.dirname(os.path.dirname(p)) for f in frames for p in f.images.values()
    }
    for p in rig.write_sys_configs(
        cal, camera_model_dir, sorted(config_dirs), cfg.install_sys_config
    ):
        print(f"  wrote {p}")

    print("[blue]Fitting homographies and writing DIVE registration files[/blue]")
    reg_dir = os.path.join(camera_model_dir, "dive_registration")
    cams = {
        n: StandardCamera(
            c.width,
            c.height,
            c.K,
            c.dist,
            cal.camera_position(n),
            cal.camera_quaternion(n),
        )
        for n, c in cal.cameras.items()
    }
    range_m = cfg.registration_range_m or cal.scene_range_m
    registered = {names[im.name][1] for im in model.images.values() if im.has_pose}
    gif_frames = [f for f in frames if f.time in registered]
    gif_dir = os.path.join(camera_model_dir, "gifs")
    print(
        f"  homographies exact at {range_m:.0f} m range; "
        f"ground speed {cal.ground_speed_mps:.0f} m/s"
    )
    pairs = []
    for channel in sorted({n.split("_")[0] for n in cams}):
        for left_mod, right_mod in PAIRS:
            left, right = f"{channel}_{left_mod}", f"{channel}_{right_mod}"
            if left not in cams or right not in cams:
                continue
            try:
                h, stats = registration.model_homography(
                    cams[left], cams[right], range_m
                )
            except ValueError as e:
                print(f"  [yellow]{left} -> {right}: {e}[/yellow]")
                continue
            source = registration.source_stamp(cfg.flight_dir, {"rig": rig_name})
            path = registration.write_dive_registration(
                reg_dir, left, right, h, stats, source
            )
            print(f"  wrote {path}  (fit rms {stats['rmsPx']:.2f} px)")
            images = write_gifs(
                gif_frames, names, image_dir, left, right, h, gif_dir, cfg.gif_frames
            )
            pairs.append(
                {"left": left, "right": right, "h": h, "stats": stats, **images}
            )

    report_path = os.path.join(camera_model_dir, f"{rig_name}_calibration_report.pdf")
    write_report(report_path, cal, pairs)
    print(f"[green]Report written to {report_path}[/green]")


if __name__ == "__main__":
    main(sys.argv[1:])

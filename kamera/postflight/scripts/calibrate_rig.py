"""Calibrate a KAMERA rig from a flight: rig+prior mapping, one boresight
solve, and per-camera StandardCamera export.

Assumes an existing feature-extracted and matched COLMAP database (from
the feature_extractor / matcher steps) under ``<flight>/colmap_rgb``.

Example:
    python calibrate_rig.py /data/fl09_85mm_25_5deg
"""

import argparse
import os
import pathlib
import shutil

import pycolmap
from rich import print

from kamera.postflight.boresight import (
    export_rig_camera_models,
    solve_rig_boresight,
)
from kamera.postflight.rig import (
    basename_to_time,
    configure_rig_and_frames,
    run_rig_mapping,
    write_pose_priors,
)
from kamera.sensor_models.nav_state import NavStateINSJson


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("flight_dir")
    p.add_argument("--colmap-dir", default=None)
    p.add_argument("--workspace", default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--ref-modality", default="rgb")
    p.add_argument("--prior-std", type=float, default=2.0)
    p.add_argument(
        "--reuse-model",
        default=None,
        help="Path to an existing rig-constrained, ENU sparse model; "
        "skips mapping and only runs the boresight + export.",
    )
    args = p.parse_args()

    flight_dir = args.flight_dir
    colmap_dir = args.colmap_dir or os.path.join(flight_dir, "colmap_rgb")
    workspace = args.workspace or os.path.join(flight_dir, "colmap_rig")
    save_dir = args.save_dir or os.path.join(flight_dir, "kamera_models")
    image_path = os.path.join(colmap_dir, "images0")

    nav = NavStateINSJson(pathlib.Path(flight_dir).rglob("*_meta.json"))
    times = basename_to_time(flight_dir)

    if args.reuse_model:
        reconstruction = pycolmap.Reconstruction(args.reuse_model)
    else:
        os.makedirs(workspace, exist_ok=True)
        dst_db = os.path.join(workspace, "database.db")
        if not os.path.exists(dst_db):
            print(f"Copying database into {dst_db}")
            shutil.copyfile(os.path.join(colmap_dir, "database.db"), dst_db)

        db = pycolmap.Database.open(dst_db)
        try:
            # priors first; mapping's pass 1 clears the (multi-sensor) rig
            # but leaves priors in place.
            configure_rig_and_frames(db, ref_modality=args.ref_modality)
            write_pose_priors(db, flight_dir, nav, position_std=args.prior_std)
        finally:
            db.close()

        recs = run_rig_mapping(
            dst_db, image_path, workspace, ref_modality=args.ref_modality
        )
        if not recs:
            raise SystemError("Rig mapping produced no reconstruction.")
        reconstruction = max(recs.values(), key=lambda r: r.num_reg_frames)
        print(f"Final model: {reconstruction.num_reg_frames} frames.")

    estimate = solve_rig_boresight(reconstruction, nav, times)
    export_rig_camera_models(reconstruction, estimate, nav, save_dir)


if __name__ == "__main__":
    main()

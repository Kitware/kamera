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
    best_reconstruction,
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
            write_pose_priors(db, flight_dir, nav, position_std=args.prior_std)
        finally:
            db.close()

        sparse_dir = os.path.join(workspace, "sparse")
        recs = run_rig_mapping(dst_db, image_path, sparse_dir)
        if not recs:
            raise SystemError("Mapping produced no reconstruction.")
        reconstruction = best_reconstruction(recs)
        print(f"Best model: {int(reconstruction.num_reg_images())} images.")

    estimate = solve_rig_boresight(
        reconstruction, nav, times, ref_modality=args.ref_modality
    )
    export_rig_camera_models(
        reconstruction, estimate, nav, save_dir, ref_modality=args.ref_modality
    )


if __name__ == "__main__":
    main()

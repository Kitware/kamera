"""Build a rig-constrained, INS-prior-registered COLMAP model for a flight.

Starts from an existing feature-extracted and matched COLMAP database
(e.g. the one under ``<flight>/colmap_rgb``), configures rigs, frames,
and pose priors in a copy of it under the workspace, and runs mapping.

Example:
    python build_rig_model.py /data/fl09_85mm_25_5deg
"""

import argparse
import os
import shutil

from rich import print

from kamera.postflight.rig import (
    configure_rig_and_frames,
    run_rig_mapping,
    write_pose_priors,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("flight_dir", help="KAMERA flight directory (contains *_meta.json)")
    p.add_argument(
        "--colmap-dir",
        default=None,
        help="Source colmap workspace with database.db and images0 "
        "(default: <flight_dir>/colmap_rgb)",
    )
    p.add_argument(
        "--workspace",
        default=None,
        help="Output workspace; database.db is copied here if absent "
        "(default: <flight_dir>/colmap_rig)",
    )
    p.add_argument(
        "--mapper",
        choices=["incremental", "global", "none"],
        default="incremental",
        help="Mapper to run after configuration; 'none' configures only.",
    )
    p.add_argument(
        "--prior-std",
        type=float,
        default=2.0,
        help="INS position prior standard deviation in meters.",
    )
    p.add_argument("--no-priors", action="store_true", help="Skip pose priors.")
    p.add_argument(
        "--ref-modality",
        default="rgb",
        help="Modality of the rig's reference sensor.",
    )
    args = p.parse_args()

    colmap_dir = args.colmap_dir or os.path.join(args.flight_dir, "colmap_rgb")
    workspace = args.workspace or os.path.join(args.flight_dir, "colmap_rig")
    src_db = os.path.join(colmap_dir, "database.db")
    dst_db = os.path.join(workspace, "database.db")
    image_path = os.path.join(colmap_dir, "images0")

    os.makedirs(workspace, exist_ok=True)
    if not os.path.exists(dst_db):
        print(f"Copying {src_db} -> {dst_db}")
        shutil.copyfile(src_db, dst_db)

    import pycolmap

    db = pycolmap.Database.open(dst_db)
    try:
        configure_rig_and_frames(db, ref_modality=args.ref_modality)
        if not args.no_priors:
            write_pose_priors(db, args.flight_dir, position_std=args.prior_std)
    finally:
        db.close()

    if args.mapper == "none":
        print("Configuration done; skipping mapping.")
        return

    sparse_dir = os.path.join(workspace, "sparse")
    print(f"Running {args.mapper} mapping into {sparse_dir}...")
    recs = run_rig_mapping(
        dst_db,
        image_path,
        sparse_dir,
        use_priors=not args.no_priors,
        mapper=args.mapper,
    )
    print(f"Produced {len(recs)} model(s).")
    for idx, rec in recs.items():
        print(f"--- model {idx}:")
        print(rec.summary())


if __name__ == "__main__":
    main()

"""Calibrate a KAMERA rig from a flight: prior mapping, one boresight
solve per modality group, and per-camera StandardCamera export.

Each modality group is an independently reconstructed COLMAP workspace
(EO = rgb+uv under ``colmap_rgb``; IR under ``colmap_ir``, since SIFT
cannot match long-wave IR to visible). A group's boresight is solved
against the INS, so every camera's exported mount is camera->INS in the
same physical INS frame -- EO and IR mounts are mutually consistent
without any cross-modal matching or transfer calibration.

Examples:
    # EO + IR, mapping each workspace fresh
    python calibrate_rig.py /data/fl09_85mm_25_5deg

    # reuse existing ENU (sim3-aligned or prior-mapped) models, no remap
    python calibrate_rig.py /data/fl09_85mm_25_5deg --reuse-aligned
"""

import argparse
import os
import pathlib
import shutil
from typing import Dict, List, Optional, Tuple

import pycolmap
from rich import print

from kamera.postflight.boresight import (
    export_rig_camera_models,
    solve_rig_boresight,
)
from kamera.postflight.registration_gifs import write_registration_gifs
from kamera.postflight.registration_homography import write_registration_homographies
from kamera.postflight.rig import (
    _order_by_ref,
    basename_to_time,
    best_reconstruction,
    build_colmap_database,
    derive_sensor_from_rig,
    run_rig_mapping,
    write_pose_priors,
)
from kamera.postflight.rig_model import (
    build_rig_model,
    camera_record,
    group_record,
    reprojection_error_px,
    write_rig_json,
)
from kamera.sensor_models.nav_state import NavStateINSJson

# (reference modality, colmap workspace subdirectory) per group.
DEFAULT_GROUPS = [("rgb", "colmap_rgb"), ("ir", "colmap_ir")]


def _largest_aligned_model(colmap_dir: str) -> Optional[str]:
    """Path to the aligned/ submodel with the most images, if any."""
    aligned = os.path.join(colmap_dir, "aligned")
    if not os.path.isdir(aligned):
        return None
    best, best_n = None, -1
    for name in os.listdir(aligned):
        path = os.path.join(aligned, name)
        try:
            n = int(pycolmap.Reconstruction(path).num_images())
        except Exception:
            continue
        if n > best_n:
            best, best_n = path, n
    return best


def _resolve_model(
    flight_dir: str,
    colmap_dir: str,
    workspace: str,
    nav: NavStateINSJson,
    prior_std: float,
    reuse_aligned: bool,
) -> Tuple["pycolmap.Reconstruction", str]:
    """Get an ENU reconstruction for a group: reuse an existing aligned
    model if asked, else prior-map the workspace database. Returns
    (reconstruction, source_label) for provenance."""
    if reuse_aligned:
        model = _largest_aligned_model(colmap_dir)
        if model is not None:
            print(f"Reusing aligned model {model}")
            return pycolmap.Reconstruction(model), f"reused-aligned:{model}"
        print(f"[yellow]No aligned model under {colmap_dir}; mapping instead.")

    os.makedirs(workspace, exist_ok=True)
    dst_db = os.path.join(workspace, "database.db")
    image_path = os.path.join(colmap_dir, "images0")
    src_db = os.path.join(colmap_dir, "database.db")
    source = "prior-mapped"
    if not os.path.exists(dst_db):
        if os.path.exists(src_db):
            # reuse a database already feature-extracted + matched
            print(f"Copying database into {dst_db}")
            shutil.copyfile(src_db, dst_db)
            source = "prior-mapped (existing database)"
        else:
            # from scratch: extract features and match from images0
            build_colmap_database(dst_db, image_path, flight_dir, nav, prior_std)
            source = "prior-mapped (extracted + spatial-matched)"
    db = pycolmap.Database.open(dst_db)
    try:
        write_pose_priors(db, flight_dir, nav, position_std=prior_std)
    finally:
        db.close()
    recs = run_rig_mapping(dst_db, image_path, os.path.join(workspace, "sparse"))
    if not recs:
        raise SystemError(f"Mapping produced no reconstruction for {colmap_dir}.")
    rec = best_reconstruction(recs)
    print(f"Best model: {int(rec.num_reg_images())} images.")
    return rec, source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("flight_dir")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--prior-std", type=float, default=2.0)
    p.add_argument(
        "--reuse-aligned",
        action="store_true",
        help="Use each workspace's existing aligned/ model instead of "
        "re-mapping (the boresight is gauge-independent).",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=None,
        metavar="MODALITY:SUBDIR",
        help="Override modality groups, e.g. rgb:colmap_rgb ir:colmap_ir.",
    )
    p.add_argument("--no-gifs", action="store_true", help="Skip registration gifs.")
    p.add_argument("--num-gifs", type=int, default=5, help="Gifs per camera.")
    args = p.parse_args()

    flight_dir = args.flight_dir
    save_dir = args.save_dir or os.path.join(flight_dir, "kamera_models")
    groups = (
        [tuple(g.split(":", 1)) for g in args.groups]
        if args.groups
        else DEFAULT_GROUPS
    )

    nav = NavStateINSJson(pathlib.Path(flight_dir).rglob("*_meta.json"))
    times = basename_to_time(flight_dir)

    all_models: Dict[str, object] = {}
    image_dirs: Dict[str, str] = {}
    rig_groups: List[Dict] = []
    rig_cameras: List[Dict] = []
    identity_rec = None  # any group's reconstruction, for flight identity + ENU
    for ref_modality, subdir in groups:
        colmap_dir = os.path.join(flight_dir, subdir)
        if not os.path.isdir(colmap_dir):
            print(f"[yellow]Skipping {ref_modality}: {colmap_dir} not found.")
            continue
        print(f"\n[bold blue]=== Calibrating {ref_modality} group ({subdir}) ===")
        workspace = os.path.join(flight_dir, f"colmap_rig_{ref_modality}")
        rec, source = _resolve_model(
            flight_dir, colmap_dir, workspace, nav, args.prior_std, args.reuse_aligned
        )
        identity_rec = identity_rec or rec
        estimate = solve_rig_boresight(rec, nav, times, ref_modality=ref_modality)
        sensor_from_rig = derive_sensor_from_rig(rec, ref_modality=ref_modality)
        models = export_rig_camera_models(
            rec, estimate, nav, save_dir,
            ref_modality=ref_modality, sensor_from_rig=sensor_from_rig,
        )
        all_models.update(models)

        ref_folder = _order_by_ref(models, ref_modality)[0]
        rig_groups.append(
            group_record(ref_modality, ref_folder, estimate, source, int(rec.num_reg_images()))
        )
        for folder, model in models.items():
            n_imgs = sum(
                1
                for im in rec.images.values()
                if im.has_pose and im.name.rsplit("/", 1)[0] == folder
            )
            rig_cameras.append(
                camera_record(
                    folder, model, sensor_from_rig.get(folder),
                    is_reference=(folder == ref_folder),
                    reprojection_px=reprojection_error_px(rec, model, folder, times),
                    num_images=n_imgs,
                )
            )
            image_dirs[folder] = os.path.join(colmap_dir, "images0", folder)

    print("\n[bold green]=== Calibration complete ===")
    print(f"Wrote {len(all_models)} camera models to {save_dir}:")
    for folder in sorted(all_models):
        print(f"  {folder}")

    if identity_rec is not None:
        rig = build_rig_model(identity_rec, nav, rig_groups, rig_cameras)
        rig_path = write_rig_json(save_dir, rig)
        print(f"Wrote complete rig model: {rig_path}")

    if all_models:
        cal_paths = write_registration_homographies(all_models, save_dir)
        if cal_paths:
            print(f"Wrote {len(cal_paths)} DIVE calibration.json files:")
            for path in cal_paths:
                print(f"  {path}")

    if not args.no_gifs and all_models:
        print("\n[bold blue]=== Writing registration gifs ===")
        gif_dir = os.path.join(save_dir, "registration_gifs")
        gifs = write_registration_gifs(
            all_models, image_dirs, times, gif_dir, num_gifs=args.num_gifs
        )
        print(f"Wrote {len(gifs)} gifs to {gif_dir}.")


if __name__ == "__main__":
    main()

"""Calibrate a KAMERA rig from a flight: prior mapping, one boresight
solve per modality group, and per-camera StandardCamera export.

Each modality group is an independently reconstructed COLMAP workspace
(EO = rgb+uv under ``colmap_rgb``; IR under ``colmap_ir``, since SIFT
cannot match long-wave IR to visible). A group's boresight is solved
against the INS, so every camera's exported mount is camera->INS in the
same physical INS frame -- EO and IR mounts are mutually consistent
without any cross-modal matching or transfer calibration.

With ``--fuse`` (and the optional ``fusion`` dependency group installed)
the IR images are additionally registered directly into the EO model via
deep cross-modal matching (MINIMA-LoFTR; see ``fusion.py``), and the
boresight/extrinsics/export re-run once on that single multimodal model
-- replacing the INS-relayed EO<->IR link with direct image evidence.
The per-group solve still runs first: it supplies the IR intrinsics and
the warp initialization, and remains the fallback for any IR camera that
fails to fuse.

Examples:
    # EO + IR, mapping each workspace fresh
    python calibrate_rig.py /data/fl09_85mm_25_5deg

    # reuse existing ENU (sim3-aligned or prior-mapped) models, no remap
    python calibrate_rig.py /data/fl09_85mm_25_5deg --reuse-aligned

    # one fused multimodal model (needs: uv sync --group fusion)
    python calibrate_rig.py /data/fl09_85mm_25_5deg --fuse
"""

import argparse
import json
import os
import pathlib
import shutil
from typing import Dict, List, Optional, Tuple

import pycolmap
import ubelt as ub
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


def _fuse_groups(
    args, flight_dir, save_dir, per_group, all_models, image_dirs, times, nav
):
    """Register the IR group into the EO model (see fusion.py), re-solve
    boresight + extrinsics once on the fused reconstruction, and re-export.

    `all_models` is updated in place so the downstream QC (gifs, DIVE
    homographies) runs off the fused mounts. Returns the replacement
    (rig_groups, rig_cameras, extra_provenance); IR cameras that fail to
    fuse keep their two-boresight export and group record.
    """
    # lazy: torch/vismatch only load on the --fuse path
    from kamera.postflight.deep_match import DeepMatcher
    from kamera.postflight.fusion import fuse_ir_into_eo, mount_delta_deg

    print(f"\n[bold blue]=== Fusing IR into the EO model ({args.fuse_matcher}) ===")
    eo, ir = per_group["rgb"], per_group["ir"]
    fused_rec = eo["rec"]  # mutated into the fused model
    report = fuse_ir_into_eo(
        fused_rec,
        ir["rec"],
        all_models,
        image_dirs,
        times,
        DeepMatcher(args.fuse_matcher),
        pairs_per_ir=args.fuse_pairs_per_ir,
        max_dt_s=args.fuse_max_dt,
        snap_px=args.fuse_snap_px,
        ransac_px=args.fuse_ransac_px,
        min_inliers=args.fuse_min_inliers,
        warp_scale=args.fuse_warp_scale,
        run_ba=args.fuse_ba or args.fuse_refine_ir_intrinsics,
        refine_ir_intrinsics=args.fuse_refine_ir_intrinsics,
        max_images=args.fuse_max_images,
    )
    fused_dir = ub.ensuredir(
        os.path.join(flight_dir, "colmap_rig_fused", "sparse", "0")
    )
    fused_rec.write(fused_dir)
    print(f"Wrote fused model to {fused_dir}")

    fused = _calibrate_group(fused_rec, nav, times, save_dir, "rgb")
    fused_models = fused["models"]

    # QC: how far did direct image evidence move each IR mount from the
    # INS-relayed (two-boresight) solution?
    deltas = {}
    for folder in sorted(ir["models"]):
        if folder in fused_models:
            deltas[folder] = mount_delta_deg(
                all_models[folder], fused_models[folder]
            )
            print(
                f"  {folder}: fused mount is {deltas[folder]:.3f} deg from "
                "the two-boresight mount"
            )
    fallback = [f for f in ir["models"] if f not in fused_models]
    for folder in fallback:
        print(
            f"[yellow]  {folder}: too few fused frames; keeping its "
            "two-boresight calibration."
        )
    all_models.update(fused_models)

    source = (
        f"fused:{args.fuse_matcher} "
        f"(rgb: {eo['source']}; ir: {ir['source']})"
    )
    rig_groups = [
        group_record(
            "rgb+ir", fused["ref_folder"], fused["estimate"], source,
            int(fused_rec.num_reg_images()),
        )
    ]
    if fallback:
        rig_groups.append(
            group_record(
                "ir", ir["ref_folder"], ir["estimate"],
                ir["source"] + " (fusion fallback)",
                int(ir["rec"].num_reg_images()),
            )
        )
    rig_cameras = list(fused["camera_records"].values()) + [
        ir["camera_records"][f] for f in fallback
    ]

    report["mount_delta_vs_two_boresight_deg"] = deltas
    report_path = os.path.join(save_dir, "fusion_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote fusion report: {report_path}")

    provenance = {
        "fusion": {k: v for k, v in report.items() if k != "per_image"}
    }
    return rig_groups, rig_cameras, provenance


def _camera_records(rec, models, sensor_from_rig, ref_folder, times):
    """One camera_record per exported model, counted against `rec`,
    keyed by camera folder."""
    records = {}
    for folder, model in models.items():
        n_imgs = sum(
            1
            for im in rec.images.values()
            if im.has_pose and im.name.rsplit("/", 1)[0] == folder
        )
        records[folder] = camera_record(
            folder,
            model,
            sensor_from_rig.get(folder),
            is_reference=(folder == ref_folder),
            reprojection_px=reprojection_error_px(rec, model, folder, times),
            num_images=n_imgs,
        )
    return records


def _calibrate_group(rec, nav, times, save_dir, ref_modality):
    """Boresight, rig extrinsics, per-camera yaml export, and rig-JSON
    camera records for one ENU reconstruction."""
    estimate = solve_rig_boresight(rec, nav, times, ref_modality=ref_modality)
    sensor_from_rig = derive_sensor_from_rig(rec, ref_modality=ref_modality)
    models = export_rig_camera_models(
        rec, estimate, nav, save_dir,
        ref_modality=ref_modality, sensor_from_rig=sensor_from_rig,
    )
    ref_folder = _order_by_ref(models, ref_modality)[0]
    return {
        "estimate": estimate,
        "sensor_from_rig": sensor_from_rig,
        "models": models,
        "ref_folder": ref_folder,
        "camera_records": _camera_records(
            rec, models, sensor_from_rig, ref_folder, times
        ),
    }


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
    fuse = p.add_argument_group(
        "cross-modal fusion (requires: uv sync --group fusion)"
    )
    fuse.add_argument(
        "--fuse",
        action="store_true",
        help="Register IR images directly into the EO model with deep "
        "cross-modal matches and export from the single fused model.",
    )
    fuse.add_argument(
        "--fuse-matcher",
        default="minima-loftr",
        help="vismatch matcher name (e.g. minima-loftr, minima-roma).",
    )
    fuse.add_argument(
        "--fuse-pairs-per-ir",
        type=int,
        default=1,
        help="EO partner images matched per IR image (1 = the colocated "
        "same-trigger image; more adds neighboring triggers).",
    )
    fuse.add_argument(
        "--fuse-max-dt",
        type=float,
        default=15.0,
        help="Max seconds between an IR image and an EO partner trigger.",
    )
    fuse.add_argument(
        "--fuse-snap-px",
        type=float,
        default=16.0,
        help="Max distance (full-res EO px) for a matched EO pixel to "
        "attach to a triangulated observation as a track observation.",
    )
    fuse.add_argument(
        "--fuse-ransac-px",
        type=float,
        default=3.0,
        help="PnP RANSAC threshold in IR pixels.",
    )
    fuse.add_argument(
        "--fuse-min-inliers",
        type=int,
        default=12,
        help="Minimum PnP inliers to accept an IR registration.",
    )
    fuse.add_argument(
        "--fuse-warp-scale",
        type=float,
        default=1.0,
        help="Scale of the EO-warped-to-IR matching image relative to "
        "the IR size.",
    )
    fuse.add_argument(
        "--fuse-max-images",
        type=int,
        default=None,
        help="Uniformly subsample the IR images (smoke runs).",
    )
    fuse.add_argument(
        "--fuse-ba",
        action="store_true",
        help="Bundle-adjust the fused model (EO frozen). Off by default: "
        "IR poses come from the joint model alignment, and per-image BA "
        "against sparse cross-modal tracks re-adds per-image wobble.",
    )
    fuse.add_argument(
        "--fuse-refine-ir-intrinsics",
        action="store_true",
        help="Let the fused bundle adjustment refine IR focal/distortion "
        "(implies --fuse-ba).",
    )
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
    per_group: Dict[str, Dict] = {}
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
        group = _calibrate_group(rec, nav, times, save_dir, ref_modality)
        group.update(rec=rec, source=source)
        per_group[ref_modality] = group
        all_models.update(group["models"])
        for folder in group["models"]:
            image_dirs[folder] = os.path.join(colmap_dir, "images0", folder)

    rig_groups: List[Dict] = [
        group_record(
            mod, g["ref_folder"], g["estimate"], g["source"],
            int(g["rec"].num_reg_images()),
        )
        for mod, g in per_group.items()
    ]
    rig_cameras: List[Dict] = [
        record for g in per_group.values() for record in g["camera_records"].values()
    ]
    extra_provenance = None

    if args.fuse and "rgb" in per_group and "ir" in per_group:
        rig_groups, rig_cameras, extra_provenance = _fuse_groups(
            args, flight_dir, save_dir, per_group, all_models, image_dirs,
            times, nav,
        )
    elif args.fuse:
        print(
            "[yellow]--fuse needs both an rgb and an ir group; "
            "skipping fusion."
        )

    print("\n[bold green]=== Calibration complete ===")
    print(f"Wrote {len(all_models)} camera models to {save_dir}:")
    for folder in sorted(all_models):
        print(f"  {folder}")

    if identity_rec is not None:
        rig = build_rig_model(
            identity_rec, nav, rig_groups, rig_cameras,
            extra_provenance=extra_provenance,
        )
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

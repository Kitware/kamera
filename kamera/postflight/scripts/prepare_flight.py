"""Stage a raw KAMERA flight into the images0 layout the calibration
pipeline reads, optionally paring the frame set.

Reorganizes ``<raw>/<view>/*`` (center_view / left_view / right_view,
each with all modalities + meta jsons) into
``<flight_dir>/colmap_<group>/images0/<prefix>_<channel>_<modality>/``
via symlinks, keeping synchronized triggers together. Then run
calibrate_rig.py on the same flight_dir.

Examples:
    # keep everything, symlinked
    python prepare_flight.py /data/fl004/images_21deg_N56RF /data/fl004

    # pare to ~one frame per 100 m of travel, reporting overlap
    python prepare_flight.py /data/fl004/images_21deg_N56RF /data/fl004 \
        --spacing 100 --focal-px 15360
"""

import argparse
import os

from rich import print

from kamera.postflight.flight_prep import stage_flight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw_dir", help="Raw imagery dir containing the view folders.")
    p.add_argument("flight_dir", help="Flight dir to stage colmap_* trees into.")
    p.add_argument(
        "--prefix",
        default=None,
        help="Camera-folder prefix (default: raw dir name minus 'images_').",
    )
    p.add_argument(
        "--spacing",
        type=float,
        default=0.0,
        help="Keep a trigger only after this many meters of travel (0 = all).",
    )
    p.add_argument("--every", type=int, default=1, help="Keep every Nth survivor.")
    p.add_argument("--max-frames", type=int, default=None, help="Cap trigger count.")
    p.add_argument(
        "--focal-px",
        type=float,
        default=0.0,
        help="Focal length in px, to report estimated forward overlap.",
    )
    p.add_argument("--height-px", type=int, default=4384, help="Image height in px.")
    p.add_argument("--copy", action="store_true", help="Copy instead of symlink.")
    args = p.parse_args()

    prefix = args.prefix or os.path.basename(args.raw_dir.rstrip("/")).removeprefix(
        "images_"
    )
    print(f"Prefix: {prefix}")
    counts = stage_flight(
        args.raw_dir,
        args.flight_dir,
        prefix,
        spacing_m=args.spacing,
        every=args.every,
        max_frames=args.max_frames,
        copy=args.copy,
        focal_px=args.focal_px,
        height_px=args.height_px,
    )
    print("\nStaged images per camera:")
    for folder in sorted(counts):
        print(f"  {folder}: {counts[folder]}")
    print(f"\nNext: python calibrate_rig.py {args.flight_dir}")


if __name__ == "__main__":
    main()

"""Stage a raw KAMERA flight into the images0 layout the calibration
pipeline reads (calibrate_rig.py does this automatically on first run;
this script exists for staging-only workflows).

Reorganizes ``<raw>/<view>/*`` (center_view / left_view / right_view,
each with all modalities + meta jsons) into
``<flight_dir>/colmap_<group>/images0/<prefix>_<channel>_<modality>/``
via symlinks, keeping synchronized triggers together.

Example:
    python prepare_flight.py /data/fl004/images_21deg_N56RF /data/fl004
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
    p.add_argument("--copy", action="store_true", help="Copy instead of symlink.")
    args = p.parse_args()

    prefix = args.prefix or os.path.basename(args.raw_dir.rstrip("/")).removeprefix(
        "images_"
    )
    print(f"Prefix: {prefix}")
    counts = stage_flight(args.raw_dir, args.flight_dir, prefix, copy=args.copy)
    print("\nStaged images per camera:")
    for folder in sorted(counts):
        print(f"  {folder}: {counts[folder]}")
    print(f"\nNext: python calibrate_rig.py {args.flight_dir}")


if __name__ == "__main__":
    main()

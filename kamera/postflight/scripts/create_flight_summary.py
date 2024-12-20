#!/usr/bin/env python
from __future__ import division, print_function
import argparse
import os
import pathlib

# Custom package imports.
from kamera.postflight import utilities


def main():
    parser = argparse.ArgumentParser(
        description="Convert all images from a " "flight into shapefiles."
    )
    parser.add_argument(
        "-flight_dir",
        help="Flight directory containing "
        "subdirectories 'left_view', 'center_view', and 'right_view', "
        "each containing imagery and meta.json files. Defaults to None.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-output_dir",
        help="Output directory (defaults to 'processed_results'.).",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    flight_dir = args.flight_dir
    output_dir = args.output_dir

    # uncomment these if you wish to skip the argument assigment
    # flight_dir = '/example_flight_dir'
    # output_dir = '/example_output_dir'

    if not output_dir:
      base_dir = pathlib.Path(flight_dir).parents[0]
      output_dir = os.path.join(base_dir, "processed_results")

    if not flight_dir:
      raise SystemError("No flight dir specified! Please pass one as an argument or hardcode one in the file.")

    utilities.create_flight_summary(flight_dir, output_dir)


if __name__ == "__main__":
    main()

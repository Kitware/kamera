#!/usr/bin/env python
"""
Library handling projection operations of a standard camera model.
"""
from __future__ import division, print_function
import argparse

# KAMERA imports.
from postflight_scripts import utilities


def main():
    parser = argparse.ArgumentParser(description='Convert all images from a '
                                     'flight into GeoTIFF.')
    parser.add_argument("flight_dir", help='Flight directory containing '
                        'subdirectories \'LEFT\', \'CENT\', and \'RIGHT\', '
                        'each containing imagery and meta.json files.',
                        type=str)
    parser.add_argument('-output_dir', help='Output directory (defaults to '
                        '\'geotiffs\' inside the flight directory.).',
                        type=str, default=None)

    args = parser.parse_args()

    utilities.create_all_geotiff(args.flight_dir, args.output_dir)


if __name__ == '__main__':
    main()


#flight_dir = '/host_filesystem/media/mattb/7e7167ba-ad6f-4720-9ced-b699f49ba3aa/kamera/calibration_2019/03'
#create_all_geotiff(flight_dir)
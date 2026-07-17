#!/usr/bin/env python
"""
Script to convert Bayered images to deBayered RGB images.
"""
from __future__ import division, print_function
import argparse

# KAMERA imports.
from kamera.postflight import utilities


def main():
    parser = argparse.ArgumentParser(description='Search for images with GLOB '
                                     'pattern *rgb.tif and if Bayered, '
                                     'deBayer the image. If \'dst_dir\' is '
                                     'different from \'src_dir\', images are '
                                     'saved to a path where \'dst_dir\' '
                                     'replaces the path in \'src_dir\'.')
    parser.add_argument("src_dir", help="Path to search for images",
                        type=str)
    parser.add_argument("-dst_dir", help='Destination path to use on saving '
                        'that replaces the path in \'src_dir\'.',
                        type=str, default=None)

    args = parser.parse_args()

    utilities.debayer_dir_tree(args.src_dir, args.dst_dir)


if __name__ == '__main__':
    main()

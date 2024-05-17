#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
np.seterr(all='raise') # Divide by zeros are a BAD SIGN. let's make it error them
import os
import sys
import glob
import argparse
import warnings
from functools import partial
from pprint import pprint

# KAMERA imports.
from kamera.postflight import utilities


def HEADING(text):
    ln = min(len(text) + 3, 44)
    sep = '=' * ln
    return '\n'.join([sep, text, sep])


def menu_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(description='Convert all images from a '
                                                 'flight into GeoTIFF.')
    parser.add_argument('-i', "--flight_dir", help="""Flight directory containing Flight-specific data dir, e.g.
                                                /mnt/flight_data/project2019/fl23/
                                                containing FOV (e.g. 'center_view') directories
                                           each containing imagery and meta.json files""",
                        type=str)
    parser.add_argument('-o', '--output_dir', help='Output directory (defaults to '
                                            '\'geotiffs\' inside the flight directory.).',
                        type=str, default=None)
    parser.add_argument('--geotiff_quality', help='Quality 1-100',
                        type=int, default=75)
    parser.add_argument('-v', '--verbosity', help='Verbosity of print output',
                        type=int, default=0)
    parser.add_argument('--multi', help='Use multithreading', action='store_true')
    parser.add_argument('--summary', help='Run flight summary', action='store_true')
    parser.add_argument('--detections', help='Run detection summary', action='store_true')
    parser.add_argument('--homographies', help='Run image_to_image_homographies summary', action='store_true')
    parser.add_argument('--geotiff', help='Run geotiff conversion', action='store_true')

    return parser

def main(args):
    # type: (argparse.Namespace) -> int
    """
    Run post-flight-processing tasks on the flight directory
    :param flight_dir: Flight-specific data dir, e.g. /mnt/flight_data/project2019/fl23/
    :return:

    - Create all geotiffs from images
    - Create flight summary
    - measure_image_to_image_homographies_flight_dir ???
    - detection_summary
    """
    print("{}:main() args: {}".format(__file__, args))
    flight_dir = args.flight_dir
    geotiff_quality = args.geotiff_quality
    multi_threaded = args.multi
    if not flight_dir:
        raise RuntimeError('Must specify --flight_dir')
    if not os.path.exists(flight_dir):
        raise RuntimeError("Flight dir does not exist: {}".format(flight_dir))
    dtype = utilities.get_dir_type(flight_dir)
    if dtype != 'flight':
        raise RuntimeError("Called run_postflight:main on `{}` directory, wants flight dir: {}".format(dtype, flight_dir))

    if not any([args.summary, args.detections, args.homographies, args.geotiff]):
        print("The Tao Does Nothing, But Leaves Nothing Undone\n See --help for command options")
        return 0

    if args.summary:
        print(HEADING('Processing flight summary'))
        process_summary = utilities.create_flight_summary(flight_dir)
        pprint(process_summary)

    # Curry all the configuration parameters here
    def detection_summary_part(detection_csv):
        return utilities.detection_summary(flight_dir=flight_dir, detection_csvs=detection_csv, animal_min_meters=0.2,
                                     animal_max_meters=7, geo_registration_error=10)

    # get all the csvs in detection timestamped directories
    detection_csv_list = []
    detection_csv_list += glob.glob(os.path.join(flight_dir, "detections", '*.csv'))

    if args.detections:
        print(HEADING('Processing detection CSVs'))
        detections_processed = list(map(detection_summary_part, detection_csv_list))
        pprint(detections_processed)
        if not len(detections_processed):
            warnings.warn('Did not process any detection files')

    if args.geotiff:
        print(HEADING('Processing geotiffs'))
        process_summary = utilities.create_all_geotiff(flight_dir,
                                                       quality=geotiff_quality,
                                                       multi_threaded=multi_threaded,
                                                       verbosity=args.verbosity)
        if not process_summary['shapefile_count']:
            raise RuntimeError("Failed to generate any shapefiles")

    if args.homographies:
        print(HEADING('Processing image-to-image homographies'))
        utilities.measure_image_to_image_homographies_flight_dir(flight_dir,
                                                         multi_threaded=args.multi,
                                                         save_viz_gif=False)
    print("========= FINISHED POSTPROC =========")


if __name__ == '__main__':
    parser = menu_parser()
    sys.exit(main(parser.parse_args()))


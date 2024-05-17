#!/usr/bin/env python
"""
ckwg +31
Copyright 2019 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

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
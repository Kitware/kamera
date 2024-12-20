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

# Custom package imports.
from kamera.postflight import utilities


def main():
    parser = argparse.ArgumentParser(
        description="Convert all images from a " "flight into shapefiles."
    )
    parser.add_argument(
        "flight_dir",
        help="Flight directory containing "
        "subdirectories 'left_view', 'center_view', and 'right_view', "
        "each containing imagery and meta.json files. Defaults to None.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-output_dir",
        help="Output directory (defaults to " "'None'.).",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    flight_dir = args.flight_dir
    output_dir = args.output_dir

    # uncomment these if you wish to skip the argument assigment
    # flight_dir = '/example_flight_dir'
    # output_dir = '/example_output_dir'

    utilities.create_flight_summary(flight_dir, output_dir)


if __name__ == "__main__":
    main()

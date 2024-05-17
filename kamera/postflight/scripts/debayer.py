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

Script to convert Bayered images to deBayered RGB images.

"""
from __future__ import division, print_function
import argparse

# KAMERA imports.
from postflight_scripts import utilities


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

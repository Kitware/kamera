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
import numpy as np
import os
import glob
import argparse
import shutil


def fix(flight_dir):
    """Process all images in list of filenames.

    :param img_fnames: List of image paths to read Bayered imagery from.
    :type img_fnames: list

    :param img_fname_out: List of image paths to save deBayered imagery to.
    :type img_fname_out: list

    """
    left_dir = '%s/LEFT' % flight_dir
    right_dir = '%s/RIGHT' % flight_dir

    left_ir_tmp_dir = '%s/left_ir_temp' % flight_dir
    try:
        os.makedirs(left_ir_tmp_dir)
    except (OSError, IOError):
        pass

    right_ir_tmp_dir = '%s/right_ir_temp' % flight_dir
    try:
        os.makedirs(right_ir_tmp_dir)
    except (OSError, IOError):
        pass

    # Move out IR images from LEFT directory into right_ir_tmp_dir
    for in_fname in glob.glob('%s/*ir.tif' % left_dir):
        out_fname = in_fname.replace('_L_', '_R_')
        out_fname = '%s/%s' % (right_ir_tmp_dir, os.path.split(out_fname)[1])
        shutil.move(in_fname, out_fname)

    # Move out IR images from RIGHT directory into left_ir_tmp_dir
    for in_fname in glob.glob('%s/*ir.tif' % right_dir):
        out_fname = in_fname.replace('_R_', '_L_')
        out_fname = '%s/%s' % (left_ir_tmp_dir, os.path.split(out_fname)[1])
        shutil.move(in_fname, out_fname)

    # Move out IR images from right_ir_tmp_dir directory into RIGHT
    for in_fname in glob.glob('%s/*ir.tif' % right_ir_tmp_dir):
        out_fname = '%s/%s' % (right_dir, os.path.split(in_fname)[1])
        shutil.move(in_fname, out_fname)

    # Move out IR images from left_ir_tmp_dir directory into LEFT
    for in_fname in glob.glob('%s/*ir.tif' % left_ir_tmp_dir):
        out_fname = '%s/%s' % (left_dir, os.path.split(in_fname)[1])
        shutil.move(in_fname, out_fname)

    shutil.rmtree(left_ir_tmp_dir)
    shutil.rmtree(right_ir_tmp_dir)


def main():
    parser = argparse.ArgumentParser(description='Swap LEFT and RIGHT IR '
                                     'images.')
    parser.add_argument("flight_dir", help='Flight directory with LEFT and '
                        'RIGHT subdirectories.',
                        type=str)

    args = parser.parse_args()

    fix(args.flight_dir)


if __name__ == '__main__':
    main()

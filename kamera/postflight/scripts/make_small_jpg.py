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
import cv2
import os
import glob
import argparse
from postflight_scripts import utilities


def save_jpgs(img_fnames, img_fnames_out, quality=30, stretch_constrast=True):
    """Process all images in list of filenames.

    :param img_fnames: List of image paths to read Bayered imagery from.
    :type img_fnames: list

    :param img_fname_out: List of image paths to save deBayered imagery to.
    :type img_fname_out: list

    """
    for i in range(len(img_fnames)):
        img_fname = img_fnames[i]
        img_fname_out = img_fnames_out[i]

        print('Reading:', img_fname)
        img = cv2.imread(img_fname, -1)

        if img is None:
            continue

        if img_fname[-7:] == 'rgb.tif':
            # Wasn't debayered yet.
            img2 = utilities.debayer_image(img, 'bayer_gbrg8')

            if img2 is img:
                img = img[:, :, ::-1]
            else:
                img = img2

        if stretch_constrast:
            img = img.astype(np.float)
            img -= np.percentile(img.ravel(), 1)
            img[img < 0] = 0
            img /= np.percentile(img.ravel(), 99)/255
            img[img > 225] = 255
            img = np.round(img).astype(np.uint8)

            clahe = cv2.createCLAHE(clipLimit=1,
                                    tileGridSize=(5, 5))
            if img.ndim == 3:
                HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                HLS[:, :, 1] = clahe.apply(HLS[:, :, 1])
                img = cv2.cvtColor(HLS, cv2.COLOR_HLS2BGR)
            else:
                img = clahe.apply(img)

        print('Saving:', img_fname_out)
        if img.shape[1]*img.shape[0] < 1500*1500:
            # Don't compress the IR.
            cv2.imwrite(img_fname_out, img, [cv2.IMWRITE_JPEG_QUALITY,
                                             100])
        else:
            cv2.imwrite(img_fname_out, img, [cv2.IMWRITE_JPEG_QUALITY,
                                             quality])


def process_images(src_dir, dst_dir=None):
    dst_dir = '%s/small_jpg' % src_dir

    try:
        os.makedirs(dst_dir)
    except OSError:
        pass

    img_fnames = []
    img_fnames_out = []
    for root, dirnames, filenames in os.walk(src_dir):
        for img_fname in glob.glob('%s/*.tif' % root):
            img_fnames.append(img_fname)
            img_fname_out = img_fname.replace(src_dir, dst_dir)
            img_fnames_out.append('%s.jpg' % os.path.splitext(img_fname_out)[0])

    save_jpgs(img_fnames, img_fnames_out)


def main():
    parser = argparse.ArgumentParser(description='Create a directory of '
                                     'reduced-size jpegs of the imagery. '
                                     'Useful for quickly flipping through the '
                                     'imagery.')
    parser.add_argument("src_dir", help="Path to search for images",
                        type=str)
    parser.add_argument("-dst_dir", help='Destination path to use on saving.',
                        type=str, default=None)

    args = parser.parse_args()

    process_images(args.src_dir, args.dst_dir)


if __name__ == '__main__':
    main()

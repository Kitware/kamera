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
import os
import glob

# KAMERA imports.
from postflight_scripts import utilities


# ---------------------------- Define Paths ----------------------------------
flight_dir = 'fl04'
detection_csvs = glob.glob('%s/detections/*.csv' % flight_dir)
# ----------------------------------------------------------------------------


# Make sure 'RIGHT' is spelled correctly.
try:
    os.rename('%s/RGHT' % flight_dir, '%s/RIGHT' % flight_dir)
except OSError:
    pass

# Debayer all of the RGB images in all subdirectories of the flight directory.
utilities.debayer_dir_tree('%s/LEFT' % flight_dir, num_threads=10)
utilities.debayer_dir_tree('%s/CENT' % flight_dir, num_threads=10)
utilities.debayer_dir_tree('%s/RIGHT' % flight_dir, num_threads=10)

# Create a 'processed_results/geotiffs' folder within the flight directory and
# create a GeoTIFF for each collected image. Uses jpeg compression on the
# GeoTIFF.
utilities.create_all_geotiff(flight_dir, quality=75, multi_threaded=True)

# Create a flight summary for the imagery and metadata in the flight directory.
# If camera models work created/modified for this specific flight, you should
# make sure the camera model yaml files are included in a subdirectory
# 'camera_models' within the flight directory. Otherwise, default camera models
# from 2019-06-01 are used.
utilities.create_flight_summary(flight_dir)

# Option step to measure alignment of sequential images from each modality to
# generate the folder 'processed_results/homographies_img_to_img'. This
# calculation can take a while (couple seconds per image), but it will provide
# very precise assessment of redudant detections.
utilities.measure_image_to_image_homographies_flight_dir(flight_dir,
                                                         multi_threaded=True,
                                                         save_viz_gif=False)

utilities.detection_summary(flight_dir, detection_csvs, animal_min_meters=0.2,
                            animal_max_meters=7, geo_registration_error=10)

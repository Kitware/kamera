#! /usr/bin/python
"""
ckwg +31
Copyright 2018 by Kitware, Inc.
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

"""
import sys
import sqlite3
import numpy as np
import glob
import os
from scipy.spatial import distance_matrix
import cv2
import matplotlib.pyplot as plt

# Colmap Processing imports.
from colmap_processing.geo_conversions import llh_to_enu
from colmap_processing.database import COLMAPDatabase, blob_to_array, \
    pair_id_to_image_ids


# ----------------------------------------------------------------------------
# Landmark annotation GUI workspace path.
landmark_gui_workspace = '/media/workspace'

# Colmap directory where all of the Colmap files will be generated. All images
# should be placed in a sub-directory 'images'.
colmap_workspace_path = '/media/colmap'

# Camera model type.
model = 'OPENCV'
# ----------------------------------------------------------------------------


db = COLMAPDatabase('%s/colmap_database.db' % colmap_workspace_path)
db.create_tables()

image_dir = '%s/images0' % colmap_workspace_path

image_fnames = []
for ext in ['.jpg', '.tiff', '.png', '.bmp', '.jpeg']:
    image_fnames = image_fnames + glob.glob('%s/*%s' % (image_dir, ext))


img_fname_to_cam_id = {}
img_fname_to_img_id = {}
for image_id, fname in enumerate(image_fnames):
    image_id = image_id + 1
    img = cv2.imread(fname, 0)
    image_fname = os.path.splitext(os.path.split(fname)[1])[0]

    height, width = img.shape[:2]

    if model == 'OPENCV':
        f = 1.501825029214076494e+03
        k1 = -3.539975553110535356e-01
        k2 = 1.221648902040088081e-01
        params = [f, f, width/2.0, height/2.0, k1, k2, 0, 0]
        model_ind = 4
    else:
        raise Exception('Need to implement for model \'%s\'' % model)

    # Every image is from a different camera.
    camera_id = image_id

    img_fname_to_img_id[image_fname] = image_id
    img_fname_to_cam_id[image_fname] = camera_id

    db.add_camera(model_ind, width, height, params, prior_focal_length=1500,
                  camera_id=camera_id)
    db.add_image(os.path.split(fname)[1], camera_id, image_id=image_id)


# Parse landmark annotation GUI workspace.
img_keypoints_to_landmark = {}
for image_id, fname in enumerate(image_fnames):
    img_fname = os.path.splitext(os.path.split(fname)[1])[0]
    fname = ('%s/%s_image_points.txt' % (landmark_gui_workspace, img_fname))
    try:
        points = np.loadtxt(fname)
    except (OSError, IOError):
        points = np.zeros((0, 3))

    image_id = img_fname_to_img_id[img_fname]
    camera_id = img_fname_to_cam_id[img_fname]

    db.add_keypoints(image_id, points[:, 1:])

    img_keypoints_to_landmark[image_id] = points[:, 0].astype(np.int).tolist()


img_w_matches = list(img_keypoints_to_landmark.keys())
for ii in range(len(img_w_matches) - 1):
    image_id1 = img_w_matches[ii]
    matches1 = img_keypoints_to_landmark[image_id1]
    for jj in range(ii + 1, len(img_w_matches)):
        image_id2 = img_w_matches[jj]
        matches2 = img_keypoints_to_landmark[image_id2]

        matches = []
        for i in range(len(matches1)):
            try:
                j = matches2.index(matches1[i])
                matches.append([i, j])
            except ValueError:
                pass

        if len(matches) > 0:
            matches = np.array(matches, dtype=np.int)
            db.add_matches(image_id1, image_id2, matches)
            db.add_two_view_geometry(image_id1, image_id2, matches)

db.commit()
db.close()


db = COLMAPDatabase('/media/colmap_database.db')


# Read and check cameras.
rows = db.execute("SELECT * FROM cameras")
for row in rows:
    camera_id, model, width, height, params, prior = row
    params = blob_to_array(params, np.float64)
    print('Camera ID: %s   Model: %s   Resolution (%i, %i)   Params: %s' %
          (camera_id, model, width, height, params))

# Read and check images.
rows = db.execute("SELECT * FROM images")
for row in rows:
    camera_id, name, image_id = row[:3]
    print('Camera ID: %s   Name: %s   Image ID: %s' %
          (camera_id, name, image_id))

# Read and check matches.
db.get_all_pair_id()
db.get_match_dictionary()
db.get_keypoint_from_image_dict()

db.close()

#! /usr/bin/python
"""
ckwg +31
Copyright 2021 by Kitware, Inc.
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
from __future__ import division, print_function
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import json
import itertools
import copy

# Colmap Processing imports.
import colmap_processing.colmap_interface as colmap_interface
from colmap_processing.database import COLMAPDatabase, pair_id_to_image_ids, \
    blob_to_array


# ----------------------------------------------------------------------------
# Colmap directory.
colmap_dir = '/mnt/data/colmap'

# Source database.
source_db_fname = '/mnt/data/colmap/database.db'

# Path to the sparse dir containing images.bin, cameras.bin, and points3D.bin.
sparse_dir = '%s/sparse/0' % colmap_dir

sparse_out_dir = '%s/sparse' % colmap_dir
# ----------------------------------------------------------------------------

# Read in the details of all images.
cameras, images, points3D = colmap_interface.read_model(sparse_dir)

source_db = COLMAPDatabase.connect(source_db_fname)
kp_dict = source_db.get_keypoint_from_image_dict()
descr_dict = source_db.get_descriptors_from_image_dict()
match_dict = source_db.get_match_dictionary()
two_view_geoms = source_db.get_all_two_view_geometry()


base_fname_to_camera = {}
base_fname_to_camera_id = {}
camera_id_to_base_fname = {}
name_to_camera = {}
base_fname_to_src_images = {}
for image in images:
    img_base = images[image].name.split('/')[0]
    camera_id = images[image].camera_id
    camera = cameras[camera_id]

    if img_base not in base_fname_to_camera_id:
        base_fname_to_camera_id[img_base] = camera_id

    assert base_fname_to_camera_id[img_base] == camera_id

    camera_id_to_base_fname[camera_id] = img_base
    base_fname_to_camera[img_base] = camera

    if img_base not in base_fname_to_src_images:
        base_fname_to_src_images[img_base] = []

    base_fname_to_src_images[img_base].append(images[image])


new_db = {}
for name in camera_id_to_base_fname.values():
    fname = '%s/%s.db' % (colmap_dir, name)
    new_db[name] = db = COLMAPDatabase.connect(fname)
    db.create_tables()

    camera = base_fname_to_camera[name]
    db.add_camera(colmap_interface.CAMERA_MODEL_NAMES_TO_IND[camera.model],
                  camera.width, camera.height, camera.params, camera_id=1)

for name in base_fname_to_src_images:
    db = new_db[name]

    # Add images and keypoints and descriptors into the database.
    for image in base_fname_to_src_images[name]:
        db.add_image(image.name, 1, image.qvec, image.tvec, image.id)
        db.add_keypoints(image.id, kp_dict[image.id])
        db.add_descriptors(image.id, descr_dict[image.id])

    # Add images and keypoints and descriptors into the database.
    image_ids = set(db.get_keypoint_from_image_dict().keys())
    for pair in match_dict:
        image_id1 = int(pair[0])
        image_id2 = int(pair[1])
        if image_id1 in image_ids and image_id2 in image_ids:
            db.add_matches(image_id1, image_id2, match_dict[pair])

    for i in range(len(two_view_geoms[0])):
        pair_id = two_view_geoms[0][i]
        image_ids = two_view_geoms[1][i]

        image_id1 = int(image_ids[0])
        image_id2 = int(image_ids[1])
        if image_id1 in image_ids and image_id2 in image_ids:
            inlier_matches = two_view_geoms[2][i]
            F = two_view_geoms[3][i]
            E = two_view_geoms[4][i]
            H = two_view_geoms[5][i]
            config = two_view_geoms[6][i]

            db.add_two_view_geometry(image_id1, image_id2, inlier_matches, F,
                                     E, H, config)


for name in new_db:
    db = new_db[name]
    db.commit()
    db.close()


if False:
    fname = '/mnt/data/database.db'
    db = COLMAPDatabase.connect(fname)

    rows = db.execute("SELECT * FROM cameras")
    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)


for name in base_fname_to_camera:
    outdir_ = '%s/%s' % (sparse_out_dir, name)
    camera  = base_fname_to_camera[name]

    try:
        os.makedirs(outdir_)
    except (OSError, IOError):
        pass

    camera = base_fname_to_camera[name]
    camera = colmap_interface.Camera(1, camera.model, camera.width,
                                     camera.height, camera.params)

    images_ = dict([(image.id, image)
                    for image in base_fname_to_src_images[name]])

    points3D_ = {}

    point3D_ids = set()
    for image in base_fname_to_src_images[name]:
        point3D_ids = point3D_ids.union(set(image.point3D_ids))

    points3D_ = {ind:points3D[ind] for ind in point3D_ids if ind >= 0}

    colmap_interface.write_model({1:camera}, images_, points3D_, outdir_,
                                 ext=".bin")
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
from __future__ import division, print_function
import numpy as np
import os
import cv2
import subprocess
import matplotlib.pyplot as plt
import glob
import natsort
import math
import PIL
from osgeo import osr, gdal

# Colmap Processing imports.
from colmap_processing.geo_conversions import llh_to_enu
from colmap_processing.colmap_interface import read_images_binary, Image, \
    read_points3D_binary, read_cameras_binary, qvec2rotmat
from colmap_processing.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
import colmap_processing.vtk_util as vtk_util
from colmap_processing.geo_conversions import enu_to_llh, llh_to_enu, \
    rmat_ecef_enu, rmat_enu_ecef
from colmap_processing.static_camera_model import load_static_camera_from_file
from colmap_processing.vtk_util import render_distored_image



# ----------------------------------------------------------------------------
save_dir = '/media/data/test'
location = 'test'


if location == 'test':
    # Meshed 3-D model used to render an synthetic view for sanity checking and
    # to produce the depth map.
    mesh_fname = 'mesh.ply'
    mesh_lat0 = 0    # degrees
    mesh_lon0 = 0  # degrees
    mesh_h0 = 73          # meters above WGS84 ellipsoid
else:
    raise Exception('Unrecognized location \'%s\'' % location)

# VTK renderings are limited to monitor resolution (width x height).
monitor_resolution = (1080, 1080)

clipping_range = [1, 2000]
# ----------------------------------------------------------------------------


# Read model into VTK.
try:
    model_reader
    assert prev_loaded_fname == mesh_fname
except:
    model_reader = vtk_util.load_world_model(mesh_fname)
    prev_loaded_fname = mesh_fname


# -------------------------------- Define Camera -----------------------------
if False:
    # Manually specific camera.
    res_x = 1920
    res_y = 1080
    pos = [-36.75 - 5, 39.25, 11.74]
    pan = -180 + 90 + 90
    tilt = -45
    vfov = 90

    vtk_camera = vtk_util.CameraPanTilt(res_x, res_y, vfov, pos, pan, tilt)
else:
    # Read existing camera from file.
    camera_fname = 'camera_model.yaml'
    ret = load_static_camera_from_file(camera_fname)
    height, width, K, dist, R, depth_map, latitude, longitude, altitude = ret
    cam_pos = llh_to_enu(latitude, longitude, altitude, mesh_lat0, mesh_lon0,
                         mesh_h0)

    # R currently is relative to ENU coordinate system at latitude0,
    # longitude0, altitude0, but we need it to be relative to latitude,
    # longitude, altitude.
    Rp = np.dot(rmat_enu_ecef(mesh_lat0, mesh_lon0),
               rmat_ecef_enu(latitude, longitude))
    R = np.dot(R, Rp)

ret = vtk_util.render_distored_image(width, height, K, dist, cam_pos, R,
                                     model_reader, return_depth=True,
                                     monitor_resolution=(1080, 1080),
                                     clipping_range=clipping_range)
rendered_view, depth, E, N, U = ret


if False:
    im = PIL.Image.fromarray(depth, mode='F')  # float32
    depth_map_fname = '%s/camera_model_depth_map.tif' % save_dir
    im.save(depth_map_fname)

    depth_image = depth.copy()
    depth_image[depth_image > clipping_range[1]*0.9] = 0
    depth_image -= depth_image.min()
    depth_image /= depth_image.max()/255
    depth_image = np.round(depth_image).astype(np.uint8)

    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    cv2.imwrite('%s/depth_vizualization.png' % save_dir,
                depth_image[:, :, ::-1])

plt.imshow(rendered_view)


cv2.imwrite('%s/rendered.jpg' % save_dir, rendered_view[:, :, ::-1])

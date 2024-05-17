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
import trimesh
import math
import PIL
from osgeo import osr, gdal
from scipy.optimize import fmin, minimize, fminbound

# Colmap Processing imports.
from colmap_processing.geo_conversions import llh_to_enu
import colmap_processing.vtk_util as vtk_util
from colmap_processing.geo_conversions import enu_to_llh, llh_to_enu, \
    rmat_ecef_enu, rmat_enu_ecef
from colmap_processing.static_camera_model import save_static_camera, \
    load_static_camera_from_file, write_camera_krtd_file, project_to_camera, \
    unproject_from_camera


# ----------------------------------------------------------------------------
camera_model1_fname = 'camera_model.yaml'
camera_model2_fname = 'camera_model.yaml'
ponts_fname = 'points.txt'

#MUTC.
mesh_fname = '/home/c.ply'

latitude0 = 0    # degrees
longitude0 = 0  # degrees
altitude0 = 0            # meters above WGS84 ellipsoid

im_pts1 = np.loadtxt(ponts_fname)[:, :2]
# ----------------------------------------------------------------------------


img1 = cv2.imread(source_img)[:, :, ::-1]
img2 = cv2.imread(dest_img)[:, :, ::-1]


# Read model into VTK.
model_reader = vtk_util.load_world_model(mesh_fname)

# Load camera models.
ret = load_static_camera_from_file(camera_model1_fname)
height1, width1, K1, dist1, R1, depth_map1, latitude1, longitude1, altitude1 = ret
cam_pos1 = llh_to_enu(latitude1, longitude1, altitude1, latitude0, longitude0,
                      altitude0)
dist1[:] = 0

ret = load_static_camera_from_file(camera_model2_fname)
height2, width2, K2, dist2, R2, depth_map2, latitude2, longitude2, altitude2 = ret
cam_pos2 = llh_to_enu(latitude2, longitude2, altitude2, latitude0, longitude0,
                      altitude0)
dist2[:] = 0

# Render views from each camera.
clipping_range = [18, 234]
ret = vtk_util.render_distored_image(width1, height1, K1, dist1, cam_pos1, R1,
                                     model_reader, return_depth=True,
                                     monitor_resolution=(1920, 1080),
                                     clipping_range=clipping_range)
img1, depth1, E1, N1, U1 = ret

clipping_range = [32, 267]
ret = vtk_util.render_distored_image(width2, height2, K2, dist2, cam_pos2, R2,
                                     model_reader, return_depth=True,
                                     monitor_resolution=(1920, 1080),
                                     clipping_range=clipping_range)
img2, depth2, E2, N, U2 = ret


if False:
    wrld_pts = np.array([[-15.576996, -81.941727, 1.275864],
                         [-15.322388, -82.196335, 1.274765],
                         [-15.322388, -81.941727, 1.271240]]).T
    im_pts1 = project_to_camera(wrld_pts, K1, dist1, R1, cam_pos1)
    im_pts2 = project_to_camera(wrld_pts, K2, dist2, R2, cam_pos2)

    plt.figure()
    plt.imshow(img1)
    plt.plot(im_pts1[:, 0], im_pts1[:, 1], 'go')

    plt.figure()
    plt.imshow(img2)
    plt.plot(im_pts2[:, 0], im_pts2[:, 1], 'go')


    wrld_pts1 = unproject_from_camera_embree(im_pts, K, dist, R, cam_pos,
                                             embree_scene)
    print(wrld_pts1 - wrld_pts)


    wrld_pts1 = unproject_from_camera(im_pts1, K1, dist1, R1, cam_pos1,
                                      depth_map1)
    print(wrld_pts1 - wrld_pts)
    wrld_pts2 = unproject_from_camera(im_pts2, K2, dist2, R2, cam_pos2,
                                      depth_map2)

plt.figure()
plt.imshow(img1)
plt.plot(im_pts1[:, 0], im_pts1[:, 1], 'go')

plt.figure()
plt.imshow(img2)
pts = []
for s in np.linspace(0.999, 1.001, 3):
    wrld_pts1 = unproject_from_camera(im_pts1, K1, dist1, R1, cam_pos1,
                                      depth_map1*s)
    im_pts2 = project_to_camera(wrld_pts1, K2, dist2, R2, cam_pos2)
    pts.append(im_pts2)

pts = np.array(pts)

for i in range(pts.shape[1]):
    im_pts2 = pts[:, i, :]
    plt.plot(im_pts2[:, 0], im_pts2[:, 1], 'r-')

plt.xlim([0, width2])
plt.ylim([height2, 0])
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
from scipy.optimize import fmin, minimize, fminbound

# Colmap Processing imports.
from colmap_processing.geo_conversions import llh_to_enu
from colmap_processing.colmap_interface import read_images_binary, Image, \
    read_points3D_binary, read_cameras_binary, qvec2rotmat
from colmap_processing.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
import colmap_processing.vtk_util as vtk_util
from colmap_processing.geo_conversions import enu_to_llh, llh_to_enu, \
    rmat_ecef_enu, rmat_enu_ecef
from colmap_processing.static_camera_model import save_static_camera, \
    load_static_camera_from_file, write_camera_krtd_file


# ----------------------------------------------------------------------------
# Reference image.
cm_dir = 'uav_video'
image_fname = '%s/ref_view.png' % cm_dir

# File containing correspondences between image points (columns 0 an d1) and
# latitude (degrees), longtigude (degrees), and height above the WGS84
# ellipsoid (meters) (columns 2-4).
points_fname = '%s/points.txt' % cm_dir

save_dir = os.path.split(image_fname)[0]

location = 'khq'

# VTK renderings are limited to monitor resolution (width x height).
monitor_resolution = (1200, 1200)

dist = [0, 0, 0, 0]
optimize_k1 = True
optimize_k2 = False
optimize_k3 = False
optimize_k4 = False
fix_principal_point = False
fix_aspect_ratio = True
# ----------------------------------------------------------------------------


if location == 'khq':
    # Meshed 3-D model used to render an synthetic view for sanity checking and
    # to produce the depth map.
    mesh_fname = 'mesh.ply'
    mesh_lat0 = 42.86453893    # degrees
    mesh_lon0 = -73.77125128  # degrees
    mesh_h0 = 73          # meters above WGS84 ellipsoid

else:
    raise Exception('Unrecognized location \'%s\'' % location)


# Load in point correspondences.
ret = np.loadtxt(points_fname)
im_pts = ret[:, :2]
llh = ret[:, 2:]

enu_pts = [llh_to_enu(_[0], _[1], _[2], mesh_lat0, mesh_lon0, mesh_h0)
           for _ in llh]
enu_pts = np.array(enu_pts)

# Load in the image.
real_image = cv2.imread(image_fname)

if real_image.ndim == 3:
    real_image = real_image[:, :, ::-1]

height, width = real_image.shape[:2]

dist = np.array(dist, dtype=np.float32)

# ------------------------------- First Pass ---------------------------------
# Set optimization parameters for first pass.
flags = cv2.CALIB_ZERO_TANGENT_DIST
flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS

if fix_aspect_ratio:
    flags = flags | cv2.CALIB_FIX_ASPECT_RATIO

flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
flags = flags | cv2.CALIB_FIX_K1
flags = flags | cv2.CALIB_FIX_K2
flags = flags | cv2.CALIB_FIX_K3
flags = flags | cv2.CALIB_FIX_K4
flags = flags | cv2.CALIB_FIX_K5
flags = flags | cv2.CALIB_FIX_K6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000,
            0.0000001)

# Try a bunch of different initial guesses for focal length.
results = []
for f in np.logspace(0, 5, 100):
    K = np.identity(3)
    K[0, 2] = width/2
    K[1, 2] = height/2
    K[0, 0] = K[1, 1] = f
    ret = cv2.calibrateCamera([enu_pts.astype(np.float32)],
                              [im_pts.astype(np.float32)], (width, height),
                              cameraMatrix=K, distCoeffs=dist, flags=flags,
                              criteria=criteria)
    err, K, dist, rvecs, tvecs = ret
    if K[0, 0] > 0 and K[1, 1] > 0:
        results.append(ret)

ind = np.argmin([_[0] for _ in results])
err, K, dist, rvecs, tvecs = results[ind]
print('Error:', err)
rvec, tvec = rvecs[0], tvecs[0]
# ----------------------------------------------------------------------------


# ------------------------------- Second Pass --------------------------------
flags = cv2.CALIB_ZERO_TANGENT_DIST
flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS

if fix_aspect_ratio:
    flags = flags | cv2.CALIB_FIX_ASPECT_RATIO

flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT

if not optimize_k1:
    flags = flags | cv2.CALIB_FIX_K1

if not optimize_k2:
    flags = flags | cv2.CALIB_FIX_K2

if not optimize_k3:
    flags = flags | cv2.CALIB_FIX_K3

if not optimize_k4:
    flags = flags | cv2.CALIB_FIX_K4

flags = flags | cv2.CALIB_FIX_K5
flags = flags | cv2.CALIB_FIX_K6

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000,
            0.0000001)

# Try a bunch of different initial guesses for focal length.
ret = cv2.calibrateCamera([enu_pts.astype(np.float32)],
                          [im_pts.astype(np.float32)], (width, height),
                          cameraMatrix=K, distCoeffs=dist, flags=flags,
                          criteria=criteria)
err, K, dist, rvecs, tvecs = ret
print('Error:', err)
rvec, tvec = rvecs[0], tvecs[0]
# ----------------------------------------------------------------------------


# ------------------------------- Third Pass --------------------------------
flags = cv2.CALIB_ZERO_TANGENT_DIST
flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS

if fix_aspect_ratio:
    flags = flags | cv2.CALIB_FIX_ASPECT_RATIO

if fix_principal_point:
    flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT

if not optimize_k1:
    flags = flags | cv2.CALIB_FIX_K1

if not optimize_k2:
    flags = flags | cv2.CALIB_FIX_K2

if not optimize_k3:
    flags = flags | cv2.CALIB_FIX_K3

if not optimize_k4:
    flags = flags | cv2.CALIB_FIX_K4

flags = flags | cv2.CALIB_FIX_K5
flags = flags | cv2.CALIB_FIX_K6

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000,
            0.0000001)

# Try a bunch of different initial guesses for focal length.
ret = cv2.calibrateCamera([enu_pts.astype(np.float32)],
                          [im_pts.astype(np.float32)], (width, height),
                          cameraMatrix=K, distCoeffs=dist, flags=flags,
                          criteria=criteria)
err, K, dist, rvecs, tvecs = ret
print('Error:', err)
rvec, tvec = rvecs[0], tvecs[0]
# ----------------------------------------------------------------------------
plt.close('all')


def show_err(rvec, tvec, K, dist, plot=False):
    im_pts_ = cv2.projectPoints(enu_pts, rvec, tvec, K, dist)[0]
    im_pts_ = np.squeeze(im_pts_)
    err = np.sqrt(np.sum((im_pts - im_pts_)**2, 1))
    err = np.mean(err)

    if plot:
        plt.imshow(real_image)
        plt.plot(im_pts.T[0], im_pts.T[1], 'bo')
        plt.plot(im_pts_.T[0], im_pts_.T[1], 'ro')
        for i in range(len(im_pts_)):
            plt.plot([im_pts[i, 0], im_pts_[i, 0]],
                     [im_pts[i, 1], im_pts_[i, 1]], 'g-')

    return err


show_err(rvec, tvec, K, dist, plot=True)


R = cv2.Rodrigues(rvec)[0]
cam_pos = -np.dot(R.T, tvec).ravel()


# Read model into VTK.
try:
    model_reader
    assert prev_loaded_fname == mesh_fname
except:
    model_reader = vtk_util.load_world_model(mesh_fname)
    prev_loaded_fname = mesh_fname


# ----------------------------------------------------------------------------

clipping_range = [1, 2000]
ret = vtk_util.render_distored_image(width, height, K, dist, cam_pos, R,
                                     model_reader, return_depth=True,
                                     monitor_resolution=(1920, 1080),
                                     clipping_range=clipping_range)
img, depth, E, N, U = ret


latitude, longitude, altitude = enu_to_llh(cam_pos[0], cam_pos[1], cam_pos[2],
                                           mesh_lat0, mesh_lon0, mesh_h0)

cv2.imwrite('%s/rendered_view.png' % save_dir, img[:, :, ::-1])

# R currently is relative to ENU coordinate system at latitude0,
# longitude0, altitude0, but we need it to be relative to latitude,
# longitude, altitude.
Rp = np.dot(rmat_enu_ecef(mesh_lat0, mesh_lon0),
            rmat_ecef_enu(latitude, longitude))

filename = '%s/camera_model.yaml' % save_dir
save_static_camera(filename, height, width, K, dist, np.dot(R, Rp),
                   depth, latitude, longitude, altitude)

# Save (x, y, z) meters per pixel.
if False:
    filename = '%s/camera_model.krtd' % save_dir
    tvec = -np.dot(R, cam_pos).ravel()
    write_camera_krtd_file([K, R, tvec, dist], filename)

    ENU = np.dstack([E, N, U]).astype(np.float32)
    filename = '%s/xyz_per_pixel.npy' % save_dir
    np.save(filename, ENU, allow_pickle=False)

depth_image = depth.copy()
depth_image[depth_image > clipping_range[1]*0.9] = 0
depth_image -= depth_image.min()
depth_image /= depth_image.max()/255
depth_image = np.round(depth_image).astype(np.uint8)

depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
cv2.imwrite('%s/depth_vizualization.png' % save_dir,
            depth_image[:, :, ::-1])
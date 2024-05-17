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

This script helps to georegister the 3-D reconstruction coordinate system used
by Colmap. We assume that enough of the scene is jointly visible (e.g., enough
high-altitude views) so that georegistration is a rigid rotation and isotropic
scaling.

(1) The first step is to geo-register keypoints within images. This is done by
first saving images with keypoints associated with a sparse-reconstructed 3-D
point superimposed. These images should be loaded into the
landmark_registration GUI. The ENU origin should be added as a point with
latitude and longitude both zero.


"""
from __future__ import division, print_function, absolute_import
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import time
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import threading
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Colmap Processing imports.
from colmap_processing.geo_conversions import llh_to_enu, enu_to_llh
from colmap_processing.colmap_interface import read_images_binary, Image, \
    read_points3D_binary, read_cameras_binary, qvec2rotmat
from colmap_processing.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array


# ----------------------------------------------------------------------------
# Base path to the colmap directory.
data_dir = '/mnt/data'
result_fname = '%s/georegistration_matrix.txt' % data_dir


# Load level_points.txt, which encodes all points that are at the same
# elevation. The file should contain an image ID followed by the image
# coordinates of the point.
level_pts = np.loadtxt('%s/level_points.txt' % data_dir).T

# Load point_lower_higher.txt, where the first point has a lower elevation than
# the second.
xyz1, xyz2 = np.loadtxt('%s/point_lower_higher.txt' % data_dir)
v_up = xyz2 - xyz1

# Rows where first three are model x,y,z coordinates and the last two elements
# are the latitude and longitude in degrees.
tmp = np.loadtxt('%s/points.txt' % data_dir)
latlonalt = tmp[:, 3:]
model_xyz = tmp[:, :3].T
origin_xyz = model_xyz[:, 0]


def level_err(x):
    R = cv2.Rodrigues(x[:3])[0]

    if np.dot(R, v_up)[2] < 0:
        return 1e10

    xyz = np.dot(R, level_pts)
    #err = max(xyz[2]) - min(xyz[2])
    err = np.std(xyz[2])
    #print(x, err)
    return err


# Solve for optimal transform.
best_err = np.inf
for _ in range(100):
    x = (np.random.rand(3)*2 - 1)*np.pi

    for _ in range(5):
        x = minimize(level_err, x, tol=1e-12).x

    err_ = level_err(x)
    if err_ < best_err:
        best_x = x
        best_err = err_

print('Final level error:', best_err)


# Rotation matrix that puts model right-side up and level.
R = cv2.Rodrigues(best_x)[0]

if False:
    # Show points plotted on image.
    xyz = np.dot(R, level_pts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(xyz[0], xyz[1], xyz[2], '.')
    plt.xlabel('Model X')
    plt.ylabel('Model Y')


if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(model_xyz[0], model_xyz[1], model_xyz[2], '.')
    plt.xlabel('Model X')
    plt.ylabel('Model Y')

# Pick a reasonable origin to start.
lat0, lon0 = np.mean(latlonalt, axis=0)[:2]

easting_northing = np.array([llh_to_enu(_[0], _[1], 0, lat0, lon0, 0)[:2]
                            for _ in latlonalt]).T

if False:
    fig = plt.figure()
    plt.plot(easting_northing[0], easting_northing[1], '.')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')

# Create homogenous 2-D leveled version of model points.
model_xyz_leveled = np.dot(R, model_xyz)
model_xyz_leveled[2] = 1

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(model_xyz_leveled[0], model_xyz_leveled[1], model_xyz_leveled[2],
             '.')
    plt.xlabel('Model X')
    plt.ylabel('Model Y')


def get_m(x):
    theta = x[0]
    scale = x[1]
    tx = x[2]
    ty = x[3]

    c = np.cos(theta)
    s = np.sin(theta)

    R_ = np.identity(4)
    R_[:2, :2] = np.array([[c, -s], [s, c]])

    S_ = np.identity(4)
    S_[:3, :3] *= scale

    T_ = np.identity(4)
    T_[0, 3] = -tx
    T_[1, 3] = -ty

    return np.dot(T_, np.dot(S_, R_))


def align_err(x):
    if x[1] < 0:
        return 1e10

    m = get_m(x)
    m = m[:, [0, 1, 3]]
    err = easting_northing - np.dot(m, model_xyz_leveled)[:2]
    err = np.sqrt(np.mean(np.sum(err**2, axis=0)))
    print('Error in meters rms:', err)
    return err


def plot_err(x):
    m = get_m(x)
    m = m[:, [0, 1, 3]]
    pts1 = np.dot(m, model_xyz_leveled)[:2]
    plt.plot(pts1[0], pts1[1], 'ro')
    plt.plot(easting_northing[0], easting_northing[1], 'bo')
    for i in range(pts1.shape[1]):
        plt.plot([pts1[0], easting_northing[0]],
                 [pts1[1], easting_northing[1]], 'b-')


# Solve for optimal transform.
x = np.array([np.random.rand(1)[0]*np.pi*2, 10, 0, 0])
for _ in range(4):
    x = minimize(align_err, x).x

M2 = get_m(x)
M1 = np.identity(4)
M1[:3, :3] = R
M = np.dot(M2, M1)

# M now warps from model coordinates to east/northing/up at lat0 and lon0. But,
# we want to update lat0, lon0, and h0 such that origin_xyz is (0, 0, 0).
enu0 = np.dot(M, np.hstack([origin_xyz, 1]))[:3]
lat0, lon0, h0 = enu_to_llh(enu0[0], enu0[1], enu0[2], lat0, lon0, 0)

if False:
    # Define origin_xy
    llh_to_enu(enu0[0], enu0[1], enu0[2], lat0, lon0, 0)

M[:3, 3] -= np.dot(M, np.hstack([origin_xyz, 1]))[:3]

print('Should be all zeros', np.dot(M, np.hstack([origin_xyz, 1]))[:3])

print('Transformation from model coordinates to east/north/up meters relative '
      'to the origin at latitude=%0.8f, latitude=%0.8f, height=%0.3f meters is'
      % (lat0, lon0, h0))
print(M)


with open(result_fname, 'w') as f:
    f.write('# This registration matrix accepts coordinates in the '
            'coordinates system used by\n# the structure-from-motion '
            'reconstruction and returns easting, northing, up\n# coordinates '
            'in meters. Therefere, it applies to the reconstructed poses of '
            'the\n# cameras and sparse and dense point reconstructions.\n\n')
    for i in range(4):
        f.write('%0.10f %0.10f %0.10f %0.10f\n' % (tuple(M[i])))

    f.write('\n# The origin of the east-north-up coordinate system is located '
            'at:\nlatitude0 = %0.8f    # degrees\nlongitude0 = %0.8f  # '
            'degrees\nheight0 = %0.3f          # meters above WGS84 '
            'ellipsoid\n' % (lat0, lon0, h0))

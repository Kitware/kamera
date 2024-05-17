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
from osgeo import osr, gdal
from scipy.optimize import fmin, minimize, fminbound
from scipy.linalg import lstsq
import itertools

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
from colmap_processing.calibration import calibrate_camera_to_xyz, \
    cam_depth_map_plane
from colmap_processing.camera_models import StandardCamera, DepthCamera
from colmap_processing.camera_models import quaternion_from_matrix, \
    quaternion_inverse
from colmap_processing.platform_pose import PlatformPoseFixed


# ----------------------------------------------------------------------------
# Reference image.
cm_dir = '/mnt'
image_fname = '%s/ref_view.jpg' % cm_dir

# File containing correspondences between image points (columns 0 an d1) and
# easting (m), northing (m), and height above the WGS84 ellipsoid (meters)
# (columns 2-4).
points_fname = '%s/points.txt' % cm_dir

points_in_meters = True

mesh_lat0 = 39.04977294
mesh_lon0 = -85.52924953
mesh_h0 = 205

# Assume flat ground with this elevation.
ground_elevation = -1.458672

# COCO json encoding vertical and horizontal lines. .
lines_fname = '%s/lines.json' % cm_dir

save_dir = os.path.split(image_fname)[0]

# VTK renderings are limited to monitor resolution (width x height).
monitor_resolution = (1000, 1000)

dist = [0, 0, 0, 0]
optimize_k1 = True
optimize_k2 = True
optimize_k3 = False
optimize_k4 = False
fix_principal_point = False
fix_aspect_ratio = True
# ----------------------------------------------------------------------------

# Load in point correspondences.
ret = np.loadtxt(points_fname)
im_pts = ret[:, :2].T
pts = ret[:, 2:]

if points_in_meters:
    enu_pts = np.array(pts, dtype=np.float32).T
else:
    enu_pts = [llh_to_enu(_[0], _[1], _[2], mesh_lat0, mesh_lon0, mesh_h0)
               for _ in pts]
    enu_pts = np.array(enu_pts, dtype=np.float32).T

if False:
    enu_pts = enu_pts.T.tolist()
    im_pts = im_pts.T.tolist()
    for i in [0, 1, 2, 3]:
        for _ in range(20):
            enu_pts.append(enu_pts[i])
            im_pts.append(im_pts[i])

    enu_pts = np.array(enu_pts).T
    im_pts = np.array(im_pts).T

with open(lines_fname) as json_file:
    lines0 = json.load(json_file)

lines = [np.array(an['segmentation']).reshape(-1, 2).T
         for an in lines0['annotations']]

# Load in the image.
img = cv2.imread(image_fname)

if img.ndim == 3:
    img = img[:, :, ::-1]

height, width = img.shape[:2]

dist = np.array(dist, dtype=np.float32)
# ----------------------------------------------------------------------------


# ------------------------------- Visualize ----------------------------------
# This is a hack, we need better specification of the vertical versus
# horizontal lines.

vert_lines = [lines[i] for i in [0, 1, 2, 3, 4, 5]]
horz_lines = [lines[i] for i in [6, 7, 8, 9, 10, 11, 12]]

plt.imshow(img)
for line in vert_lines:
    plt.plot(line[0], line[1], 'b-', linewidth=4)

for line in horz_lines:
    plt.plot(line[0], line[1], 'r-', linewidth=4)
# ----------------------------------------------------------------------------


#--------------------- Find a Reasonable Starting Point ----------------------
# Spoof more points to stabilize the process.
num_pts = im_pts.shape[1]
im_pts_spoof  = []
enu_pts_spoof  = []

for i, j in list(itertools.combinations(range(num_pts), 2)):
    for t in np.linspace(0, 1, 2):
        im_pts_spoof.append(im_pts[:, i]*t + im_pts[:, j]*(1 - t))
        enu_pts_spoof.append(enu_pts[:, i]*t + enu_pts[:, j]*(1 - t))

im_pts_spoof = np.array(im_pts_spoof)
enu_pts_spoof = np.array(enu_pts_spoof)

ret0 = calibrate_camera_to_xyz(im_pts_spoof, enu_pts_spoof, img.shape[0],
                              img.shape[1], fix_aspect_ratio=True,
                              fix_principal_point=False, fix_k1=True,
                              fix_k2=True, fix_k3=True, fix_k4=True,
                              fix_k5=True, fix_k6=True, plot_results=False,
                              ref_image=None)
K, dist, rvec, tvec = ret0
rvec = rvec.ravel()
tvec = tvec.ravel()
R = cv2.Rodrigues(rvec)[0]
cam_pos = np.dot(R.T, -tvec)


def unproject_lines(xy, K, dist, rvec, tvec, plot=False):
    ray_dir = np.ones((3, xy.shape[1]), dtype=np.float)
    ray_dir[:2] = np.squeeze(cv2.undistortPoints(xy, K, dist, R=None), 1).T
    ray_dir /=  np.sqrt(np.sum(ray_dir**2, 0))
    R = cv2.Rodrigues(rvec)[0]
    cam_pos = np.dot(R.T, -tvec)
    xyz = np.atleast_2d(cam_pos).T + np.dot(R.T, ray_dir)

    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = xyz.mean(axis=1)

    demeaned = (xyz.T - datamean)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(demeaned)
    axis = vv[0]
    xyz_straightened = np.atleast_2d(datamean).T + np.dot(demeaned, axis)*np.atleast_2d(axis).T

    if plot:
        plt.close('all')
        plt.imshow(img)
        im_pts_ = cv2.projectPoints(xyz, rvec, tvec, K, dist)[0]
        im_pts_ = np.squeeze(im_pts_).T
        plt.plot(im_pts_[0], im_pts_[1], 'r.')
        plt.plot(xy[0], xy[1], 'b.')
        im_pts_ = cv2.projectPoints(xyz_straightened, rvec, tvec, K, dist)[0]
        im_pts_ = np.squeeze(im_pts_).T
        plt.plot(im_pts_[0], im_pts_[1], 'g.')
        np.sqrt(np.sum((xy - im_pts_)**2, 0))

    return xyz_straightened


# Make straight lines straight.
last_err = None
for _  in range(100):
    xys = im_pts_spoof.copy().T
    xyzs = enu_pts_spoof.copy().T
    for xy in vert_lines + horz_lines:
        xys = np.hstack([xys, xy])
        xyzs = np.hstack([xyzs, unproject_lines(xy, K, dist, rvec, tvec)])


    xyzs = np.array(xyzs)
    xys = np.array(xys)


    flags = cv2.CALIB_ZERO_TANGENT_DIST
    flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
    #flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
    #flags = flags | cv2.CALIB_FIX_ASPECT_RATIO
    #flags = flags | cv2.CALIB_FIX_K1
    flags = flags | cv2.CALIB_FIX_K2
    flags = flags | cv2.CALIB_FIX_K3
    flags = flags | cv2.CALIB_FIX_K4
    flags = flags | cv2.CALIB_FIX_K5
    flags = flags | cv2.CALIB_FIX_K6
    ret = cv2.calibrateCamera([xyzs.T.astype(np.float32)],
                              [xys.T.astype(np.float32)],
                              (width, height), cameraMatrix=K.copy(),
                              distCoeffs=dist.copy(), flags=flags)
    if last_err is not None and ret[0] > last_err:
        break

    err, K, dist, rvecs, tvecs = ret
    last_err = err
    print(err)
    rvec = rvecs[0].ravel()
    tvec = tvecs[0].ravel()
    R = cv2.Rodrigues(rvec)[0]
    cam_pos = np.dot(R.T, -tvec)


if False:
    im_pts_ = cv2.projectPoints(xyzs, rvec, tvec, K, dist)[0]
    im_pts_ = np.squeeze(im_pts_).T
    plt.imshow(img)
    plt.plot(im_pts_[0], im_pts_[1], 'r.')
    plt.plot(xys[0], xys[1], 'b.')


if False:
    im_pts_ = cv2.projectPoints(enu_pts_spoof, rvec, tvec, K, dist)[0]
    im_pts_ = np.squeeze(im_pts_).T
    plt.imshow(img)
    plt.plot(im_pts_[0], im_pts_[1], 'r.')
    plt.plot(im_pts_spoof.T[0], im_pts_spoof.T[1], 'b.')


if False:
    plt.close('all')
    im_pts_ = cv2.projectPoints(enu_pts.T, rvec, tvec, K, dist)[0]
    im_pts_ = np.squeeze(im_pts_).T
    plt.imshow(img)
    plt.plot(im_pts_[0], im_pts_[1], 'r.')
    plt.plot(im_pts[0], im_pts[1], 'b.')
# ----------------------------------------------------------------------------


# ------------------------------- First Pass ---------------------------------
def get_params_from_x(x):
    rvec = x[:3]
    tvec = x[3:6]
    fx = abs(x[6])
    fy = abs(x[7])

    dist = np.zeros(5)
    if len(x) > 8:
        dist[0] = x[8]

    if len(x) > 9:
        dist[1] = x[9]

    if len(x) > 10:
        dist[2] = x[10]

    R = cv2.Rodrigues(rvec)[0]

    K = np.identity(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = img.shape[1]/2
    K[1, 2] = img.shape[0]/2

    return K, rvec, R, tvec, dist


def error(x):
    #print('x', x)
    K, rvec, R, tvec, dist = get_params_from_x(x)

    cam_pos = np.dot(R.T, -tvec)

    # Project in truthed enu points.
    im_pts_ = cv2.projectPoints(enu_pts.T, rvec, tvec, K, dist)[0]
    im_pts_ = np.squeeze(im_pts_)

    ray_dir0 = (enu_pts.T - cam_pos).T
    ray_dir0 /=  np.sqrt(np.sum(ray_dir0**2, 0))
    ray_dir0 = np.dot(R, ray_dir0)

    if np.any(ray_dir0[2] < 0):
        return 1e10

    ray_dir = np.ones((3, len(im_pts_)), dtype=np.float)
    ray_dir[:2] = np.squeeze(cv2.undistortPoints(im_pts_, K, dist, R=None), 1).T
    ray_dir /=  np.sqrt(np.sum(ray_dir**2, 0))

    if np.any(np.sum(ray_dir * ray_dir0, axis=0) < 0.99):
        return 1e10

    err = np.mean(np.sqrt(np.sum((im_pts.T - im_pts_)**2, 1)))

    print('Reproj error:', err)

    #err *= 10

    # Unproject rays into the camera coordinate system.
    for xy in vert_lines:
        ray_dir = np.ones((3, xy.shape[1]), dtype=np.float)
        ray_dir0 = cv2.undistortPoints(xy.T, K, dist, R=None)
        ray_dir[:2] = np.squeeze(ray_dir0, 1).T
        ray_dir = np.dot(R.T, ray_dir)
        ray_pos =  ray_dir/np.sqrt(np.sum(ray_dir**2, 0))
        err -= sum(np.diff(ray_pos[2]))*1000

    # Unproject rays into the camera coordinate system.
    for xy in horz_lines:
        ray_dir = np.ones((3, xy.shape[1]), dtype=np.float)
        ray_dir0 = cv2.undistortPoints(xy.T, K, dist, R=None)
        ray_dir[:2] = np.squeeze(ray_dir0, 1).T
        ray_dir = np.dot(R.T, ray_dir)

        # we need to pick a set of point ranges so that all points end up at
        # the same z value. The degenerate solution is that all are at range 0,
        # so we force the first point to be at range 1.
        ray_dir /=  np.sqrt(np.sum(ray_dir**2, 0))

        z = np.mean(ray_dir[2])
        d = z/ray_dir[2]
        xyz = ray_dir*d

        # These should be lines
        x, y = xyz[:2]

        p = np.polyfit(x, y, 1)
        y2 = np.polyval(p, x)

        if False:
            plt.plot(x, y, 'r.')
            plt.plot(x, y2, 'b.')

        err += sum(abs(y - y2))*1e6

    print(err)
    return err


"""
best_err = np.inf
best_x = None
for _ in range(100):
    rvec = np.random.rand(3)

    # Decide on a reasonable tvec.
    R = cv2.Rodrigues(rvec)[0]
    tvec = -np.dot(R, cam_pos)


    tvec = enu_pts[0] + np.random.rand(3)*10

    f = np.random.rand()*10000
    x = np.hstack([rvec, tvec, f])
    x = minimize(error, x).x
    err = error(x)
    if err < best_err:
        best_err = err
        best_x = x
"""

#dist = np.zeros(5)
x = np.hstack([rvec.ravel(), tvec.ravel(), K[0, 0], K[1, 1], dist[0], dist[1]])

error(x)

def err1(k1):
    x_ = x.copy()
    x_[-2] = k1

    def err2(k2):
        x__ = x_.copy()
        x__[-1] = k2
        return error(x__)

    ret = fminbound(err2, -1000, 1000)

    return err2(ret)

k1 = fminbound(err1, -10, 10)
x_ = x.copy();  x_[-2] = k1

if error(x) > error(x_):
    x = x_

def err2(k2):
    x_ = x.copy()
    x_[-1] = k2
    return error(x_)

k2 = fminbound(err2, -100, 100)
x_ = x.copy();  x_[-1] = k1

if error(x) > error(x_):
    x = x_

K, rvec, R, tvec, dist = get_params_from_x(x)

flags = cv2.CALIB_ZERO_TANGENT_DIST
flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
#flags = flags | cv2.CALIB_FIX_ASPECT_RATIO
flags = flags | cv2.CALIB_FIX_K1
flags = flags | cv2.CALIB_FIX_K2
flags = flags | cv2.CALIB_FIX_K3
flags = flags | cv2.CALIB_FIX_K4
flags = flags | cv2.CALIB_FIX_K5
flags = flags | cv2.CALIB_FIX_K6
ret = cv2.calibrateCamera([enu_pts_spoof.astype(np.float32)],
                          [im_pts_spoof.astype(np.float32)],
                          (width, height), cameraMatrix=K.copy(),
                          distCoeffs=dist.copy(), flags=flags)

x_ = np.hstack([ret[3][0].ravel(), ret[4][0].ravel(), ret[1][0, 0],
                ret[1][1, 1], ret[2][0], ret[2][1]])
if error(x) > error(x_):
    x = x_

print(error(x))
K, rvec, R, tvec, dist = get_params_from_x(x)


if False:
    for _  in range(10):
        x = minimize(error, x).x


K, rvec, R, tvec, dist = get_params_from_x(x)
cam_pos = np.dot(R.T, -tvec)

undistorted_frame = cv2.undistort(img, K, dist)
plt.imshow(undistorted_frame)


if False:
    im_pts_ = cv2.projectPoints(enu_pts, rvec, tvec, K, dist)[0]
    im_pts_ = np.squeeze(im_pts_).T
    plt.imshow(img)
    plt.plot(im_pts_[0], im_pts_[1], 'ro')
    plt.plot(im_pts[0], im_pts[1], 'bo')


write_camera_krtd_file([K, R, tvec, dist], '%s/camera_model.krtd' % save_dir)


latitude, longitude, altitude = enu_to_llh(cam_pos[0], cam_pos[1],
                                           cam_pos[2], mesh_lat0,
                                           mesh_lon0, mesh_h0)


# Save camera model specification. Remembering that 'camera_models' considers
# quaterions to be coordinate system rotations not transformations. So, we have
# to invert the standard computer vision rotation matrix to create the
# orientation quaterions.
quat = quaternion_inverse(quaternion_from_matrix(R))
pos = -np.dot(R.T, tvec).ravel()
platform_pose_provider = PlatformPoseFixed(pos, quat)
cm = StandardCamera(width, height, K, dist, [0, 0, 0], [0, 0, 0, 1],
                    platform_pose_provider)
depth = cam_depth_map_plane(cm, [0, 0, ground_elevation], [0, 0, -1])
cm = DepthCamera(width, height, K, dist, [0, 0, 0], [0, 0, 0, 1], depth,
                 platform_pose_provider)

# R currently is relative to ENU coordinate system at latitude0,
# longitude0, altitude0, but we need it to be relative to latitude,
# longitude, altitude.
Rp = np.dot(rmat_enu_ecef(mesh_lat0, mesh_lon0),
            rmat_ecef_enu(latitude, longitude))

save_static_camera('%s/camera_model.yaml' % save_dir, height, width, K, dist,
                   np.dot(R, Rp), depth, latitude, longitude, altitude)

x = np.linspace(0, width, width + 1)
y = np.linspace(0, height, height + 1)
x = (x[1:] + x[:-1])/2
y = (y[1:] + y[:-1])/2
X, Y = np.meshgrid(x, y)
im_pts = np.vstack([X.ravel(), Y.ravel()])
xyz = cm.unproject_to_depth(im_pts)
E = xyz[0].reshape(X.shape)
N = xyz[1].reshape(X.shape)
U = xyz[2].reshape(X.shape)
ENU = np.dstack([E, N, U]).astype(np.float32)
filename = '%s/xyz_per_pixel.npy' % save_dir
np.save(filename, ENU, allow_pickle=False)
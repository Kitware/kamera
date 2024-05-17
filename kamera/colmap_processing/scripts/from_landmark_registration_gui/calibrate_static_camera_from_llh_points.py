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
    read_points3d_binary, read_cameras_binary, qvec2rotmat
from colmap_processing.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
import colmap_processing.vtk_util as vtk_util
from colmap_processing.geo_conversions import enu_to_llh, llh_to_enu, \
    rmat_ecef_enu, rmat_enu_ecef
from colmap_processing.static_camera_model import save_static_camera, \
    load_static_camera_from_file, write_camera_krtd_file
from colmap_processing.camera_models import StandardCamera, DepthCamera
from colmap_processing.calibration import calibrate_camera_to_xyz, \
    cam_depth_map_plane
from colmap_processing.platform_pose import PlatformPoseFixed
from colmap_processing.camera_models import quaternion_from_matrix, \
    quaternion_inverse


# ----------------------------------------------------------------------------
# Landmark annotation GUI workspace path.
landmark_gui_workspace = '/home/user/libraries/georegistration_guis/data/workspace'

# Image Directory.
image_dir = '/home/user/libraries/georegistration_guis/data/frames'

# Directory to save camera models within.
save_dir = '/home/user/libraries/georegistration_guis/data/camera_models'

# Define the origin of the easting/northing/up coordinate system.
lat0 = 0
lon0 = 0
h0 = 0

# VTK renderings are limited to monitor resolution (width x height).
monitor_resolution = (2200, 1200)

dist = [0, 0, 0, 0]
optimize_k1 = True
optimize_k2 = False
optimize_k3 = False
optimize_k4 = False
fix_principal_point = False
fix_aspect_ratio = True
# ----------------------------------------------------------------------------


image_fnames = []
for ext in ['.jpg', '.tiff', '.png', '.bmp', '.jpeg']:
    image_fnames = image_fnames + glob.glob('%s/*%s' % (image_dir, ext))

img_fname_to_cam_id = {}
img_id_to_cam_id = {}
img_cam_id_to_fname = {}
img_fname_to_img_id = {}
camera_models = {}
images = {}
for image_id, fname in enumerate(image_fnames):
    image_id = image_id + 1
    img = cv2.imread(fname)

    if img.ndim == 3:
        img = img[:, :, ::-1]

    image_fname = os.path.splitext(os.path.split(fname)[1])[0]

    height, width = img.shape[:2]

    # Every image is from a different camera.
    camera_id = image_id

    img_fname_to_img_id[image_fname] = image_id
    img_cam_id_to_fname[image_id] = image_fname
    img_fname_to_cam_id[image_fname] = camera_id
    img_id_to_cam_id[image_id] = camera_id
    images[image_id] = img

    # Initialize with dummy values.
    camera_models[camera_id] = StandardCamera(width, height, np.identity(3),
                                              np.zeros(4),  np.zeros(3),
                                              [0, 0, 0, 1])

# ----------------------------------------------------------------------------
# Parse landmark annotation GUI workspace.

# img_keypoints is a dictionary that accepts the integer image index and
# returns a list with the first element being the indices into landmarks and
# the second element is the associated image coordinates for each landmark.
img_keypoints = {}
for image_id, fname in enumerate(image_fnames):
    img_fname = os.path.splitext(os.path.split(fname)[1])[0]
    fname = ('%s/%s_image_points.txt' % (landmark_gui_workspace, img_fname))
    try:
        points = np.loadtxt(fname)
    except (OSError, IOError):
        points = np.zeros((0, 3))

    image_id = img_fname_to_img_id[img_fname]
    camera_id = img_fname_to_cam_id[img_fname]

    img_keypoints[image_id] = [points[:, 0].astype(np.int), points[:, 1:]]

points_fname = '%s/ground_control_points.txt' % landmark_gui_workspace
ret = np.loadtxt(points_fname)
enu_pts = {int(ret[i, 0]): llh_to_enu(ret[i, 1], ret[i, 2], ret[i, 3], lat0,
                                      lon0, h0)
           for i in range(len(ret))}


def calibrate_camera_id(image_id, save_dir_, fix_aspect_ratio=True,
                        fix_principal_point=True, fix_k1=True, fix_k2=True,
                        fix_k3=True, fix_k4=True, fix_k5=True, fix_k6=True):
    print(img_cam_id_to_fname[image_id])
    l_ind, im_pts0 = img_keypoints[image_id]
    wrld_pts = []
    im_pts = []
    # Only collect keypoints with an associated easting/northing/up position.
    for i in range(len(l_ind)):
        if l_ind[i] in enu_pts:
            if abs(enu_pts[l_ind[i]][2]) < 0.01:
                wrld_pts.append(enu_pts[l_ind[i]])
                im_pts.append(im_pts0[i])

    im_pts = np.array(im_pts)
    wrld_pts = np.array(wrld_pts)
    ref_image = images[image_id]
    cm0 = camera_models[img_id_to_cam_id[image_id]]
    height = cm0.height
    width = cm0.width

    ret = calibrate_camera_to_xyz(im_pts, wrld_pts, height, width,
                                  fix_aspect_ratio=fix_aspect_ratio,
                                  fix_principal_point=fix_principal_point,
                                  fix_k1=fix_k1, fix_k2=fix_k2, fix_k3=fix_k3,
                                  fix_k4=fix_k4, fix_k5=fix_k5, fix_k6=fix_k6,
                                  plot_results=False, ref_image=ref_image)

    K, dist, rvec, tvec = ret

    # Save camera model specification. Remembering that 'camera_models' considers
    # quaterions to be coordinate system rotations not transformations. So, we have
    # to invert the standard computer vision rotation matrix to create the
    # orientation quaterions.
    R = cv2.Rodrigues(rvec)[0]
    quat = quaternion_inverse(quaternion_from_matrix(R))
    pos = -np.dot(R.T, tvec).ravel()
    platform_pose_provider = PlatformPoseFixed(pos, quat)
    cm = StandardCamera(width, height, K, dist, [0, 0, 0], [0, 0, 0, 1],
                        platform_pose_provider)

    # Assume level plane at z = 0.
    plane_point = [0, 0, 0]
    plane_normal = [0, 0, 1]

    depth_map = cam_depth_map_plane(cm, plane_point, plane_normal)

    cm = DepthCamera(width, height, K, dist, [0, 0, 0], [0, 0, 0, 1], depth_map,
                     platform_pose_provider)

    camera_models[img_id_to_cam_id[image_id]] = cm

    # Save camera model.
    try:
        os.makedirs(save_dir_)
    except (OSError, IOError):
        pass

    if True:
        # Sanity check.
        err = np.sqrt(np.sum((im_pts.T - cm.project(wrld_pts.T))**2, axis=0))
        err = np.sort(err)
        print('Min / Mean / Max reprojection error: (%0.3f, %0.3f, %0.3f) '
              'pixels' % (err.min(), np.mean(err), err.max()))

        np.savetxt('%s/projection_pixel_err.txt' % save_dir_, err, fmt='%.3f')

        wrld_pts2 = cm.unproject_to_depth(im_pts.T)
        err = np.sqrt(np.sum((wrld_pts2 - wrld_pts.T)**2, axis=0))
        err = np.sort(err)
        print('Min / Mean / Max unprojection error: (%0.3f, %0.3f, %0.3f) '
              'meters' % (err.min(), np.mean(err), err.max()))

        np.savetxt('%s/unprojection_meters_err.txt' % save_dir_, err,
                   fmt='%.3f')

    fname = '%s/camera_model.yaml' % save_dir_
    cm.save_to_file(fname)

    if ref_image.ndim == 3:
        cv2.imwrite('%s/ref_view.jpg' % save_dir_, ref_image[:, :, ::-1])
        cv2.imwrite('%s/undistorted.jpg' % save_dir_,
                    cv2.undistort(ref_image[:, :, ::-1], K, dist))
    else:
        cv2.imwrite('%s/ref_view.jpg' % save_dir_, ref_image)
        cv2.imwrite('%s/undistorted.jpg' % save_dir_,
                    cv2.undistort(ref_image, K, dist))

    filename = '%s/camera_model.krtd' % save_dir_
    write_camera_krtd_file([K, R, tvec, dist], filename)

    X, Y = np.meshgrid(np.linspace(0.5, cm.width - .5, cm.width),
                       np.linspace(0.5, cm.height - .5, cm.height))
    X, Y, Z = cm.unproject_to_depth(np.vstack([X.ravel(), Y.ravel()]))
    X.shape = (cm.height, cm.width)
    Y.shape = (cm.height, cm.width)
    Z.shape = (cm.height, cm.width)

    XYZ = np.dstack([X, Y, Z]).astype(np.float32)
    filename = '%s/xyz_per_pixel.npy' % save_dir_
    np.save(filename, XYZ, allow_pickle=False)

    plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
    font = {'size' : 26}
    plt.rc('font', **font)
    plt.rc('axes', linewidth=4)
    plt.imshow(ref_image)
    im_pts2 = cm.project(wrld_pts.T).T
    for i in range(len(im_pts)):
        plt.plot(im_pts[i, 0], im_pts[i, 1], 'bo')
        plt.plot(im_pts2[i, 0], im_pts2[i, 1], 'ro')
        plt.plot([im_pts[i, 0], im_pts2[i, 0]],
                 [im_pts[i, 1], im_pts2[i, 1]], 'k--')

    plt.savefig('%s/model_pixel_error.pdf' % save_dir_)

    plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
    font = {'size' : 40}
    plt.rc('font', **font)
    plt.rc('axes', linewidth=4)
    wrld_pts2 = cm.unproject_to_depth(im_pts.T).T
    for i in range(len(im_pts)):
        plt.plot(wrld_pts[i, 0], wrld_pts[i, 1], 'bo')
        plt.plot(wrld_pts2[i, 0], wrld_pts2[i, 1], 'ro')
        plt.plot([wrld_pts[i, 0], wrld_pts2[i, 0]],
                 [wrld_pts[i, 1], wrld_pts2[i, 1]], 'k--')

    plt.xlabel('X (meters)', fontsize=40)
    plt.ylabel('Y (meters)', fontsize=40)

    plt.savefig('%s/model_meters_error.pdf' % save_dir_)


if True:
    img_to_process = ['images',]
    image_ids = [img_fname_to_cam_id[i] for i in img_to_process]
else:
    image_ids = images.keys()

for image_id in image_ids:
    try:
        plt.close('all')
        save_dir_ = '%s/%s_1' % (save_dir, img_cam_id_to_fname[image_id])
        calibrate_camera_id(image_id, save_dir_, fix_aspect_ratio=True,
                            fix_principal_point=True, fix_k1=True, fix_k2=True,
                            fix_k3=True, fix_k4=True, fix_k5=True, fix_k6=True)

        if False:
            plt.close('all')
            save_dir_ = '%s/%s_2' % (save_dir, img_cam_id_to_fname[image_id])
            calibrate_camera_id(image_id, save_dir_, fix_aspect_ratio=False,
                                fix_principal_point=True, fix_k1=False,
                                fix_k2=True, fix_k3=True, fix_k4=True,
                                fix_k5=True, fix_k6=True)

        if False:
            plt.close('all')
            save_dir_ = '%s/%s_3' % (save_dir, img_cam_id_to_fname[image_id])
            calibrate_camera_id(image_id, save_dir_, fix_aspect_ratio=False,
                                fix_principal_point=False, fix_k1=False,
                                fix_k2=True, fix_k3=True, fix_k4=True,
                                fix_k5=True, fix_k6=True)

        if False:
            plt.close('all')
            save_dir_ = '%s/%s_4' % (save_dir, img_cam_id_to_fname[image_id])
            calibrate_camera_id(image_id, save_dir_, fix_aspect_ratio=False,
                                fix_principal_point=False, fix_k1=False,
                                fix_k2=False, fix_k3=True, fix_k4=True,
                                fix_k5=True, fix_k6=True)
    except:
        pass
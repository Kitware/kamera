#! /usr/bin/python
from __future__ import division, print_function
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from osgeo import osr, gdal
from scipy.optimize import fmin, minimize, fminbound
import transformations

# Colmap Processing imports.
from colmap_processing.geo_conversions import llh_to_enu
from colmap_processing.colmap_interface import read_images_binary, Image, \
    read_points3d_binary, read_cameras_binary, qvec2rotmat
from colmap_processing.camera_models import StandardCamera
from colmap_processing.platform_pose import PlatformPoseInterp


# ----------------------------------------------------------------------------
# Define the directory where all of the relavant COLMAP files are saved.
# If you are placing your data within the 'data' folder of this repository,
# this will be mapped to '/home_user/adapt_postprocessing/data' inside the
# Docker container.
data_dir = '/media/data'

image_subdirs = ['1', '2']

# COLMAP data directory.
images_bin_fname = '%s/images.bin' % data_dir
camera_bin_fname = '%s/cameras.bin' % data_dir
points_bin_fname = '%s/points3D.bin' % data_dir
# ----------------------------------------------------------------------------


# Read in the details of all images.
images = read_images_binary(images_bin_fname)
cameras = read_cameras_binary(camera_bin_fname)
points = read_points3d_binary(points_bin_fname)


# Pretend image index is the time.
platform_pose_provider = PlatformPoseInterp()
for image_id in images:
    image = images[image_id]

    R = qvec2rotmat(image.qvec)
    pos = -np.dot(R.T, image.tvec)

    # The qvec used by Colmap is a (w, x, y, z) quaternion representing the
    # rotation of a vector defined in the world coordinate system into the
    # camera coordinate system. However, the 'camera_models' module assumes
    # (x, y, z, w) quaternions representing a coordinate system rotation.
    quat = transformations.quaternion_inverse(image.qvec)
    quat = [quat[1], quat[2], quat[3], quat[0]]

    t = image_id
    platform_pose_provider.add_to_pose_time_series(t, pos, quat)


std_cams = {}
for camera_id in set([images[image_id].camera_id for image_id in images]):
    colmap_camera = cameras[image.camera_id]

    if colmap_camera.model == 'OPENCV':
        fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params

    K = K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([d1, d2, d3, d4])
    std_cams[image.camera_id] = StandardCamera(colmap_camera.width,
                                               colmap_camera.height, K, dist,
                                               [0, 0, 0], [0, 0, 0, 1],
                                               platform_pose_provider)


# Calculate reprojection error.
for image_id in images:
    image = images[image_id]
    colmap_camera = cameras[image.camera_id]
    fname = '%s/%s.txt' % (data_dir, os.path.splitext(image.name)[0])
    R = qvec2rotmat(image.qvec)

    ind = image.point3D_ids >= 0

    im_pts = image.xys[ind].T
    wrld_pts = np.array([points[i].xyz for i in image.point3D_ids[ind]]).T

    if colmap_camera.model == 'OPENCV':
        fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([d1, d2, d3, d4])
    tvec = image.tvec
    rvec = cv2.Rodrigues(R)[0]
    im_pts2 = np.squeeze(cv2.projectPoints(wrld_pts.T, rvec, tvec, K, dist)[0]).T
    err = np.sqrt(np.sum(im_pts2 - im_pts, axis=0))

    std_cams[image.camera_id].project(wrld_pts, t=image_id)


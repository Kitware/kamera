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
import json
from random import shuffle
import numpy as np
from scipy.optimize import minimize, fminbound
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2
import PIL

# Custom package imports.
from sensor_models import (
        quaternion_multiply,
        quaternion_from_matrix,
        quaternion_from_euler,
        quaternion_slerp,
        quaternion_inverse,
        quaternion_matrix
        )
from postflight_scripts import utilities
from sensor_models.nav_conversions import enu_to_llh
from sensor_models.nav_state import NavStateINSJson, NavStateFixed
from colmap_processing.camera_models import StandardCamera
from colmap_processing.colmap_interface import read_images_binary, Image, \
    read_points3D_binary, read_cameras_binary, qvec2rotmat, \
    standard_cameras_from_colmap
from colmap_processing.image_renderer import render_view


# ---------------------------- Define Paths ----------------------------------
# KAMERA flight directory where each sub-directory contains meta.json files.
flight_dir = '/host_filesystem/mnt/homenas2/kamera/Calibration/fl08'

# You should have a colmap directory where all of the Colmap-generated files
# reside.
colmap_dir = '/host_filesystem/mnt/data10tb/kamera_fl8/colmap'

# Directory with all of the raw images.
colmap_images_subdir = 'images0'

# Sub-directory containing the images.bin and cameras.bin. Set to '' if in the
# top-level Colmap directory.
sparse_recon_subdir = 'sparse'
aligned_sparse_recon_subdir = 'aligned'

# Location to save KAMERA camera models.
save_dir = '%s/kamera_models' % flight_dir
# ----------------------------------------------------------------------------


# Establish correspondence between real-world exposure times base of file
# names.
fname_to_time = {}
for json_fname in glob.glob('%s/**/*_meta.json' % flight_dir):
    try:
        with open(json_fname) as json_file:
            d = json.load(json_file)

            # Time that the image was taken.
            fname = os.path.split(json_fname)[1].replace('_meta.json', '')
            fname_to_time[fname] = d['evt']['time']
    except (OSError, IOError):
        pass

nav_state_provider = NavStateINSJson('%s/**/*_meta.json' % flight_dir)


# ----------------------------------------------------------------------------
# Assemble the list of filenames with paths relative to the 'images0' directory
# that we point Colmap to as the raw image directory. This may be a directory
# of images, or it might be a directory of subdirectories, each of which
# contains images from one camera.

# Read in the Colmap details of all images.
images_bin_fname = '%s/%s/images.bin' % (colmap_dir, sparse_recon_subdir)
colmap_images = read_images_binary(images_bin_fname)


def process_images(colmap_images):
    """

    Returns:
    :param img_fnames: Image filename associated with each of the images in
        'colmap_images'.
    :type img_fnames: list of str

    :param img_times: INS-reported time associated with the trigger of each
        image in 'colmap_images'.
    :type img_times:

    :param ins_poses: INS-reported pose, (x, y, z) position and (x, y, z, w)
        quaternion, associated with the trigger of time of each image in
        'colmap_images'.
    :type ins_poses:

    :param sfm_poses: Colmap-reported reported pose, (x, y, z) position and
        (x, y, z, w) quaternion, associated with the trigger time of each image
        in 'colmap_images'.
    :type sfm_poses:

    """
    img_fnames = []
    img_times = []
    ins_poses = []
    sfm_poses = []
    for image_num in colmap_images:
        image = colmap_images[image_num]
        base_name = '_'.join(os.path.split(image.name)[1].split('_')[:-1])
        try:
            t = fname_to_time[base_name]

            # Query the navigation state recorded by the INS for this time.
            pose = nav_state_provider.pose(t)

            # Query Colmaps pose for the camera.
            R = qvec2rotmat(image.qvec)
            pos = -np.dot(R.T, image.tvec)

            # The qvec used by Colmap is a (w, x, y, z) quaternion
            # representing the rotation of a vector defined in the world
            # coordinate system into the camera coordinate system. However,
            # the 'camera_models' module assumes (x, y, z, w) quaternions
            # representing a coordinate system rotation. Also, the quaternion
            # used by 'camera_models' represents a coordinate system rotation
            # versus the coordinate system transform of Colmap's convention,
            # so we need an inverse.

            #quat = transformations.quaternion_inverse(image.qvec)
            quat = image.qvec / np.linalg.norm(image.qvec)
            quat[0] = -quat[0]

            quat = [quat[1], quat[2], quat[3], quat[0]]

            sfm_pose = [pos, quat]

            img_times.append(t)
            ins_poses.append(pose)
            img_fnames.append(image.name)
            sfm_poses.append(sfm_pose)
        except KeyError:
            print('Couldn\'t find a _meta.json file associated with \'%s\'' %
                  base_name)

    ind = np.argsort(img_fnames)
    img_fnames = [img_fnames[i] for i in ind]
    img_times = [img_times[i] for i in ind]

    return img_fnames, img_times, ins_poses, sfm_poses


img_fnames, img_times, ins_poses, sfm_poses = process_images(colmap_images)
# ----------------------------------------------------------------------------


# We take the INS-reported position (converted from latitude, longitude, and
# altitude into easting/northing/up coordinates) and assign it to each image.
print('Latiude of ENU coordinate system:', nav_state_provider.lat0, 'degrees')
print('Longitude of ENU coordinate system:', nav_state_provider.lon0,
      'degrees')
print('Height above the WGS84 ellipsoid of the ENU coordinate system:',
      nav_state_provider.h0, 'meters')

# Colmap then uses this pairing to solve for a similarity transform to best-
# match the SfM poses it recovered into these positions. All Colmap coordinates
# in this aligned version of its reconstruction will then be in easting/
# northing/up meters coordinates
align_fname = '%s/image_locations.txt' % colmap_dir
with open(align_fname, 'w') as fo:
    for i in range(len(img_fnames)):
        name = img_fnames[i]
        pos = ins_poses[i][0]
        fo.write('%s %0.8f %0.8f %0.8f\n' % (name, pos[0], pos[1], pos[2]))


try:
    os.makedirs('%s/%s' % (colmap_dir, aligned_sparse_recon_subdir))
except (OSError, IOError):
    pass

print('Now run\nnoaa_kamera/src/kitware-ros-pkg/postflight_scripts/scripts/'
      'colmap/model_aligner.sh %s %s %s %s' % (colmap_dir.replace('/host_filesystem', ''),
                                               sparse_recon_subdir,
                                               'image_locations.txt',
                                               aligned_sparse_recon_subdir))

# ---------------------------------------------------------------------------
if False:
    # Sanity check, pick the coordinates for a point in the 3-D model and
    # convert them to latitude and longitude.
    enu = np.array((640.446167, 822.111633, -9.576390))
    print(enu_to_llh(enu[0], enu[1], enu[2], nav_state_provider.lat0,
                     nav_state_provider.lon0, nav_state_provider.h0))

# Read in the Colmap details of all images.
if True:
    images_bin_fname = '%s/%s/images.bin' % (colmap_dir,
                                             aligned_sparse_recon_subdir)
    colmap_images = read_images_binary(images_bin_fname)
    points_bin_fname = '%s/%s/points3D.bin' % (colmap_dir,
                                               aligned_sparse_recon_subdir)
    points3d = read_points3D_binary(points_bin_fname)
    camera_bin_fname = '%s/%s/cameras.bin' % (colmap_dir,
                                              aligned_sparse_recon_subdir)
    colmap_cameras = read_cameras_binary(camera_bin_fname)
else:
    # For sanity checking that the original unadjusted results line up and the
    # code itself is sound.
    images_bin_fname = '%s/%s/images.bin' % (colmap_dir, sparse_recon_subdir)
    colmap_images = read_images_binary(images_bin_fname)
    points_bin_fname = '%s/%s/points3D.bin' % (colmap_dir, sparse_recon_subdir)
    points3d = read_points3D_binary(points_bin_fname)
    camera_bin_fname = '%s/%s/cameras.bin' % (colmap_dir, sparse_recon_subdir)
    colmap_cameras = read_cameras_binary(camera_bin_fname)


if False:
    pts_3d = []
    for pt_id in points3d:
        pts_3d.append(points3d[pt_id].xyz)

    pts_3d = np.array(pts_3d).T
    plt.plot(pts_3d[0], pts_3d[1], 'ro')


# Load in all of the Colmap results into more-convenient structures.
points_per_image = {}
camera_from_camera_str = {}
for image_num in colmap_images:
    image = colmap_images[image_num]
    camera_str = image.name.split('/')[0]
    camera_from_camera_str[camera_str] = colmap_cameras[image.camera_id]

    xys = image.xys
    pt_ids = image.point3D_ids
    ind = pt_ids != -1
    pt_ids = pt_ids[ind]
    xys = xys[ind]
    xyzs = np.array([points3d[pt_id].xyz for pt_id in pt_ids])
    base_name = os.path.splitext(os.path.split(image.name)[1])[0]
    try:
        t = fname_to_time['_'.join(base_name.split('_')[:-1])]
        points_per_image[image.name] = (xys, xyzs, t)
    except KeyError:
        pass

img_fnames, img_times, ins_poses, sfm_poses = process_images(colmap_images)


if False:
    # Loop over all images and apply the camera model to project 3-D points
    # into the image and compare to the measured versions to calculate
    # reprojection error.
    err = []
    for i in range(len(img_fnames)):
        print('%i/%i' % (i + 1, len(img_fnames)))
        fname = img_fnames[i]
        sfm_pose = sfm_poses[i]
        camera_str = os.path.split(fname)[0]

        colmap_camera = camera_from_camera_str[camera_str]

        if colmap_camera.model == 'OPENCV':
                fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params
        elif colmap_camera.model == 'PINHOLE':
            fx, fy, cx, cy = colmap_camera.params
            d1 = d2 = d3 = d4 = 0

        K = K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([d1, d2, d3, d4])

        cm = StandardCamera(colmap_camera.width, colmap_camera.height, K, dist,
                            [0, 0, 0], [0, 0, 0, 1], None, frame_id=None,
                            nav_state_provider=NavStateFixed(*sfm_pose))
        xy, xyz, t = points_per_image[fname]
        err_ = np.sqrt(np.sum((xy - cm.project(xyz.T, t).T)**2, axis=1))
        err = err + err_.tolist()

    plt.hist(err, 1000)


camera_strs = set([os.path.split(fname)[0] for fname in img_fnames])


# ------------------------------ Calibrate RGB -------------------------------
for camera_str in camera_strs:
    if 'rgb' not in camera_str:
        continue

    ins_quat_ = []
    sfm_quat_ = []
    points_per_image_ = []
    for i in range(len(img_fnames)):
        fname = img_fnames[i]
        if os.path.split(fname)[0] == camera_str:
            ins_quat_.append(ins_poses[i][1])
            sfm_quat_.append(sfm_poses[i][1])
            try:
                points_per_image_.append(points_per_image[fname])
            except KeyError:
                pass

    # Both quaternions are of the form (x, y, z, w) and represent a coordinate
    # system rotation.
    #q_sfm = quaternion_inverse(q_cam)*quaternion_inverse(q_ins)
    cam_quats = [quaternion_inverse(quaternion_multiply(sfm_quat_[k],
                                                        ins_quat_[k]))
                 for k in range(len(ins_quat_))]

    colmap_camera = camera_from_camera_str[camera_str]

    if colmap_camera.model == 'OPENCV':
            fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params
    elif colmap_camera.model == 'PINHOLE':
        fx, fy, cx, cy = colmap_camera.params
        d1 = d2 = d3 = d4 = 0

    K = K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([d1, d2, d3, d4])

    def cam_quat_error(cam_quat):
        cam_quat = cam_quat/np.linalg.norm(cam_quat)
        camera_model = StandardCamera(colmap_camera.width,
                                      colmap_camera.height,
                                      K, dist, [0, 0, 0], cam_quat, None,
                                      frame_id=None,
                                      nav_state_provider=nav_state_provider)

        err = []
        for xys, xyzs, t in points_per_image_:
            if False:
                # Reprojection error.
                xys2 = camera_model.project(xyzs.T, t)
                err_ = np.sqrt(np.sum((xys2 - xys.T)**2, axis=0))
            else:
                # Error in meters.

                # Rays coming out of the camera in the direction of the imaged points.
                ray_pos, ray_dir = camera_model.unproject(xys.T, t)

                # Direction coming out of the camera pointing at the actual 3-D points'
                # locatinos.
                ray_dir2 = xyzs.T - ray_pos
                d = np.sqrt(np.sum((ray_dir2)**2, axis=0))
                ray_dir2 /= d

                dp = np.minimum(np.sum(ray_dir*ray_dir2, axis=0), 1)
                dp = np.maximum(dp, -1)
                theta = np.arccos(dp)
                err_ = np.sin(theta)*d
                #err.append(np.percentile(err_, 90))
                err.append(np.mean(err_))

        err = np.array(err)
        #err = err[err < np.percentile(err, 90)]

        err = np.mean(err)
        print('RMS reproject error for quat', cam_quat, ': %0.8f' % err)
        return err


    shuffle(cam_quats)
    best_quat = None
    best_err = np.inf
    for i in range(len(cam_quats)):
        if True:
            cam_quat = cam_quats[i]
        else:
            cam_quat = np.random.rand(4)*2-1

        err = cam_quat_error(cam_quat)

        if err < best_err:
            best_err = err
            best_quat = cam_quat

        if best_err < 10:
            break

    ret = minimize(cam_quat_error, best_quat)
    best_quat = ret.x/np.linalg.norm(ret.x)
    ret = minimize(cam_quat_error, best_quat, method='BFGS')
    best_quat = ret.x/np.linalg.norm(ret.x)
    ret = minimize(cam_quat_error, best_quat, method='Powell')
    best_quat = ret.x/np.linalg.norm(ret.x)

    # Sequential 1-D optimizations.
    for i in range(4):
        def set_x(x):
            quat = best_quat.copy()
            quat = quat/np.linalg.norm(quat)
            while abs(quat[i] - x) > 1e-6:
                quat[i] = x;
                quat = quat/np.linalg.norm(quat)

            return quat

        def func(x):
            return cam_quat_error(set_x(x))

        x = np.linspace(-1, 1, 100);   x = sorted(np.hstack([x, best_quat[i]]))
        y = [func(x_) for x_ in x]
        x = fminbound(func, x[np.argmin(y) - 1], x[np.argmin(y) + 1], xtol=1e-8)
        best_quat = set_x(x)

    camera_model = StandardCamera(colmap_camera.width, colmap_camera.height,
                                  K, dist, [0, 0, 0], best_quat, '',  '',
                                  nav_state_provider=nav_state_provider)

    try:
        os.makedirs(save_dir)
    except (KeyError, OSError):
        pass

    camera_model.save_to_file('%s/%s.yaml' % (save_dir, camera_str))

    # Final error analysis.
    err_meters = []
    err_pixels = []
    err_pixels_per_frame = []
    err_angle = []
    ifov = np.mean(camera_model.ifov())
    for xys, xyzs, t in points_per_image_:
        xys2 = camera_model.project(xyzs.T, t)
        err_pixels_ = np.sqrt(np.sum((xys2 - xys.T)**2, axis=0))
        err_pixels_per_frame.append([t, err_pixels_.mean()])
        err_pixels = err_pixels + err_pixels_.tolist()

        # Rays coming out of the camera in the direction of the imaged points.
        ray_pos, ray_dir = camera_model.unproject(xys.T, t)

        # Direction coming out of the camera pointing at the actual 3-D points'
        # locatinos.
        ray_dir2 = xyzs.T - ray_pos
        dist = np.sqrt(np.sum((ray_dir2)**2, axis=0))
        ray_dir2 /= dist

        dp = np.minimum(np.sum(ray_dir*ray_dir2, axis=0), 1)
        dp = np.maximum(dp, -1)
        theta = np.arccos(dp)
        err_angle = err_angle + theta.tolist()
        err_meters = err_meters + (np.sin(theta)*dist).tolist()

    err_meters = np.sort(err_meters)
    err_pixels = np.sort(err_pixels)
    err_angle = np.sort(err_angle)
    err_pixels_per_frame = np.array(err_pixels_per_frame).T

    # Save figures.
    plt.close('all')
    with PdfPages('%s/%s_error_analysis.pdf' % (save_dir, camera_str)) as pdf:
        fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 40})
        plt.rc('axes', linewidth=4)
        plt.semilogy(np.linspace(0, 100, len(err_meters)), err_meters)
        plt.xlabel('Percentile', fontsize=50)
        plt.ylabel('Error (meters)', fontsize=50)
        plt.yticks([0.01, 0.1, 1, 10, 100, 1000],
                   [0.01, 0.1, 1, 10, 100, 1000])
        fig.subplots_adjust(bottom=0.13)
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(right=0.96)
        fig.subplots_adjust(left=0.15)
        pdf.savefig()

        fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 40})
        plt.rc('axes', linewidth=4)
        plt.plot(np.linspace(0, 100, len(err_meters)), err_angle*180/np.pi)
        plt.xlabel('Percentile', fontsize=50)
        plt.ylabel('Error (degrees)', fontsize=50)
        plt.ylim([0, 10])
        fig.subplots_adjust(bottom=0.13)
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(right=0.96)
        fig.subplots_adjust(left=0.15)
        pdf.savefig()

        fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 40})
        plt.rc('axes', linewidth=4)
        plt.semilogy(np.linspace(0, 100, len(err_pixels)), err_pixels)
        plt.xlabel('Percentile', fontsize=50)
        plt.ylabel('Error (pixels)', fontsize=50)
        plt.yticks([0.01, 0.1, 1, 10, 100, 1000],
                   [0.01, 0.1, 1, 10, 100, 1000])
        fig.subplots_adjust(bottom=0.13)
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(right=0.96)
        fig.subplots_adjust(left=0.15)
        pdf.savefig()

        fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 40})
        plt.rc('axes', linewidth=4)
        plt.plot(err_pixels_per_frame[0], err_pixels_per_frame[1], 'ro')
        plt.xlabel('Time (seconds)', fontsize=50)
        plt.ylabel('Error (pixels)', fontsize=50)
        fig.subplots_adjust(bottom=0.13)
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(right=0.96)
        fig.subplots_adjust(left=0.15)
        pdf.savefig()


# ----------------------------- Calibrate UV ---------------------------------
images_bin_fname = '%s/%s/images.bin' % (colmap_dir, sparse_recon_subdir)
colmap_images = read_images_binary(images_bin_fname)
camera_bin_fname = '%s/%s/cameras.bin' % (colmap_dir, sparse_recon_subdir)
colmap_cameras = read_cameras_binary(camera_bin_fname)

camera_from_camera_str = {}
for image_num in colmap_images:
    image = colmap_images[image_num]
    camera_str = image.name.split('/')[0]
    camera_from_camera_str[camera_str] = colmap_cameras[image.camera_id]


nav_state_fixed = NavStateFixed(np.zeros(3), [0, 0, 0, 1])

for camera_str in camera_strs:
    if 'uv' not in camera_str:
        continue

    rgb_str = camera_str.replace('uv', 'rgb')
    cm_rgb = StandardCamera.load_from_file('%s/%s.yaml' % (save_dir, rgb_str),
                                           nav_state_provider=nav_state_fixed)

    im_pts = []
    im_pts_rgb = []

    # Build up pairs of image coordinates between the two cameras from image
    # pairs acquired from the same time.
    image_nums = sorted(list(colmap_images.keys()))
    for image_num in image_nums:
        print('%i/%i' % (image_num + 1, image_nums[-1]))
        image = colmap_images[image_num]
        if image.name.split('/')[0] != camera_str:
            continue

        base_name = os.path.splitext(os.path.split(image.name)[1])[0]

        try:
            t1 = fname_to_time['_'.join(base_name.split('_')[:-1])]
        except KeyError:
            continue

        # image is a Colmap image object from camera 'camera_str' and the image
        # was acquired at time t1.

        for image_num_rgb in colmap_images:
            image_rgb = colmap_images[image_num_rgb]
            if image_rgb.name.split('/')[0] != rgb_str:
                continue

            base_name2 = os.path.splitext(os.path.split(image_rgb.name)[1])[0]

            try:
                t2 = fname_to_time['_'.join(base_name2.split('_')[:-1])]
            except KeyError:
                continue

            if t1 == t2:
                # Both 'image' and 'image_rgb' are from the same time.
                pt_ids1 = image.point3D_ids
                ind = pt_ids1 != -1
                xys1 = dict(zip(pt_ids1[ind], image.xys[ind]))

                pt_ids2 = image_rgb.point3D_ids
                ind = pt_ids2 != -1
                xys2 = dict(zip(pt_ids2[ind], image_rgb.xys[ind]))

                match_ids = set(xys1.keys()).intersection(set(xys2.keys()))
                if len(match_ids) == 0:
                    continue

                for match_id in match_ids:
                    im_pts.append(xys1[match_id])
                    im_pts_rgb.append(xys2[match_id])

                # We don't expect any other matches.
                break

    im_pts = np.array(im_pts)
    im_pts_rgb = np.array(im_pts_rgb)

    if False:
        plt.subplot(121)
        plt.plot(im_pts[:, 0], im_pts[:, 1], 'ro')
        plt.subplot(122)
        plt.plot(im_pts_rgb[:, 0], im_pts_rgb[:, 1], 'bo')

    # Treat as co-located cameras (they are) and unproject out of RGB and into
    # the other camera.
    ray_pos, ray_dir = cm_rgb.unproject(im_pts_rgb.T)
    wrld_pts = ray_dir.T*1e8

    colmap_camera = camera_from_camera_str[camera_str]

    if colmap_camera.model == 'OPENCV':
        fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params
    elif colmap_camera.model == 'PINHOLE':
        fx, fy, cx, cy = colmap_camera.params
        d1 = d2 = d3 = d4 = 0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([d1, d2, d3, d4], dtype=np.float32)

    flags = cv2.CALIB_ZERO_TANGENT_DIST
    flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
    flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
    flags = flags | cv2.CALIB_FIX_K1
    flags = flags | cv2.CALIB_FIX_K2
    flags = flags | cv2.CALIB_FIX_K3
    flags = flags | cv2.CALIB_FIX_K4
    flags = flags | cv2.CALIB_FIX_K5
    flags = flags | cv2.CALIB_FIX_K6

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000,
                0.0000001)

    ret = cv2.calibrateCamera([wrld_pts.astype(np.float32)],
                              [im_pts.astype(np.float32)],
                              (colmap_camera.width, colmap_camera.height),
                              cameraMatrix=K.copy(), distCoeffs=dist.copy(),
                              flags=flags, criteria=criteria)

    err, _, _, rvecs, tvecs = ret

    R = np.identity(4)
    R[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
    cam_quat = quaternion_from_matrix(R.T)

    # Only optimize 3/4 components of the quaternion.
    static_quat_ind = np.argmax(np.abs(cam_quat))
    dynamic_quat_ind = list(range(4));
    dynamic_quat_ind.remove(static_quat_ind)
    dynamic_quat_ind = np.array(dynamic_quat_ind)
    x0 = cam_quat[dynamic_quat_ind]/cam_quat[static_quat_ind]

    def get_cm(x):
        cam_quat = np.ones(4)
        cam_quat[dynamic_quat_ind] = x[:3]
        cam_quat = cam_quat/np.linalg.norm(cam_quat)

        if len(x) > 3:
            fx_ = x[3]
            fy_ = x[4]
        else:
            fx_ = fx
            fy_ = fy

        if len(x) > 5:
            dist_ = x[5:]
        else:
            dist_ = dist

        K = np.array([[fx_, 0, cx], [0, fy_, cy], [0, 0, 1]])
        cm = StandardCamera(colmap_camera.width, colmap_camera.height,
                            K, dist_, [0, 0, 0], cam_quat, '',  '',
                            nav_state_provider=nav_state_fixed)
        return cm

    def error(x):
        cm = get_cm(x)

        err = np.sqrt(np.sum((im_pts - cm.project(wrld_pts.T).T)**2, 1))

        if False:
            err = err**2
        else:
            # Huber loss.
            delta = 20
            ind = err < delta
            err[ind] = err[ind]**2
            err[~ind] = 2*(err[~ind] - delta/2)*delta

        if False:
            x = np.linspace(0, 10)
            err = x**2
            plt.plot(x, err)
            delta = 5
            err = x.copy()
            ind = x < delta
            err[ind] = err[ind]**2
            err[~ind] = 2*(err[~ind] - delta/2)*delta
            plt.plot(x, err)

        #err = (1 - np.exp(-(err/a)**2))*b

        err = sorted(err)[:len(err) - len(err)//5]
        err = np.sqrt(np.mean(err))
        print(x, err)
        return err

    def plot_results1(x):
        cm = get_cm(x)
        err = np.sqrt(np.sum((im_pts - cm.project(wrld_pts.T).T)**2, 1))
        err = sorted(err)
        plt.plot(np.linspace(0, 100, len(err)), err)

    #x = np.hstack([cam_quat, fx, fy])
    x = x0.copy()
    ret = minimize(error, x, method='Powell');    x = ret.x
    x = np.hstack([ret.x, fx, fy])
    ret = minimize(error, x, method='Powell');    x = ret.x
    x = ret.x
    ret = minimize(error, x, method='BFGS');    x = ret.x

    if True:
        x = np.hstack([ret.x, dist])
        ret = minimize(error, x, method='Powell');    x = ret.x
        ret = minimize(error, x, method='BFGS');    x = ret.x

    cm = get_cm(x)

    cm.save_to_file('%s/%s.yaml' % (save_dir, camera_str))

    plt.close('all')
    with PdfPages('%s/%s_error_analysis.pdf' % (save_dir, camera_str)) as pdf:
        fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 40})
        plt.rc('axes', linewidth=4)
        plot_results1(x)
        plt.xlabel('Percentile', fontsize=50)
        plt.ylabel('Error (pixels)', fontsize=50)
        plt.title('Reprojection Error With RGB', fontsize=50)
        plt.ylim([0, 10])
        fig.subplots_adjust(bottom=0.13)
        fig.subplots_adjust(top=0.95)
        fig.subplots_adjust(right=0.96)
        fig.subplots_adjust(left=0.15)
        pdf.savefig()

        if False:
            fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
            plt.rc('font', **{'size': 40})
            plt.rc('axes', linewidth=4)
            plt.subplot(121)
            plt.plot(im_pts[:, 0], im_pts[:, 1], 'ro')
            plt.title('UV Points', fontsize=50)
            plt.subplot(122)
            plt.plot(im_pts_rgb[:, 0], im_pts_rgb[:, 1], 'bo')
            plt.title('RGB Points', fontsize=50)
            fig.subplots_adjust(bottom=0.13)
            fig.subplots_adjust(top=0.95)
            fig.subplots_adjust(right=0.96)
            fig.subplots_adjust(left=0.15)
            pdf.savefig()

    # Pick an image pair and register.
    gif_dir = '%s/registration_gifs' % save_dir

    try:
        os.makedirs(gif_dir)
    except (OSError, IOError):
        pass

    for k in range(10):
        inds = list(range(len(img_fnames)))
        shuffle(inds)

        for i in range(len(img_fnames)):
            fname1 = img_fnames[inds[i]]
            t1 = img_times[inds[i]]
            if os.path.split(fname1)[0] != rgb_str:
                continue

            for j in range(len(img_fnames)):
                fname2 = img_fnames[inds[j]]
                t2 = img_times[inds[j]]
                if os.path.split(fname2)[0] != camera_str or t1 != t2:
                    continue

                img2 = cv2.imread('%s/images0/%s' % (colmap_dir, fname2),
                                  cv2.IMREAD_COLOR)[:, :, ::-1]
                break

            img1 = cv2.imread('%s/images0/%s' % (colmap_dir, fname1),
                              cv2.IMREAD_COLOR)[:, :, ::-1]
            break


        img3, mask = render_view(cm_rgb, img1, 0, cm, 0, block_size=10)

        img2_ = PIL.Image.fromarray(cv2.pyrDown(cv2.pyrDown(img2)))
        img3_ = PIL.Image.fromarray(cv2.pyrDown(cv2.pyrDown(img3)))
        fname_out = '%s/%s_to_%s_registeration_%i.gif' % (gif_dir, rgb_str,
                                                          camera_str, k+1)
        img2_.save(fname_out, save_all=True, append_images=[img3_],
                   duration=250, loop=0)

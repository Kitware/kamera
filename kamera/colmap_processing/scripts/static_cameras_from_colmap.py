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
from mpl_toolkits.mplot3d import Axes3D
import glob
import natsort
import trimesh
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
save_dir = 'static_camera_models'

# Base path to the colmap directory containing 'cameras.bin' 'images.bin'.
image_dir = 'all'

# Path to the images.bin file.
images_bin_fname = '%s/images.bin' % image_dir
camera_bin_fname = '%s/cameras.bin' % image_dir
points_3d_bin_fname = '%s/points3D.bin' % image_dir

if True:
    mesh_fname = 'coarse.ply'

    latitude0 = 0      # degrees
    longitude0 = 0    # degrees
    altitude0 = 0   # meters above WGS84 ellipsoid
# ----------------------------------------------------------------------------


# Read in the details of all images.
images = read_images_binary(images_bin_fname)
cameras = read_cameras_binary(camera_bin_fname)
pts_3d = read_points3D_binary(points_3d_bin_fname)


if False:
    # Save off camera positions to be used to obtain geo-registration matrix.
    keys = list(images.keys())
    times = [float(os.path.splitext(images[key].name)[0])/1e6 for key in keys]
    times = np.array(times)
    inds = np.argsort(times)
    times = times[inds]
    keys = [keys[ind] for ind in inds]
    tvecs = [images[key].tvec for key in keys]
    rmat = [qvec2rotmat(images[key].qvec) for key in keys]
    cam_pos = [np.dot(rmat[i].T, -tvecs[i]) for i in range(len(tvecs))]
    cam_pos = np.array(cam_pos).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(cam_pos[0], cam_pos[1], cam_pos[2])
    rmat_ravel = np.array([r.ravel() for r in rmat]).T
    data = np.vstack([times, cam_pos, rmat_ravel]).T
    np.savetxt('cam_pos.txt', data)












def process(image_fname, ref_image_id, points_fname, save_dir, K,
            dist=[0, 0, 0, 0,], optimize_k1=False, optimize_k2=False,
            optimize_k3=False, optimize_k4=False):


    colmap_image = images[ref_image_id]
    camera = cameras[colmap_image.camera_id]

    point_pairs = np.loadtxt(points_fname)

    if False:
        point_pairs = point_pairs[3:]

    ref_pts = point_pairs[:, :2]
    im_pts = point_pairs[:, 2:].astype(np.float32)

    enu = [get_xyz_from_image_pt(ref_image_id, im_pt) for im_pt in ref_pts]
    enu = np.array(enu).astype(np.float32)

    calibrate_from_enu_pts(image_fname, im_pts, enu, save_dir, K, dist=dist,
                           optimize_k1=optimize_k1, optimize_k2=optimize_k2,
                           optimize_k3=optimize_k3, optimize_k4=optimize_k4)


def calibrate_from_llh_file(image_fname, points_fname, save_dir,
                            K, dist=[0, 0, 0, 0], optimize_k1=False,
                            optimize_k2=False, optimize_k3=False,
                            optimize_k4=False, fix_principal_point=True):
    ret = np.loadtxt(points_fname)
    im_pts = ret[:, :2]
    llh = ret[:, 2:]
    enu = [llh_to_enu(_[0], _[1], _[2], latitude0, longitude0, altitude0)
           for _ in llh]
    enu = np.array(enu)
    calibrate_from_enu_pts(image_fname, im_pts, enu, save_dir, K, dist=dist,
                           optimize_k1=optimize_k1, optimize_k2=optimize_k2,
                           optimize_k3=optimize_k3, optimize_k4=optimize_k4,
                           fix_principal_point=fix_principal_point)


def calibrate_manual_clicked_reference(image_fname, points_fname,
                                       ref_camera_fname, save_dir,
                                       dist=[0, 0, 0, 0], optimize_k1=False,
                                       optimize_k2=False, optimize_k3=False,
                                       optimize_k4=False,
                                       fix_principal_point=True):
    """points_fname from reference to image.

    """
    ret = np.loadtxt(points_fname)
    ref_pts = ret[:, :2]
    im_pts = ret[:, 2:]

    ret = load_static_camera_from_file(ref_camera_fname)
    K_ref, dist_ref, R, depth_map, latitude, longitude, altitude = ret[2:]
    height, width = depth_map.shape

    enu0 = llh_to_enu(latitude, longitude, altitude, latitude0, longitude0,
                      altitude0)
    enu0 = np.array(enu0)
    enu = np.zeros((len(ref_pts), 3))

    # Unproject rays into the camera coordinate system.
    ray_dir = np.ones((3, len(ref_pts)), dtype=np.float)
    ray_dir0 = cv2.undistortPoints(np.expand_dims(ref_pts, 0),
                                   K_ref, dist_ref, R=None)
    ray_dir[:2] = np.squeeze(ray_dir0, 0).T

    # Rotate rays into the local east/north/up coordinate system.
    ray_dir = np.dot(R.T, ray_dir)
    ray_dir /= np.sqrt(np.sum(ray_dir**2, 0))

    for i in range(ref_pts.shape[0]):
        x, y = ref_pts[i]
        if x == 0:
            ix = 0
        elif x == width:
            ix = int(width - 1)
        else:
            ix = int(round(x - 0.5))

        if y == 0:
            iy = 0
        elif y == height:
            iy = int(height - 1)
        else:
            iy = int(round(y - 0.5))

        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            print(x == width)
            print(y == height)
            raise ValueError('Coordinates (%0.1f,%0.f) are outside the '
                             '%ix%i image' % (x, y, width, height))

        enu[i] = enu0 + ray_dir[:, i]*depth_map[iy, ix]

    calibrate_from_enu_pts(image_fname, im_pts, enu, save_dir, K_ref,
                           dist=dist, optimize_k1=optimize_k1,
                           optimize_k2=optimize_k2, optimize_k3=optimize_k3,
                           optimize_k4=optimize_k4,
                           fix_principal_point=fix_principal_point)


def calibrate_from_enu_pts(image_fname, im_pts, enu, save_dir, K,
                           dist=[0, 0, 0, 0], optimize_k1=False,
                           optimize_k2=False, optimize_k3=False,
                           optimize_k4=False, fix_principal_point=True):
    dist = np.array(dist, dtype=np.float32)
    real_image = cv2.imread(image_fname)

    if real_image.ndim == 3:
        real_image = real_image[:, :, ::-1]

    height, width = real_image.shape[:2]

    K[0, 2] = width/2
    K[1, 2] = height/2

    if False:
        ref_image = cv2.imread(ref_image_fname)[:, :, ::-1]
        plt.figure(); plt.imshow(ref_image)
        plt.plot(ref_pts.T[0], ref_pts.T[1], 'bo')
        plt.figure(); plt.imshow(real_image)
        plt.plot(im_pts.T[0], im_pts.T[1], 'bo')

    if False:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        fig.add_subplot(111, projection='3d')
        plt.plot(enu[:,0], enu[:,1], enu[:,2], 'ro')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')


    def show_err(rvec, tvec, k, dist, plot=False):
        im_pts_ = cv2.projectPoints(enu, rvec, tvec, k, dist)[0]
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


    if False:
        # First pass.
        flags = cv2.CALIB_ZERO_TANGENT_DIST
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
        flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
        flags = flags | cv2.CALIB_FIX_ASPECT_RATIO
        flags = flags | cv2.CALIB_FIX_K1
        flags = flags | cv2.CALIB_FIX_K2
        flags = flags | cv2.CALIB_FIX_K3
        flags = flags | cv2.CALIB_FIX_K4
        flags = flags | cv2.CALIB_FIX_K5
        flags = flags | cv2.CALIB_FIX_K6
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000,
                    0.0000001)
        ret = cv2.calibrateCamera([enu.astype(np.float32)],
                                   [im_pts.astype(np.float32)], (width, height),
                                   cameraMatrix=K, distCoeffs=dist, flags=flags,
                                   criteria=criteria)
        err, K, dist, rvecs, tvecs = ret
        print('First pass error:', err)
        rvec, tvec = rvecs[0], tvecs[0]

        # Second pass.
        flags = cv2.CALIB_ZERO_TANGENT_DIST
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
        flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
        flags = flags | cv2.CALIB_FIX_K1
        flags = flags | cv2.CALIB_FIX_K2
        flags = flags | cv2.CALIB_FIX_K3
        flags = flags | cv2.CALIB_FIX_K4
        flags = flags | cv2.CALIB_FIX_K5
        flags = flags | cv2.CALIB_FIX_K6

        ret = cv2.calibrateCamera([enu.astype(np.float32)],
                                   [im_pts.astype(np.float32)],
                                   (width, height), cameraMatrix=K,
                                   distCoeffs=dist, flags=flags)
        err, K, dist, rvecs, tvecs = ret
        print('Second pass error:', err)
        rvec, tvec = rvecs[0], tvecs[0]

    #if optimize_k1 or optimize_k2 or optimize_k3 or optimize_k4:
    if True:
        # Third pass.
        flags = cv2.CALIB_ZERO_TANGENT_DIST
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS

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

        ret = cv2.calibrateCamera([enu.astype(np.float32)],
                                   [im_pts.astype(np.float32)],
                                   (width, height), cameraMatrix=K,
                                   distCoeffs=dist, flags=flags)
        err, K, dist, rvecs, tvecs = ret
        print('Third pass error:', err)
        rvec, tvec = rvecs[0], tvecs[0]


    show_err(rvec, tvec, K, dist, plot=True)

    R = cv2.Rodrigues(rvec)[0]
    cam_pos = -np.dot(R.T, tvec).ravel()

    save_camera_model(width, height, K, dist, cam_pos, R, real_image, save_dir,
                      mode=2)
    # ------------------------------------------------------------------------


def get_features(image, num_features=10000):
    # Find the keypoints and descriptors and match them.
    orb = cv2.ORB_create(nfeatures=num_features, edgeThreshold=25,
                         patchSize=31, nlevels=16,
                         scoreType=cv2.ORB_FAST_SCORE, fastThreshold=10)

    if image.ndim == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image

    img_gray = img_gray.astype(np.float)
    img_gray -= np.percentile(img_gray.ravel(), 1)
    img_gray[img_gray < 0] = 0
    img_gray /= np.percentile(img_gray.ravel(), 99)/255
    img_gray[img_gray > 225] = 255
    img_gray = np.round(img_gray).astype(np.uint8)

    return orb.detectAndCompute(img_gray, None)
    # ------------------------------------------------------------------------


def process_auto(image_fname, ref_camera_fname, save_dir, optimize_k1=False,
                 optimize_k2=False, optimize_k3=False, optimize_k4=False,
                 fix_principal_point=True, num_features=10000, homog_thresh=20,
                 final_thresh=5):
    ret = load_static_camera_from_file(ref_camera_fname)
    K, dist, R, depth_map, latitude, longitude, altitude = ret[2:]

    real_image = cv2.imread(image_fname)

    height, width = real_image.shape[:2]

    if real_image.ndim == 3:
        real_image = real_image[:, :, ::-1]

    ref_image = cv2.imread('%s/ref_view.png' %
                           os.path.split(ref_camera_fname)[0])

    if ref_image.ndim == 3:
        ref_image = ref_image[:, :, ::-1]

    kp0, des0 = get_features(ref_image, num_features=num_features)
    kp1, des1 = get_features(real_image, num_featuresref_camera_fname=num_features)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des0, des1)

    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

    print('Feature matcher found %i matches' % len(matches))

    if False:
        mask = np.logical_and(pts0[:, 1] > 355,
                              pts0[:, 1] < 649)
        pts0 = pts0[mask]
        pts1 = pts1[mask]
        mask = np.logical_and(pts1[:, 1] > 355,
                              pts1[:, 1] < 648)
        pts0 = pts0[mask]
        pts1 = pts1[mask]

    M, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, homog_thresh)
    mask = np.squeeze(mask) > 0

    pts0 = pts0[mask]
    pts1 = pts1[mask]

    print('Homography filtering yielded %i matches' % len(pts0))

    ray_dir = cv2.undistortPoints(np.expand_dims(pts0, 0), K, dist, None)
    ray_dir = np.squeeze(ray_dir, 0).astype(np.float32).T
    ray_dir = np.vstack([ray_dir, np.ones(ray_dir.shape[1])])

    ray_dir = np.dot(R.T, ray_dir)

    d = []
    for pt in pts0:
        ix, iy = np.round(pt - 0.5).astype(np.int)
        d.append(depth_map[iy, ix])

    ray_dir = ray_dir*d

    cam_pos = llh_to_enu(latitude, longitude, altitude, latitude0, longitude0,
                         altitude0)
    enu = ray_dir.T + np.atleast_2d(cam_pos)

    if False:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        fig.add_subplot(111, projection='3d')
        plt.plot(enu[:,0], enu[:,1], enu[:,2], 'ro')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')

    # First pass.
    K[0, 2] = width/2
    K[1, 2] = height/2
    flags = cv2.CALIB_ZERO_TANGENT_DIST
    flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
    flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
    flags = flags | cv2.CALIB_FIX_K1
    flags = flags | cv2.CALIB_FIX_K2
    flags = flags | cv2.CALIB_FIX_K3
    flags = flags | cv2.CALIB_FIX_K4
    flags = flags | cv2.CALIB_FIX_K5
    flags = flags | cv2.CALIB_FIX_K6

    while True:
        ret = cv2.calibrateCamera([enu.astype(np.float32)],
                                   [pts1.astype(np.float32)],
                                   (width, height), cameraMatrix=K,
                                   distCoeffs=dist, flags=flags)
        err, K, dist, rvecs, tvecs = ret
        print('Focal lengths', K[0, 0], K[1, 1])
        print('Second pass error:', err)
        rvec, tvec = rvecs[0], tvecs[0]

        im_pts_ = cv2.projectPoints(enu, rvec, tvec, K, dist)[0]
        im_pts_ = np.squeeze(im_pts_)
        err = np.sqrt(np.sum((pts1 - im_pts_)**2, 1))

        mask = err < max(100, np.percentile(err, 50))

        if np.all(mask):
            break

        enu = enu[mask]
        pts0 = pts0[mask]
        pts1 = pts1[mask]

        print('Calibration filtering yielded %i matches' % len(pts0))

    # Second pass.
    flags = cv2.CALIB_ZERO_TANGENT_DIST
    flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS

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

    while True:
        ret = cv2.calibrateCamera([enu.astype(np.float32)],
                                   [pts1.astype(np.float32)],
                                   (width, height), cameraMatrix=K,
                                   distCoeffs=dist, flags=flags)
        err, K, dist, rvecs, tvecs = ret
        print('Second pass error:', err)
        rvec, tvec = rvecs[0], tvecs[0]

        im_pts_ = cv2.projectPoints(enu, rvec, tvec, K, dist)[0]
        im_pts_ = np.squeeze(im_pts_)
        err = np.sqrt(np.sum((pts1 - im_pts_)**2, 1))

        mask = err < final_thresh

        if np.all(mask):
            break

        enu = enu[mask]
        pts0 = pts0[mask]
        pts1 = pts1[mask]

    R = cv2.Rodrigues(rvec)[0]
    cam_pos = -np.dot(R.T, tvec).ravel()

    if True:
        plt.figure()
        plt.subplot('211')
        plt.imshow(ref_image, cmap='gray', interpolation='none')
        plt.plot(pts0.T[0], pts0.T[1], 'ro')
        plt.subplot('212')
        plt.imshow(real_image, cmap='gray', interpolation='none')
        plt.plot(pts1.T[0], pts1.T[1], 'bo')

    if False:
        save_camera_model(monitor_resolution[0], monitor_resolution[1], K,
                          np.zeros(4), cam_pos, R, real_image, save_dir,
                          mode=1)

    save_camera_model(width, height, K, dist, cam_pos, R, real_image, save_dir,
                      mode=1)


def save_camera_model(width, height, K, dist, cam_pos, R, real_image,
                      save_dir, mode=1, scale=0.5):
    ret = vtk_util.render_distored_image(width, height, K, dist, cam_pos, R,
                                         model_reader, return_depth=True,
                                         monitor_resolution=(1920, 1080))
    img, depth, E, N, U = ret

    # ------------------------------------------------------------------------

    try:
        os.makedirs(save_dir)
    except OSError:
        pass

    cv2.imwrite('%s/ref_view.png' % save_dir, real_image[:, :, ::-1])

    # Read and undistort image.
    cv2.imwrite('%s/rendered_view.png' % save_dir, img[:, :, ::-1])
    # -------------------------------------------------------------------

    latitude, longitude, altitude = enu_to_llh(cam_pos[0], cam_pos[1],
                                               cam_pos[2], latitude0,
                                               longitude0, altitude0)

    # R currently is relative to ENU coordinate system at latitude0,
    # longitude0, altitude0, but we need it to be relative to latitude,
    # longitude, altitude.
    Rp = np.dot(rmat_enu_ecef(latitude0, longitude0),
                rmat_ecef_enu(latitude, longitude))

    filename = '%s/camera_model.yaml' % save_dir
    R2 = np.dot(R, Rp)
    save_static_camera(filename, height, width, K, dist, R2,
                       depth, latitude, longitude, altitude)

    filename = '%s/camera_model.krtd' % save_dir
    tvec = -np.dot(R, cam_pos).ravel()
    write_camera_krtd_file([K, R2, tvec, dist], filename)

    # Save (x, y, z) meters per pixel.
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


cam_name = 'test'
save_dir_ = '%s/%s' % (save_dir, cam_name)
points_fname = 'points.txt'
image_fname = '%s.jpg' % cam_name
ref_image_fname = '%s.jpg' % ref_image_id
process(image_fname, ref_image_id, points_fname, save_dir_, K=K_reolink,
        dist=None, optimize_k1=False, optimize_k2=False, optimize_k3=False,
        optimize_k4=False)
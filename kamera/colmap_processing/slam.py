
#!/usr/bin/env python
"""
ckwg +31
Copyright 2020 by Kitware, Inc.
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

Note: the image coordiante system has its origin at the center of the top left
pixel.

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import cv2
import copy
import os
import time
from random import shuffle
import bisect
from math import sqrt
import gtsam
from gtsam import (Cal3_S2, DoglegOptimizer, GenericProjectionFactorCal3_S2,
                   Marginals, NonlinearFactorGraph, PinholeCameraCal3_S2,
                   Point3, Pose3, PriorFactorPoint3, PriorFactorPose3, Rot3,
                   Values)
from gtsam import symbol_shorthand
L = symbol_shorthand.L
X = symbol_shorthand.X
Y = symbol_shorthand.Y
V = symbol_shorthand.V
W = symbol_shorthand.W
BIAS_KEY = 1234567
DUMMY_BIAS_KEY = 12345678

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Repository imports.
from colmap_processing.platform_pose import PlatformPoseInterp
from colmap_processing.calibration import horn
from colmap_processing.rotations import quaternion_from_matrix, \
    quaternion_matrix
from colmap_processing.colmap_interface import read_images_binary, \
    read_points3D_binary, read_cameras_binary, standard_cameras_from_colmap
from colmap_processing.colmap_interface import Image as ColmapImage


def fit_pinhole_camera(cm):
    """Solve for a best-matching pinhole (distortion-free) camera model.

    For SLAM problems where we already have an accurate model for the camera's
    intrinsic parameters, we only want to optimize camera pose and world
    geometry at runtime. Therefore, it is convenient to warp image coordinates
    within the original image to appear as if they came from a pinhole
    (distrortion-free) camera. Since we might do reprojection error
    calculations, we want to maintain the size of a 1x1 pixel projected into
    the new pinhole camera to be as closer to 1x1.

    Parameters
    ----------
    cm : camera_models.StandardCamera
        Camera model with distortion.

    Returns
    -------
    cm_pinhole : camera_models.StandardCamera
        Camera model without distortion.
    """
    cm_pinhole = copy.deepcopy(cm)
    cm_pinhole.dist = 0
    im_pts = cm.points_along_image_border(1000)

    ray_pos, ray_dir = cm.unproject(im_pts, 0)
    im_pts2 = cm_pinhole.project(ray_pos + ray_dir, 0)

    #plt.plot(im_pts[0],im_pts[1]); plt.plot(im_pts2[0],im_pts2[1])

    dx1 = - min([min(im_pts2[0]), 0])
    dy1 = - min([min(im_pts2[1]), 0])
    dx2 =  max([max(im_pts2[0]) - cm.width, 0])
    dy2 = max([max(im_pts2[1]) - cm.height, 0])
    cm_pinhole.cx = cm_pinhole.cx + dx1
    cm_pinhole.cy = cm_pinhole.cy + dy1
    cm_pinhole.width = cm_pinhole.width + int(np.ceil(dx1 + dx2))
    cm_pinhole.height = cm_pinhole.height + int(np.ceil(dy1 + dy2))

    return cm_pinhole


def draw_keypoints(image, pts, radius=2, color=(255, 0, 0),
                   thickness=1, copy=False):
    """
    :param image: Image/
    :type image: Numpy image

    :param pts: Keypoints to be drawn in the image.
    :type pts: Numpy array num_pts x 2
    """
    pts = np.round(pts).astype(int)

    if len(color) == 3 and image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif copy:
        image = image.copy()

    for pt in pts:
        cv2.circle(image, (pt[0], pt[1]), radius, color, thickness)

    return image


def map_to_pinhole_problem(cm, im_pts_at_time):
    """Map Colmap reconstruction to appear as if it were a pinhole camera.

    GTSAM can support a camera model including distortion. However, with the
    camera model being known ahead of time, it is more computationally
    efficient to pre-undistort the image points as if they came from a
    particular pinhole camera. This function determines and returns the
    appropriate pinhole camera model (i.e., distortion coefficients are zero),
    and then it undistorts and returns the image feature points. When used
    within a SfM pose optimization problem, this would yield the same poses as
    the original distorted camera/points, only faster.

    sfm_cm : camera_models.StandardCamera
        Camera model potentially with non-zero distortion coefficients.

    im_pts_at_time : dict
        Dictionary taking image time and returning the tuple
        (im_pts, point3D_ind), where 'im_pts' is a 2 x N array of image
        coordinates and that are associated with wrld_pts[:, point3D_ind].

    Returns
    -------
    sfm_cm : camera_models.StandardCamera
        Camera model potentially with zero distortion coefficients.

    im_pts_at_time : dict
        Dictionary taking image time and returning the tuple
        (im_pts, point3D_ind), where 'im_pts' is a 2 x N array of image
        coordinates and that are associated with wrld_pts[:, point3D_ind].

    """
    cm_pinhole = fit_pinhole_camera(cm)

    for t in im_pts_at_time:
        im_pts, point3D_ind = im_pts_at_time[t]
        ray_pos, ray_dir = cm.unproject(im_pts, t)
        im_pts2 = cm_pinhole.project(ray_pos + ray_dir*100, t)
        im_pts_at_time[t] = im_pts2, point3D_ind

    return cm_pinhole, im_pts_at_time


def reprojection_error(cm, im_pts_at_time, wrld_pts, wrld_pts_to_score=None,
                       plot_results=False):
    """Calculate mean reprojection error in pixels.

    wrld_pts_to_score : None | bool array-like
        Boolean array of length equal to the number of world points (columns of
        wrld_pts) indicating which world points to consider when calculating
        the error. If None, all will be scored.
    """
    err = []
    image_times = sorted(list(im_pts_at_time.keys()))
    N2 = 0
    N4 = 0
    N10 = 0
    N20 = 0
    N = 0
    point_dist = []
    for t in image_times:
        ret = im_pts_at_time[t]
        if ret is None:
            continue

        im_pts, point3D_ind = ret

        if im_pts.shape[1] == 0:
            continue

        if wrld_pts_to_score is not None:
            ind = wrld_pts_to_score[point3D_ind]
            im_pts = im_pts[:, ind]
            point3D_ind = point3D_ind[ind]
            if len(point3D_ind) == 0:
                continue

        err_ = np.sqrt(np.sum((im_pts - cm.project(wrld_pts[:, point3D_ind], t))**2, axis=0))
        N2 += sum(err_ < 2)
        N4 += sum(err_ < 4)
        N10 += sum(err_ < 10)
        N20 += sum(err_ < 20)
        N += len(err_)
        err.append(err_)

        ray_pos, ray_dir = cm.unproject(im_pts, t, normalize_ray_dir=True)
        point_dir = wrld_pts[:, point3D_ind] - ray_pos
        d = np.sum(ray_dir*point_dir, axis=0)
        point_dist.append(d)

    point_dist = np.hstack(point_dist)
    err = np.hstack(err)

    print('Error %0.1f%% < 2 pixels, %0.1f%% < 4 pixels, %0.1f%% < 10 pixels, '
          '%0.1f%% < 20 pixels' % (100*N2/N, 100*N4/N, 100*N10/N, 100*N20/N))

    print('Point distance from camera %0.1f (0%%), %0.1f (25%%), '
          '%0.1f (50%%), %0.1f (75%%), %0.1f (100%%)' %
          (np.percentile(point_dist, 0),
           np.percentile(point_dist, 25),
           np.percentile(point_dist, 50),
           np.percentile(point_dist, 75),
           np.percentile(point_dist, 100)))

    if plot_results:
        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        plt.plot(np.sort(err), '.')
        plt.xlabel('Percentile', fontsize=40)
        plt.ylabel('Image Mean Error (pixels)', fontsize=40)

    return np.mean(err)


def show_solution_errors(cm, cm_ins, im_pts_at_time, wrld_pts):
    """Calculate mean reprojection error in pixels.
    """
    image_times = sorted(list(im_pts_at_time.keys()))
    err2 = []
    err10 = []
    err20 = []
    err50 = []
    err70 = []
    gps_err = []
    for t in image_times:
        im_pts, point3D_ind = im_pts_at_time[t]
        err_ = np.sqrt(np.sum((im_pts - cm.project(wrld_pts[:, point3D_ind], t))**2, axis=0))
        err2.append(np.percentile(err_, 2))
        err10.append(np.percentile(err_, 10))
        err20.append(np.percentile(err_, 20))
        err50.append(np.percentile(err_, 50))
        err70.append(np.percentile(err_, 70))

        pos0, quat0 = cm.platform_pose_provider.pose(t)
        pos1 = cm_ins.platform_pose_provider.pos(t)
        gps_err.append(pos0 - pos1)

    gps_err = np.array(gps_err).T

    plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
    plt.rc('font', **{'size': 20})
    plt.rc('axes', linewidth=4)
    plt.subplot(2, 1, 1)
    plt.semilogy(err2, '.', label='2 percentile')
    plt.semilogy(err10, '.', label='10 percentile')
    plt.semilogy(err20, '.', label='20 percentile')
    plt.semilogy(err50, '.', label='50 percentile')
    plt.semilogy(err70, '.', label='70 percentile')
    plt.xlabel('Image', fontsize=30)
    plt.legend(fontsize=14)
    plt.ylabel('Reprojection Error (pixels)', fontsize=30)
    plt.subplot(2, 1, 2)
    plt.plot(gps_err[0], label='X')
    plt.plot(gps_err[1], label='Y')
    plt.plot(gps_err[2], label='Z')
    plt.xlabel('Image', fontsize=30)
    plt.legend(fontsize=14)
    plt.ylabel('Distance From\nINS Solution (m)', fontsize=30)
    plt.tight_layout()


def show_reproj_error_on_images(cm, im_pts_at_time, wrld_pts, image_names,
                                image_dir, out_dir, wrld_pts_to_show=None,
                                radius=6, thickness=2):
    """Show reprojection error on images.

    wrld_pts_to_show : None | bool array-like
        Boolean array of length equal to the number of world points (columns of
        wrld_pts) indicating which world points to consider when calculating
        the error. If None, all will be shown.
    """
    image_times = sorted(list(im_pts_at_time.keys()))
    for t in image_times:
        ret = im_pts_at_time[t]
        if ret is None:
            continue

        im_pts, point3D_ind = ret

        if wrld_pts_to_show is not None:
            ind = wrld_pts_to_show[point3D_ind]
            im_pts = im_pts[:, ind]
            point3D_ind = point3D_ind[ind]

        if len(point3D_ind) == 0:
            continue

        im_pts2 = cm.project(wrld_pts[:, point3D_ind], t)

        img = cv2.imread('%s/%s' % (image_dir, image_names[t]))[:, :, ::-1].copy()

        for i in range(im_pts.shape[1]):
            pt1 = np.round(im_pts[:, i]).astype(int)
            pt2 = np.round(im_pts2[:, i]).astype(int)
            cv2.circle(img, (pt1[0], pt1[1]), radius, (255, 0, 0), thickness)
            cv2.circle(img, (pt2[0], pt2[1]), radius, (0, 0, 255), thickness)
            cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0),
                     thickness)

        base = os.path.splitext(os.path.split(image_names[t])[1])[0]
        cv2.imwrite('%s/%s.jpg' % (out_dir, base), img[:, :, ::-1])


def rescale_sfm_to_ins(cm, ins, wrld_pts):
    """Rescale the SfM result via similarity transform to match INS outputs.

    The structure from motion (SfM) solution is self consistent but only
    defined up to an arbitrary similarity transform. We are going to optimize
    the results of the SfM to best match the INS ouputs. But, a good initial
    alignment can be done by solving for the similarity transform that warps
    the SfM results as close as possible to the INS results.

    Parameters
    ----------
    cm : camera_models.StandardCamera
        Structure from motion camera model.

    ins : platform_pose.PlatformPose
        Representation of the INS-reported pose as function of time.

    wrld_pts : 3 x N array
        3-D coordinates of world points that are correlated with image points.

    Returns
    -------
    cm_out : camera_models.StandardCamera
        Structure from motion camera model.

    wrld_pts_out : 3 x N array
        3-D coordinates of world points that are correlated with image points.

    """
    ppp = cm.platform_pose_provider

    # Rescale using the ins.
    pts1 = []
    pts2 = []
    for i, t  in enumerate(ppp.times):
        P = cm.get_camera_pose(t)
        R = P[:, :3]
        pts1.append(np.dot(-R.T, P[:, 3]))
        pts2.append(ins.pose(t)[0])

    pts1 = np.array(pts1).T
    pts2 = np.array(pts2).T
    s, R, trans = horn(pts1, pts2, fit_scale=True, fit_translation=True)

    wrld_pts_out = np.dot(R, s*wrld_pts) + np.atleast_2d(trans).T

    cm_out = copy.deepcopy(cm)
    ppp2 = PlatformPoseInterp(ppp.lat0, ppp.lon0, ppp.h0)
    for i in range(len(ppp.times)):
        t = ppp.times[i]
        pos, quat = ppp.pose(t)
        pos = np.dot(R, s*pos) + trans
        R0 = quaternion_matrix(quat)[:3, :3]
        R1 = np.dot(R, R0)
        quat = quaternion_from_matrix(R1)
        ppp2.add_to_pose_time_series(t, pos, quat)

    cm_out.platform_pose_provider = ppp2

    return cm_out, wrld_pts_out


def read_colmap_results(recon_dir, use_camera_id=None, max_images=None,
                        max_image_pts=None, min_track_len=None):
    """Read colmap bin data and return camera and image with 3d point pairs.

    This function assumes that the image filenames encode the integer number of
    microseconds since the Unix epoch.

    Parameters
    ----------
    recon_dir : str
        Direction in which to find colmap 'images.bin', 'cameras.bin', and
        'points3D.bin' files.

    use_camera_id : int | None
        If Colmap didn't use a single camera for all images, we can force to
        use one camera model for all images. This sets the index of the desired
        camera to use.

    :max_images: int | None
        Sets maximum number of images to consider. Reducing this value allows
        experimentation with smaller problems that are quicker to process.

    :max_image_pts: int | None
        Sets the maximum number of image feature points to consider per image.
        If set, each image that exceeds this number will have a randsom subset
        of this size returned.

    :min_track_len: int
        Only consider features that were tracked for this minimum number of
        images.

    Returns
    -------
    sfm_cm : camera_models.StandardCamera
        Camera model that accepts

    im_pts_at_time : dict
        Dictionary taking image time and returning the tuple
        (im_pts, point3D_ind), where 'im_pts' is a 2 x N array of image
        coordinates and that are associated with wrld_pts[:, point3D_ind].

    wrld_pts : 3 x N array
        3-D coordinates of world points that are correlated with image points.

    image_names : dict
        Dictionary taking image time (s) and returning image name.


    """
    # ------------------- Read Existing Colmap Reconstruction ----------------
    # Read in the Colmap details of all images.
    images_bin_fname = '%s/images.bin' % recon_dir
    colmap_images = read_images_binary(images_bin_fname)
    camera_bin_fname = '%s/cameras.bin' % recon_dir
    colmap_cameras = read_cameras_binary(camera_bin_fname)
    points_bin_fname = '%s/points3D.bin' % recon_dir
    points3d = read_points3D_binary(points_bin_fname)

    if max_images is not None:
        colmap_images0 = colmap_images
        colmap_images = {}
        for t in sorted(list(colmap_images0.keys()))[:max_images]:
            colmap_images[t] = colmap_images0[t]

    if use_camera_id is None:
        keys = list(colmap_cameras.keys())
        if len(keys) > 1:
            raise Exception('Found multiply camera models %s, need to pick '
                           'which one to use for all images' % str(keys))

        use_camera_id = keys[0]

    image_times = {}
    for ind in colmap_images:
        colmap_images[ind] = ColmapImage(colmap_images[ind].id,
                                         colmap_images[ind].qvec,
                                         colmap_images[ind].tvec,
                                         use_camera_id,
                                         colmap_images[ind].name,
                                         colmap_images[ind].xys,
                                         colmap_images[ind].point3D_ids)
        img_fname = os.path.split(colmap_images[ind].name)[1]
        image_times[ind] = float(os.path.splitext(img_fname)[0])/1000000

    sfm_cm = standard_cameras_from_colmap(colmap_cameras, colmap_images,
                                          image_times)[use_camera_id]

    wrld_pts = [points3d[i].xyz if i in points3d else None
                for i in range(max(points3d.keys()) + 1)]

    track_len = np.zeros(len(wrld_pts), dtype=int)
    for image_num in colmap_images:
        image = colmap_images[image_num]
        point3D_ind = image.point3D_ids
        track_len[point3D_ind] += 1

    used_3d = np.zeros(len(wrld_pts), dtype=bool)
    for image_num in colmap_images:
        image = colmap_images[image_num]
        point3D_ind = image.point3D_ids
        point3D_ind = image.point3D_ids
        ind = point3D_ind != -1
        point3D_ind = point3D_ind[ind]

        if min_track_len is not None:
            ind = track_len[point3D_ind] >= min_track_len
            point3D_ind = point3D_ind[ind]

        if max_image_pts is not None and len(point3D_ind) > max_image_pts:
            ind = np.argsort([track_len[i] for i in point3D_ind])[::-1][:max_image_pts]
            point3D_ind = point3D_ind[ind]

        used_3d[point3D_ind] = True

    im_pts_at_time = {}
    image_names = {}
    for image_num in colmap_images:
        image = colmap_images[image_num]
        xys = image.xys
        point3D_ind = image.point3D_ids
        ind = point3D_ind != -1
        point3D_ind = point3D_ind[ind]
        xys = xys[ind]
        ind = used_3d[point3D_ind]
        point3D_ind = point3D_ind[ind]
        xys = xys[ind]

        img_fname = os.path.split(image.name)[1]
        t = float(os.path.splitext(img_fname)[0])/1000000
        im_pts_at_time[t] = (xys.T, point3D_ind)
        image_names[t] = image.name

    # Remove ununsed 3d points.
    ind = np.where(used_3d)[0]
    orig_map = np.full(len(wrld_pts), -1, dtype=int)
    orig_map[ind] = range(len(ind))
    wrld_pts = np.array([wrld_pts[i] for i in ind]).T

    for t in im_pts_at_time:
        xys, point3D_ind = im_pts_at_time[t]
        ind = point3D_ind >= 0
        xys = xys[:, ind]
        point3D_ind = point3D_ind[ind]
        point3D_ind = orig_map[point3D_ind]
        im_pts_at_time[t] = xys, point3D_ind

    return sfm_cm, im_pts_at_time, wrld_pts, image_names


def stereo_pair_marginalize_pts(K, im_pts1, im_pts2, pixel_sigma=3, max_sep=1000,
                            max_viz_dist=1e4):
    im_pts1 = im_pts1.astype(np.float32)
    im_pts2 = im_pts2.astype(np.float32)
    E, mask = cv2.findEssentialMat(im_pts1.T, im_pts2.T, K, threshold=3)
    retval, R, t, mask, points3d0 = cv2.recoverPose(E, im_pts1.T, im_pts2.T, K,
                                                    distanceThresh=1e10,
                                                    mask=mask)
    mask = mask.ravel() > 0
    im_pts1 = im_pts1[:, mask]
    im_pts2 = im_pts2[:, mask]
    points3d0 = points3d0[:, mask]

    cam2_pos = -np.dot(R.T, t).ravel()

    if True:
        # This produces triangulation that has a balanced consistency with both
        # camera views.
        P1 = np.hstack([K, np.zeros((3, 1))])
        t *= max_sep
        P2 = np.hstack([np.dot(K, R), t])
        points3d0 = cv2.triangulatePoints(P1, P2, im_pts1, im_pts2)

    # Do a chirality check for the points.
    points3d0 = points3d0/points3d0[3]
    points3d0[:2] = points3d0[:2]*np.sign(points3d0[2])

    xyz = np.dot(np.hstack([R, t]), points3d0)
    mask = xyz[2] > 0
    im_pts1 = im_pts1[:, mask]
    im_pts2 = im_pts2[:, mask]
    points3d0 = points3d0[:, mask]

    if False:
        N = 300
        im_pts1 = im_pts1[:, :N]
        im_pts2 = im_pts2[:, :N]
        points3d0 = points3d0[:, :N]

    if False:
        # Sanity check.
        P1 = np.hstack([K, np.zeros((3, 1))])
        P2 = np.hstack([np.dot(K, R), t])
        im_pts1_ = np.dot(P1, points3d0)
        im_pts2_ = np.dot(P2, points3d0)
        im_pts1_ = im_pts1_[:2]/im_pts1_[2]
        im_pts2_ = im_pts2_[:2]/im_pts2_[2]

        N = im_pts1.shape[1]
        Sigma = np.diag([pixel_sigma**2, pixel_sigma**2])
        inv_sigma = np.linalg.inv(Sigma)
        dx = im_pts1 - im_pts1_
        err1 = sum([np.dot(np.dot(dx_, inv_sigma), dx_)/2 for dx_ in dx.T])
        dx = im_pts2 - im_pts2_
        err2 = sum([np.dot(np.dot(dx_, inv_sigma), dx_)/2 for dx_ in dx.T])
        err = err1 + err2
        print('Total error: ', err)

        #plt.plot(im_pts2[0], im_pts2[1], 'go')
        #plt.plot(im_pts2_[0], im_pts2_[1], 'bo')
        print(sqrt(np.mean(np.sum((im_pts1 - im_pts1_)**2, 0))))
        print(sqrt(np.mean(np.sum((im_pts2 - im_pts2_)**2, 0))))

    points3d = (points3d0[:3]/points3d0[3]).T

    graph = NonlinearFactorGraph()

    gtsam_camera = Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])

    poses = [Pose3(Rot3(np.identity(3)), [0, 0, 0]),
             Pose3(Rot3(R.T), cam2_pos)]

    if True:
        factor = gtsam.NonlinearEqualityPose3(X(0), poses[0])
    else:
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(np.ones(6)*1e-6))
        factor = PriorFactorPose3(X(0), poses[0], pose_noise)

    graph.push_back(factor)

    #factor = PriorFactorPose3(X(1), poses[1], pose_noise)
    #graph.push_back(factor)

    if True:
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e6, 1e6, 1e6,
                                                                max_sep*50,
                                                                max_sep*50,
                                                                max_sep*50]))
        factor = PriorFactorPose3(X(1), poses[1], pose_noise)
        graph.push_back(factor)

    measurement_noise1 = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
    measurement_noise2 = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)

    for j in range(im_pts1.shape[1]):
        factor = GenericProjectionFactorCal3_S2(im_pts1[:, j],
                                                measurement_noise1,
                                                X(0), L(j),
                                                gtsam_camera, True, True)
        graph.push_back(factor)
        factor = GenericProjectionFactorCal3_S2(im_pts2[:, j],
                                                measurement_noise2,
                                                X(1), L(j),
                                                gtsam_camera, True, True)
        graph.push_back(factor)

    initial_estimate = Values()
    for i, pose in enumerate(poses):
        initial_estimate.insert(X(i), pose)

    max_viz_dist_noise = gtsam.noiseModel.Diagonal.Sigmas([max_viz_dist,
                                                           max_viz_dist,
                                                           max_viz_dist])
    for j in range(len(points3d)):
        initial_estimate.insert(L(j), points3d[j])

        if True:
            # Constrain the points are nearby.
            factor = PriorFactorPoint3(L(j), points3d[j],
                                       max_viz_dist_noise)
            graph.push_back(factor)

    params = gtsam.LevenbergMarquardtParams()
    #params.setMaxIterations(1000)
    #params.setAbsoluteErrorTol(1e-16)
    #params.setRelativeErrorTol(1e-16)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate,
                                                  params)

    #err0 = optimizer.error()
    result = optimizer.optimize()

    marginals = gtsam.Marginals(graph, result)
    pose_cov = marginals.marginalCovariance(X(1))
    pose = result.atPose3(X(1))

    if False:
        # Sanity check.
        wrld_pts_ = np.array([result.atPoint3(L(i))
                              for i in range(len(points3d))]).T
        wrld_pts_ = np.vstack([wrld_pts_, np.ones(wrld_pts_.shape[1])])
        R_ = pose.rotation().matrix().T
        cam_pos_ = np.atleast_2d(pose.translation()).T
        P2 = np.hstack([np.dot(K, R_), -np.dot(R_, cam_pos_)])
        P1 = np.hstack([K, np.zeros((3, 1))])
        im_pts1_ = np.dot(P1, wrld_pts_)
        im_pts2_ = np.dot(P2, wrld_pts_)
        im_pts1_ = im_pts1_[:2]/im_pts1_[2]
        im_pts2_ = im_pts2_[:2]/im_pts2_[2]
        #plt.plot(im_pts2[0], im_pts2[1], 'go')
        #plt.plot(im_pts2_[0], im_pts2_[1], 'bo')

        N = im_pts1.shape[1]
        Sigma = np.diag([pixel_sigma**2, pixel_sigma**2])
        inv_sigma = np.linalg.inv(Sigma)
        dx = im_pts1 - im_pts1_
        err1 = sum([np.dot(np.dot(dx_, inv_sigma), dx_)/2 for dx_ in dx.T])
        dx = im_pts2 - im_pts2_
        err2 = sum([np.dot(np.dot(dx_, inv_sigma), dx_)/2 for dx_ in dx.T])
        err = err1 + err2
        print('Total error: ', err)

        print(sqrt(np.mean(np.sum((im_pts1 - im_pts1_)**2, 0))))
        print(sqrt(np.mean(np.sum((im_pts2 - im_pts2_)**2, 0))))

    return pose, pose_cov


class OfflineSLAM(object):
    def __init__(self, cm, min_pos_std=None, min_orientation_std=None,
                 pixel_sigma=10, ins_drift_rate=[0.1, 0.5],
                 imu_drift_rate=[0.1, 0.5], robust_pixels_k=4, robust_ins_k=4,
                 max_viz_dist=None, balance_measurements=False,
                 estimate_camera=False):
        """Initialize the offline SLAM instance.

        Parameters
        ----------
        sfm_cm : camera_models.StandardCamera
            Camera model for the camera the acquired the image feature points.

        :min_pos_std: float, 3-array of float | None
            GPS and INS may at times report a covariance for estimate position
            that is overconfident. Therefore, this parameter clamps the assumed
            estimated position standard deviation to at least this minimum
            value. If a scalar, it applied to the x, y, and z components of the
            position covariance. If a 3-array, it specifies each axis
            seperately.

        :min_orientation_std: float, 3-array of float | None
            INS may at times report a covariance for estimated orientation that
            is overconfident. Therefore, this parameter clamps the assumed
            estimated orientation standard deviations to at least this minimum
            value (radians). If a scalar, it applied to the yaw, pitch, and
            roll components of the covariance uniformly. If a 3-array, it
            specifies each (yaw, pitch, roll) seperately.

        :pixel_sigma: float
            Estimated image feature location is assumed to subject to random
            noise with standard deviation equal to this value.

        :ins_drift_rate: 2-array
            Model for the drift due to noise of the accelerometer and
            gyroscope. The first element is the drift in position in
            m/sqrt(hr). The second element is orientation drift in
            deg/sqrt(hr).

        :imu_drift_rate: 2-array
            Model for the drift due to noise of the accelerometer and
            gyroscope. The first element is the accelerometer velocity random
            walk (VRW) in m/sec/sqrt(hr). The second element is the gyro angle
            random walk (ARW) in deg/sec/sqrt(hr).

        :robust_pixels_k: float

        robust_ins_k: float

        :param max_viz_dist: float
            Maximum visible distance for landmark points. This adds a prior to
            make reconstruction more stable.

        :balance_measurements: bool
            Create an instance of the prior for pose per image feature
            measurement. This balances the weighting of image features versus
            INS pose measurements.

        :estimate_camera: bool
            Estimate a correction to the orientation of the camera relative to
            the INS. This will update cam_quat of the camera model output by
            'convert_solution'.
        """
        # The camera that images the world may have distortion, but modeling it
        # explicitly within the GTSAM optimization wastes repeated computation.
        # So, we instead define a pinhole camera model that covers the field of
        # view and locally through the image, one original-image pixel is
        # approximately equal to one pinhole-camera pixel. Therefore, assuming
        # distortion isn't too large, the calculated reprojection error won't
        # change much.
        self.cm = cm
        cm_pinhole = fit_pinhole_camera(cm)
        self.pinhole_K = cm_pinhole.K

        # Define a GTSAM pinhole camera model (no distortion) that we will use
        # to optimize pose during SLAM.
        self.gtsam_camera = Cal3_S2(cm_pinhole.fx, cm_pinhole.fy, 0.0,
                                    cm_pinhole.cx, cm_pinhole.cy)

        if min_pos_std is not None:
            if not hasattr(min_pos_std, "__len__"):
                min_pos_std = [min_pos_std, min_pos_std, min_pos_std]

            min_pos_std = np.array(min_pos_std)

        if min_orientation_std is not None:
            if not hasattr(min_orientation_std, "__len__"):
                min_orientation_std = [min_orientation_std, min_orientation_std,
                                       min_orientation_std]

            min_orientation_std = np.array(min_orientation_std)

        if imu_drift_rate is not None:
            if len(imu_drift_rate) == 2:
                imu_drift_rate = [imu_drift_rate[0], imu_drift_rate[0], imu_drift_rate[0],
                              imu_drift_rate[1], imu_drift_rate[1], imu_drift_rate[1]]

            imu_drift_rate = np.array(imu_drift_rate)

        if ins_drift_rate is not None:
            if len(ins_drift_rate) == 2:
                ins_drift_rate = [ins_drift_rate[0], ins_drift_rate[0], ins_drift_rate[0],
                              ins_drift_rate[1], ins_drift_rate[1], ins_drift_rate[1]]

            ins_drift_rate = np.array(ins_drift_rate)

        self.ins_drift_rate = ins_drift_rate
        self.imu_drift_rate = imu_drift_rate
        self.min_pos_std = min_pos_std
        self.min_orientation_std = min_orientation_std
        self.pixel_sigma = pixel_sigma
        self.robust_ins_k = robust_ins_k
        self.robust_pixels_k = robust_pixels_k
        self.max_viz_dist = max_viz_dist
        self.balance_measurements = balance_measurements
        self.estimate_camera = estimate_camera

        # Rotation matrix that moves vectors from ins coordinate system into
        # camera coordinate system.
        Rcam = cm.get_camera_pose(0)[:, :3]
        Rins = quaternion_matrix(cm.platform_pose_provider.pose(0)[1])[:3, :3].T
        self.Rins_to_cam = np.dot(Rcam, Rins.T)

        self.reset_graph()

    def __str__(self):
        string = ['OfflineSLAM\n']
        string.append('min_pos_std: %s\n' % str(self.min_pos_std))
        string.append('min_orientation_std: %s\n' % str(self.min_orientation_std))
        string.append('ins_drift_rate: %s\n' % str(self.ins_drift_rate))
        string.append('imu_drift_rate: %s\n' % str(self.imu_drift_rate))
        string.append('max_viz_dist: %s\n' % str(self.max_viz_dist))
        string.append('pixel_sigma: %s\n' % str(self.pixel_sigma))
        string.append('robust_ins_k: %s\n' % str(self.robust_ins_k))
        string.append('robust_pixels_k: %s\n' % str(self.robust_pixels_k))
        string.append('balance_measurements: %s\n' % str(self.balance_measurements))
        return ''.join(string)

    def __repr__(self):
        return self.__str__()

    def reset_graph(self):
        self.graph = NonlinearFactorGraph()
        self.num_landmarks = None

    def pose_at_time_ins(self, t, weight=1):
        """Return INS-predicted pose and pose_noise for time t.

        :weight: float
            Weighting of the log likelihood desired when using pose_noise.
        """
        # Provide a prior derived from the INS.
        min_pos_std = self.min_pos_std
        min_orientation_std = self.min_orientation_std
        pos_ins, quat_ins, std = self.cm.platform_pose_provider.pose(t,
                                                                     return_std=True)
        std_e, std_n, std_u, std_y, std_p, std_r = std

        if self.min_pos_std is not None:
            std_e = max([std_e, min_pos_std[0]])
            std_n = max([std_n, min_pos_std[1]])
            std_u = max([std_u, min_pos_std[2]])

        if self.min_orientation_std is not None:
            std_y = max([std_y, min_orientation_std[0]])
            std_p = max([std_p, min_orientation_std[1]])
            std_r = max([std_r, min_orientation_std[2]])

        Rins = quaternion_matrix(quat_ins)[:3, :3].T

        # This is the estimate for the pose of the camera as predicted by
        # the INS and the previous calibration ('Rins_to_cam') for the
        # orientation of the camera relative to the INS. This is defined
        # such that when it operates on a vector defined in the world, it
        # returns the specification of that vector within the camera
        # coordinate system.
        Rcam_from_ins = np.dot(self.Rins_to_cam, Rins)

#            if cm_sfm is not None:
#                P = cm_sfm.get_camera_pose(t)
#                Rcam = P[:, :3]
#                pos = np.dot(-Rcam.T, P[:, 3])
#            else:
#                P = self.cm.get_camera_pose(t)

        pose = Pose3(Rot3(Rcam_from_ins.T), pos_ins)

        # Vector defined within the INS coordinate system representing the
        # uncertainty of its orientation
        rot_ind_std = np.array([std_r, std_p, std_y])

        # Rotate into camera coordinate system
        std_r, std_p, std_y = np.abs(np.dot(self.Rins_to_cam, rot_ind_std))

        # Pose standard deviation defined in roll (rad), pitch (rad), yaw (rad), x
        # (m), y (m), z (m).
        d = np.array([std_r, std_p, std_y, std_e, std_n, std_u])
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(d/sqrt(weight))

        if self.robust_ins_k is not None:
            huber = gtsam.noiseModel.mEstimator.Huber.Create(self.robust_ins_k)
            pose_noise = gtsam.noiseModel.Robust.Create(huber, pose_noise)

        return pose, pose_noise

    def define_problem(self, im_pts_at_time, wrld_pts, imu_data=None,
                       time_uncertainty=None):
        """
        Parameters
        ----------
        imu_data : N x 13 array
            High-rate IMU data with the following columns:
            [0] time (s)
            [1] accel-x (m/s^2)
            [2] accel-y (m/s^2)
            [3] accel-z (m/s^2)
            [4] gyro-x (rad/s)
            [5] gyro-y (rad/s)
            [6] gyro-z (rad/s)
            [7] mag-x (Ga)
            [8] mag-y (Ga)
            [9] mag-z (Ga)
            [10] imu-temp (C)
            [11] pressure (Pa)
            [12] ambient-temp (C)

        :time_uncertainty: float
            The synchronization between image times and INS data times is
            assumed to be within plus or minus this value (s).

        """
        print('Calculating initial poses and adding pose prior estimates')
        tic = time.time()

        self.initial_estimate = Values()

        if self.estimate_camera:
            # 'self.Rins_to_cam', derived from the camera model's cam_quat
            # quaternion, hopefully provides a good initial guess for the
            # orientation of the camera relative to the navigation coordinate
            # system encoded by 'cm.platform_provider'. We want to solve for an
            # additional rotation beyond the navigation-estimated pose,
            # applied in the initially-estimated camera coordinate system so
            # that the overall solution better agrees with the image feature
            # measurements. However, gtsam does not have a direct way to do
            # this. Instead, we apply a hack to the gtsam capability used to
            # estimate IMU/gyro bias. For each ith image exposure time, we
            # define two poses, X(i)
            params = gtsam.PreintegrationParams.MakeSharedU(0)
            params.setAccelerometerCovariance(1e-5*np.identity(3))
            params.setGyroscopeCovariance(1e-5*np.identity(3))
            params.setIntegrationCovariance(1e-5*np.identity(3))
            params.setOmegaCoriolis(np.array([0, 0, 0]))
            self.fake_imu_params = params
            zeroBias = gtsam.imuBias.ConstantBias(np.array([0, 0, 0]),
                                                  np.array([0, 0, 0]))
            fake_pim = gtsam.PreintegratedImuMeasurements(params, zeroBias)
            self.fake_pim = fake_pim
            fake_pim.integrateMeasurement(np.zeros(3), np.zeros(3), 1)
            self.initial_estimate.insert(W(0), np.zeros(3))
            self.initial_estimate.insert(W(1), np.zeros(3))
            self.initial_estimate.insert(DUMMY_BIAS_KEY, zeroBias)
            same_pos_noise = gtsam.noiseModel.Diagonal.Sigmas([1e10,1e10,1e10,
                                                               1e-1,1e-1,
                                                               1e-1])

        if self.ins_drift_rate is not None:
            ins_drift_rate = np.zeros(6)

            # Swap the order since gtsam.noiseModel.Diagonal.Sigmas expects
            # orientation uncertainty first.

            # Convert from m/sqrt(hr) to m/sqrt(s)
            ins_drift_rate[3:] = ins_drift_rate[:3]/60

            # Convert from deg/sqrt(hr) to rad/sqrt(s)
            ins_drift_rate[:3] = ins_drift_rate[3:]*np.pi/180/60
        else:
            ins_drift_rate = None

        if imu_data is not None:
            # Set up all components related to integrating inertial measurement
            # unit outputs.
            assert self.imu_drift_rate is not None

            imu_data = imu_data[:, :7]
            ind = np.argsort(imu_data[:, 0])
            imu_data = imu_data[ind, :7]
            imu_times = imu_data[:, 0]

            # The constant value to assume over the whole time bin where
            # you are in bin bisect.bisect_right(imu_times, t) - 1 for time t.
            accel_gyro_data = (imu_data[:-1, 1:7] + imu_data[1:, 1:7])/2

            # Rotate from the INS coordinate system into the camera coordinate
            # system.
            accel_gyro_data[:, :3] = np.dot(self.Rins_to_cam, accel_gyro_data[:, :3].T).T
            accel_gyro_data[:, 3:] = np.dot(self.Rins_to_cam, accel_gyro_data[:, 3:].T).T

            gravity = 0
            #gravity = 9.81
            imu_params = gtsam.PreintegrationParams.MakeSharedU(gravity)

            # Convert from m/s/sqrt(hr) to m/s/sqrt(s)
            accel_sigma = self.imu_drift_rate[:3]/60

            # Convert from deg/s/sqrt(hr) to rad/s/sqrt(s)
            gyro_sigma = self.imu_drift_rate[3:]*np.pi/180/60

            imu_params.setAccelerometerCovariance(np.diag(accel_sigma**2))
            imu_params.setGyroscopeCovariance(np.diag(gyro_sigma**2))
            print(imu_params)

            #params.setIntegrationCovariance(np.zeros((3, 3)))
            imu_params.setIntegrationCovariance(1e-3*np.identity(3, np.float))

            # We are only integrating over a small time, so we can ignore the
            # coriolis rate for this application.
            imu_params.setOmegaCoriolis(np.array([0, 0, 0]))

            # We use zero bias here since we are just looking to smooth out
            # rapid changes that shouldn't exist. We are still relying on the
            # INS state output for the general constraining.
            zeroBias = gtsam.imuBias.ConstantBias(np.array([0, 0, 0]),
                                                  np.array([0, 0, 0]))

            pim = gtsam.PreintegratedImuMeasurements(imu_params, zeroBias)

            self.initial_estimate.insert(BIAS_KEY, zeroBias)

        image_times = sorted(list(im_pts_at_time.keys()))
        poses = []

        # Create the set of ground-truth landmarks
        points3d = wrld_pts.T
        wrld_pts_used = np.zeros(len(points3d), dtype=bool)


        if self.max_viz_dist is not None:
            max_viz_dist_noise = gtsam.noiseModel.Diagonal.Sigmas([self.max_viz_dist,
                                                                   self.max_viz_dist,
                                                                   self.max_viz_dist])
            origin = Point3(np.zeros(3))
            max_viz_ref = [origin for _ in range(len(points3d))]

        for i, t in enumerate(image_times):
            # Loop over all images

            # This image has 'N' points.
            im_pts0, point3D_ind = im_pts_at_time[t]

            # These image points are within the, potentially, distorted camera.
            # We need to map them to where they would have been imaged into the
            # proxy pinhole camera model.
            im_pts = cv2.undistortPoints(im_pts0.astype(np.float32), self.cm.K,
                                         self.cm.dist, None, self.pinhole_K)
            im_pts = im_pts.squeeze(axis=1).T

            if self.balance_measurements:
                weight = sqrt(im_pts.shape[1])
            else:
                weight = 1

            # ------------------- Add pose prior from INS  -------------------
            pose, pose_noise = self.pose_at_time_ins(t)
            poses.append(pose)

            if ins_drift_rate is not None and i > 0:
                for j in range(i):
                    dt = image_times[i] - image_times[j]
                    if dt > 1:
                        continue

                    dpose_noise = gtsam.noiseModel.Diagonal.Sigmas(ins_drift_rate*sqrt(dt))

                    if self.robust_ins_k is not None:
                        huber = gtsam.noiseModel.mEstimator.Huber.Create(self.robust_ins_k)
                        dpose_noise = gtsam.noiseModel.Robust.Create(huber, dpose_noise)

                    dpose = poses[j].between(poses[i])

                    factor = gtsam.BetweenFactorPose3(X(j), X(i), dpose,
                                                      dpose_noise)
                    self.graph.push_back(factor)

            if imu_data is not None and i < len(image_times) - 1:
                pim.resetIntegration()

                # We need to consider the imu outputs between times t1 and t2
                t1 = image_times[i]
                t2 = image_times[i + 1]

                if t2 - t1 < 5:
                    # This is the index for either an exact match for
                    # imu_times[ind1]=t1 or largest imu_times where imu_times[ind1]<=t1.
                    # This is the first bin that partially intersects with the timespan
                    # t1->t2. If negative, that means that there all imu_times > t1.
                    ind1 = bisect.bisect_right(imu_times, t1) - 1

                    # ind2 is the smallest imu_times value such that t2 < imu_times[ind2].
                    # If ind2 == len(imu_times), that means there is no time such that
                    # all imu_times < t2.
                    ind2 = bisect.bisect_right(imu_times, t2)

                    if ind2 == ind1 + 1:
                        # t1 and t2 live entirely inside the imu_times[ind1]->imu_times[ind2]
                        dt = t2 - t1
                        pim.integrateMeasurement(accel_gyro_data[ind1, :3],
                                                 accel_gyro_data[ind1, 3:], dt)
                    else:
                        if ind1 >= 0 and ind2 > 1:
                            # t1 to t1 + dt is inside imu_times[ind1] -> imu_times[ind1 + 1].
                            dt = imu_times[ind1 + 1] - t1
                            assert dt > 0
                            pim.integrateMeasurement(accel_gyro_data[ind1, :3],
                                                     accel_gyro_data[ind1, 3:], dt)

                        for ind in range(ind1+1, ind2 - 1):
                            # These are bins that are entire covered by the range t1 -> t2.
                            dt = imu_times[ind + 1] - imu_times[ind]
                            pim.integrateMeasurement(accel_gyro_data[ind, :3],
                                                     accel_gyro_data[ind, 3:], dt)

                        if ind2 < len(imu_times):
                            dt = t2 - imu_times[ind2 - 1]
                            if dt > 0:
                                pim.integrateMeasurement(accel_gyro_data[ind2-1, :3],
                                                         accel_gyro_data[ind2-1, 3:], dt)

                    print(np.diag(pim.preintMeasCov()))

                    factor = gtsam.ImuFactor(X(i), V(i), X(i+1),
                                             V(i+1), BIAS_KEY, pim)
                    self.graph.push_back(factor)

            if time_uncertainty is None:
                factor = PriorFactorPose3(X(i), poses[i], pose_noise)
                self.graph.push_back(factor)
            else:
                # We don't exactly know where in the +/- time_uncertainty we
                # reside, so we put a sampling of all so that we smooth out the
                # prior over the possible times.
                num_times = 50
                ts = np.linspace(t - time_uncertainty, t + time_uncertainty,
                                 num_times)
                for t_ in ts:
                    pose, pose_noise = self.pose_at_time_ins(t_,
                                                             weight=1/num_times)
                    factor = PriorFactorPose3(X(i), pose, pose_noise)
                    self.graph.push_back(factor)

            # -------------- Add image measurements for this image -----------

            measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2,
                                                                 self.pixel_sigma*weight)

            # The idea here is that we want to balance the contribution to the
            # overall likelihood between an INS factor and the factors from
            # image measurements on one image. For delta^2 loss, where delta is
            # a normalized version of some paramenter normalized by its
            # standard deviation, when delta is 1 (i.e., 1-sigma) dLoss/ddelta
            # is 2. If we similarly express the pixel error in units of sigma,
            # we derivative of the sum of the loss over all pixel measurements
            # in an image to equal 2. So, we can use Huber loss to effect this
            # by chosing Huber k such that when all pixels errors are at
            # 1-sigma, the sum of the derivatives equals 2.
            # huber loss = (delta)^2/2 up to k with loss gradient x. So at k,
            # slope isk forever. So, k = 2/N, where N is the number of points.
            #robust_pixels_k = pixel_weight*2/N
            #robust_pixels_k = 1.5
            robust_pixels_k = self.robust_pixels_k

#            robust_pixels_k = self.robust_pixels_k
#            pixel_weight = 1
#            robust_pixels_k = pixel_weight*2/N

            if robust_pixels_k is not None:
                huber = gtsam.noiseModel.mEstimator.Huber.Create(robust_pixels_k)
                measurement_noise = gtsam.noiseModel.Robust.Create(huber,
                                                                   measurement_noise)

            if self.estimate_camera:
                # We apply a hack that uses gtsam's IMU/gyro bias estimation to
                # solve for a correction to 'self.Rins_to_cam'. This isn't a
                # real IMU/gyro that we are modeling, hence the name fake_pim.
                # See the earlier comment in this method for more details.
                pose_sym = Y(i)
                self.initial_estimate.insert(Y(i), poses[i])
                factor = gtsam.ImuFactor(X(i), W(0), Y(i), W(1),
                                         DUMMY_BIAS_KEY, fake_pim)
                self.graph.push_back(factor)

                factor = gtsam.BetweenFactorPose3(X(i), Y(i), Pose3(),
                                                  same_pos_noise)
                self.graph.push_back(factor)
            else:
                pose_sym = X(i)

            for j in range(im_pts.shape[1]):
                factor = GenericProjectionFactorCal3_S2(im_pts[:, j],
                                                        measurement_noise,
                                                        pose_sym,
                                                        L(point3D_ind[j]),
                                                        self.gtsam_camera)
                wrld_pts_used[point3D_ind[j]] = True
                self.graph.push_back(factor)

                if self.max_viz_dist is not None:
                    max_viz_ref[j] = poses[-1].translation()

        # Create the data structure to hold the initial estimate to the
        # solution intentionally initialize the variables off from the ground
        # truth.
        print('Setting initial solution')

        self.pose_times = image_times
        for i, pose in enumerate(poses):
            self.initial_estimate.insert(X(i), pose)

            if imu_data is not None:
                self.initial_estimate.insert(V(i), np.zeros(3))

        self.wrld_pts_orig = wrld_pts
        self.wrld_pts_orig_ind = np.where(wrld_pts_used)[0]
        for j in self.wrld_pts_orig_ind:
            point = points3d[j]
            self.initial_estimate.insert(L(j), point)

            if self.max_viz_dist is not None:
                # Constrain the points are nearby.
                factor = PriorFactorPoint3(L(j), max_viz_ref[j],
                                           max_viz_dist_noise)
                self.graph.push_back(factor)

        self.result = self.initial_estimate

        print('Time elapsed:', time.time() - tic)

    def update_camera_parameters(self, pixel_sigma=10, robust_pixels_k=4):
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
        huber = gtsam.noiseModel.mEstimator.Huber.Create(robust_pixels_k)
        measurement_noise = gtsam.noiseModel.Robust.Create(huber,
                                                           measurement_noise)
        for i in range(self.graph.size()):
            f = self.graph.at(i)
            if isinstance(f, GenericProjectionFactorCal3_S2):
                raise Exception()
                xi, lj = f.keys()
                f2 = GenericProjectionFactorCal3_S2(f.measured(),
                                                    measurement_noise,
                                                    xi, lj, f.calibration())
                self.graph.replace(i, f2)

    def update_reproj_sigma(self, pixel_sigma=10, robust_pixels_k=4):
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
        huber = gtsam.noiseModel.mEstimator.Huber.Create(robust_pixels_k)
        measurement_noise = gtsam.noiseModel.Robust.Create(huber,
                                                           measurement_noise)
        for i in range(self.graph.size()):
            f = self.graph.at(i)
            if isinstance(f, GenericProjectionFactorCal3_S2):
                xi, lj = f.keys()
                f2 = GenericProjectionFactorCal3_S2(f.measured(),
                                                    measurement_noise,
                                                    xi, lj, f.calibration())
                self.graph.replace(i, f2)

    def solve(self):
        """Solve problem, set self.results and update self.initial_estimate.
        """
        tic = time.time()
        # Optimize the graph and print results
        print('Running optimizer')
        if False:
            params = gtsam.DoglegParams()
            params.setAbsoluteErrorTol(1e-6)
            params.setRelativeErrorTol(1e-6)
            params.setVerbosity('TERMINATION')
            optimizer = DoglegOptimizer(self.graph, self.initial_estimate, params)
            print('Optimizing:')
        elif False:
            params = gtsam.GaussNewtonParams()
            params.setAbsoluteErrorTol(1e-6)
            params.setRelativeErrorTol(1e-6)
            params.setVerbosity('TERMINATION')
            optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_estimate, params)
            print('Optimizing:')

            err0 = optimizer.error()
            for _ in range(20):
                result = optimizer.optimize()
                err = optimizer.error()
                print('Relative reduction in error:', (err0-err)/err0)
                #err0 = err
        else:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph,
                                                          self.initial_estimate,
                                                          params)

        self.result = optimizer.optimize()
        print('Time elapsed:', time.time() - tic)
        self.initial_estimate = self.result
        return optimizer.error()

    def convert_solution(self):
        if False:
            marginals = gtsam.Marginals(self.graph, self.result)
            pose_cov = marginals.marginalCovariance(X(1))
            pose = result.atPose3(X(1))

        cm2 = copy.deepcopy(self.cm)

        if self.estimate_camera:
            fake_bias = self.result.atConstantBias(DUMMY_BIAS_KEY)
            print('Bias', fake_bias)
            fake_pim = gtsam.PreintegratedImuMeasurements(self.fake_imu_params,
                                                          fake_bias)
            fake_pim.integrateMeasurement(np.zeros(3), np.zeros(3), 1)
            print('V(0)', self.result.atVector(W(0)))
            ns0 = gtsam.NavState(Pose3(), self.result.atVector(W(0)))
            ns = fake_pim.predict(ns0,
                                  self.result.atConstantBias(DUMMY_BIAS_KEY))
            print('dPos', ns.position())
            R = ns.attitude().matrix()
            #self.Rins_to_cam = np.dot(R.T, self.Rins_to_cam)
            cm2.cam_quat = quaternion_from_matrix(np.dot(R.T, self.Rins_to_cam).T)
            print(R)

        wrld_pts = self.wrld_pts_orig.copy()
        wrld_pts_ = np.array([self.result.atPoint3(L(i))
                              for i in self.wrld_pts_orig_ind]).T
        wrld_pts[:, self.wrld_pts_orig_ind] = wrld_pts_

        ppp = PlatformPoseInterp(self.cm.platform_pose_provider.lat0,
                                 self.cm.platform_pose_provider.lon0,
                                 self.cm.platform_pose_provider.h0)
        for i in range(len(self.pose_times)):
            t = self.pose_times[i]
            pose = self.result.atPose3(X(i))
            R = pose.rotation().matrix().T
            R = np.dot(self.Rins_to_cam.T, R)
            # The gtsam rotation matrix is a coordinate system rotation.
            quat = np.array(quaternion_from_matrix(R.T))
            quat *= np.sign(quat[-1])
            ppp.add_to_pose_time_series(t, [pose.x(), pose.y(), pose.z()], quat)

        cm2.platform_pose_provider = ppp

        position_err = []
        position_err0 = []
        for i, t in enumerate(self.pose_times):
            pos0, quat0 = self.cm.platform_pose_provider.pose(t)
            pos1 = cm2.platform_pose_provider.pos(t)
            position_err.append(np.linalg.norm(pos0 - pos1))
            position_err0.append(np.linalg.norm(pos0 - self.cm.platform_pose_provider.pos(t)))

        print('Mean position difference reduced from', np.mean(position_err0), 'to', np.mean(position_err))

        return cm2, wrld_pts



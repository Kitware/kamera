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
import os
from numpy import pi
import cv2
import time
import yaml
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import fmin, fminbound, minimize
import copy
import pickle
import PIL

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Repository imports.
from colmap_processing.image_renderer import stitch_images
from colmap_processing.platform_pose import PlatformPoseFixed
from colmap_processing.geo_conversions import enu_to_llh, llh_to_enu
import colmap_processing.dp as dp
from colmap_processing.camera_models import ray_intersect_plane


def horn(P, Q, fit_scale=True, fit_translation=True):
    """Method of Horn.

    :param P: Initial point cloud.
    :type P: num_dim x N

    :param Q: Destination point cloud.
    :type Q: num_dim x N

    :return: Scale, rotation matrix, and translation to be applied in that
        order that minimizes the difference between Q and (R*s*P + t).
    :rtype:

    Example:

    s = 213.5
    S = np.diag([s, s, s, 1])
    R = np.identity(4)
    R[:3, :3] = np.array([[0.411982,-0.833738,-0.367630],
                  [-0.058727,-0.426918,0.902382],
                  [-0.909297,-0.350175,-0.224845]])
    t = [1, 2, 3]
    T = np.identity(4)
    T[:3, 3] = t

    H = np.dot(np.dot(T, R), S)[:3]

    xyz1 = np.random.rand(4, 100)
    xyz1[3, :] = 1
    xyz2 = np.dot(H, xyz1)

    # Scale, rotation, translation
    s2, R2, t2 = horn(xyz1[:3], xyz2, fit_scale=True, fit_translation=True)
    print(s, s2)
    print(R[:3, :3], '\n', R2)
    print(t, t2)
    print('Max error:', np.max(np.abs(np.dot(R[:3, :3], s*xyz1[:3, :]).T + t - xyz2.T)))

    # Only rotation
    xyz2 = np.dot(R[:3, :3], xyz1[:3])
    R2 = horn(xyz1[:3], xyz2, fit_scale=False, fit_translation=False)[1]
    print(R[:3, :3], '\n', R2)

    """
    if P.shape != Q.shape:
        print("Matrices P and Q must be of the same dimensionality")

    if fit_translation:
        P0 = np.mean(P, axis=1)
        Q0 = np.mean(Q, axis=1)
        A = P - np.outer(P0, np.ones(P.shape[1]))
        B = Q - np.outer(Q0, np.ones(Q.shape[1]))
    else:
        A = P
        B = Q

    if fit_scale:
        s = np.sqrt(np.mean(B.ravel()**2)) / np.sqrt(np.mean(A.ravel()**2))

        # Apply scale.
        A = s*A

        if fit_translation:
            P0 = P0*s
    else:
        s = 1

    C = np.dot(A, B.transpose())
    U, S, V = np.linalg.svd(C)
    R = np.dot(V.transpose(), U.transpose())
    L = np.eye(3)
    if np.linalg.det(R) < 0:
        L[2][2] *= -1

    R = np.dot(V.transpose(), np.dot(L, U.transpose()))

    if fit_translation:
        t = np.dot(-R, P0) + Q0
    else:
        t = np.zeros(len(P))

    return s, R, t


def fit_plane(xyz):
    """Check whether results from cv2.calibrateCamera are valid.

    :param xyz: 3-D coordinates.
    :type xyz: N x 2

    """
    plane_point = np.mean(xyz, 0)
    x = xyz - np.atleast_2d(plane_point)
    M = np.dot(x.T, x)
    plane_normal = np.linalg.svd(M)[0][:,-1]
    plane_normal *= np.sign(plane_normal[-1])
    return plane_point, plane_normal


def check_valid_camera(width, height, K, dist, rvec, tvec):
    if K[0,0] < 0 or K[1,1] < 1:
        return False

    if K[0, 2] < 0 or K[0, 2] > width:
        return False

    if K[1, 2] < 0 or K[1, 2] > height:
        return False

    return True


def cam_depth_map_plane(camera_model, t, plane_point, plane_normal):
    """Calculate depth map for camera assuming it is viewing plane.

    """
    height = camera_model.height
    width = camera_model.width
    X, Y = np.meshgrid(np.linspace(0.5, width - 0.5, width),
                       np.linspace(0.5, height - 0.5, height))
    im_pts = np.vstack([X.ravel(), Y.ravel()])
    ray_pos, ray_dir = camera_model.unproject(im_pts, t,
                                              normalize_ray_dir=False)

    ip = ray_intersect_plane(plane_point, plane_normal, ray_pos, ray_dir,
                             epsilon=1e-6)

    # depth*ray_dir = ip - ray_pos
    depth = np.sqrt(np.sum((ip - ray_pos)**2, axis=0)/np.sum(ray_dir**2, axis=0))

    depth_map = np.reshape(depth, X.shape)
    return depth_map


def get_rvec_btw_times(cm, t1, t2):
    """Return rotation vector defined in the camera coordinate system.

    Calculate the rotation vector corresponding to the rotation of the camera
    frame from time t1 to t2. The rotation is defined within the coordinate
    system of the camera at the first time.

    :param cm: Needs method `get_camera_pose` accepting the time at which the
        pose is desired.
    :type cm:

    :param t1: First time (seconds).

    :param t2: Second time (seconds).
    :type t2: float
    """
    # R1 takes a world vector and moves it into the coordinate system of the camera
    # at t=image_times[i].
    R1 = cm.get_camera_pose(t=t1)[:, :3]
    # R2 takes a world vector and moves it into the coordinate system of the camera
    # at t=image_times[i + 1].
    R2 = cm.get_camera_pose(t=t2)[:, :3]

    # R2 = R1*R1_2
    R1_2 = np.dot(R1.T, R2)
    return cv2.Rodrigues(R1_2.T)[0].ravel()


def calibrate_camera_to_xyz(im_pts, wrld_pts, height, width,
                            fix_aspect_ratio=True, fix_principal_point=True,
                            fix_k1=True, fix_k2=True, fix_k3=True, fix_k4=True,
                            fix_k5=True, fix_k6=True, plot_results=False,
                            ref_image=None):
    """
    :im_pts: Image coordinates associated with (x, y, z) coordinates.
    :type im_pts: num_pts x 2

    :param wrld_pts: World (x, y, z) coordinates associated with image
        coordinates.
    :type wrld_pts: num_pts x 3

    :param fix_aspect_ratio: Fix the aspect ratio during optimization.
    :type fix_aspect_ratio:

    :param fix_principal_point: Fix the principal point to during optimization.
    :type fix_principal_point:

    :param fix_k1: Fix the 1st distortion coefficient during optimization.
    :type fix_k1: bool

    :param fix_k2: Fix the 2nd distortion coefficient during optimization.
    :type fix_k2: bool

    :param fix_k3: Fix the 3rd distortion coefficient during optimization
    :type fix_k3: bool

    :param fix_k4: Fix the 4rth distortion coefficient during optimization
    :type fix_k4: bool

    :param fix_k5: Fix the 5th distortion coefficient during optimization
    :type fix_k5: bool

    :param fix_k6: Fix the 6th distortion coefficient during optimization
    :type fix_k6: bool

    :param plot_results: Generate Matplotlib figures displaying results.
    :type plot_results: bool

    :param ref_image: Reference image to show during plotting of results.
    :type ref_image: Numpy image

    """
    def monte_carlo_calibrateCamera(min_focal_length=1,
                                    max_focal_length=100000,
                                    fix_aspect_ratio=True,
                                    fix_principal_point=True, fix_k1=True,
                                    fix_k2=True, fix_k3=True, fix_k4=True,
                                    fix_k5=True, fix_k6=True, max_runtime=20,
                                    best_ret=None):

        flags = cv2.CALIB_ZERO_TANGENT_DIST
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS

        if fix_principal_point:
            flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT

        if fix_aspect_ratio:
            flags = flags | cv2.CALIB_FIX_ASPECT_RATIO

        if fix_k1:
            flags = flags | cv2.CALIB_FIX_K1

        if fix_k2:
            flags = flags | cv2.CALIB_FIX_K2

        if fix_k3:
            flags = flags | cv2.CALIB_FIX_K3

        if fix_k4:
            flags = flags | cv2.CALIB_FIX_K4

        if fix_k5:
            flags = flags | cv2.CALIB_FIX_K5

        if fix_k6:
            flags = flags | cv2.CALIB_FIX_K6

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000,
                    0.0000001)

        min_flog = np.log10(min_focal_length)
        max_flog = np.log10(max_focal_length)

        if best_ret is None:
            best_err = np.inf
        else:
            best_err = best_ret[0]

        cutoff_time = time.time() + max_runtime
        while time.time() < cutoff_time:
            # Monte Carlo starting parameters.

            # Draw focal length logarithmic uniformly over exponents.
            n = float(np.random.rand(1))*(max_flog - min_flog) + min_flog
            f = 10**n
            K = np.identity(3)
            K[0, 2] = width/2
            K[1, 2] = height/2
            K[0, 0] = K[1, 1] = f

            ret = cv2.calibrateCamera([wrld_pts.astype(np.float32)],
                                      [im_pts.astype(np.float32)],
                                      (width, height), cameraMatrix=K,
                                      distCoeffs=np.zeros(5), flags=flags,
                                      criteria=criteria)

            err, K, dist, rvecs, tvecs = ret
            rvec, tvec = rvecs[0], tvecs[0]
            ret = err, K, dist, rvec, tvec

            if not check_valid_camera(width, height, K, dist, rvec, tvec):
                continue

            im_pts2 = cv2.projectPoints(wrld_pts, rvec, tvec, K, dist)
            err_ = np.sqrt(np.sum((np.squeeze(im_pts2[0]) - im_pts)**2, axis=1))

            # Check that the world points are actually in front of the camera.
            R = cv2.Rodrigues(rvec)[0]
            cam_pos = -np.dot(R.T, tvec).ravel()
            ray_dir0 = (wrld_pts - cam_pos).T
            ray_dir0 /=  np.sqrt(np.sum(ray_dir0**2, 0))
            ray_dir0 = np.dot(R, ray_dir0)

            ray_dir = np.ones((3, len(im_pts)), dtype=np.float)
            ray_dir[:2] = np.squeeze(cv2.undistortPoints(im_pts, K, dist, R=None), 1).T
            ray_dir /=  np.sqrt(np.sum(ray_dir**2, 0))

            if np.any(np.sum(ray_dir * ray_dir0, axis=0) < 0.999):
                continue

            # Ignore correspondences that are 10X the mean error.
            ind = err < 10*err_.mean()

            ret = cv2.calibrateCamera([wrld_pts[ind].astype(np.float32)],
                                      [im_pts[ind].astype(np.float32)],
                                      (width, height), cameraMatrix=K,
                                      distCoeffs=np.zeros(5), flags=flags,
                                      criteria=criteria)

            err, K, dist, rvecs, tvecs = ret
            ret = err, K, dist, rvecs[0], tvecs[0]

            if not check_valid_camera(width, height, K, dist, rvec, tvec):
                continue

            if err < best_err:
                best_err = err
                best_ret = ret
                print('Current reprojection error:', best_err)

        return best_ret

    print('First Pass')
    best_ret = monte_carlo_calibrateCamera(min_focal_length=1,
                                           max_focal_length=100000,
                                           fix_aspect_ratio=True,
                                           fix_principal_point=True,
                                           fix_k1=True, fix_k2=True,
                                           fix_k3=True, fix_k4=True,
                                           fix_k5=True, fix_k6=True,
                                           max_runtime=20, best_ret=None)

    err, K, dist, rvec, tvec = best_ret

    if not fix_aspect_ratio or not fix_principal_point:
        print('Second Pass')
        best_ret = monte_carlo_calibrateCamera(min_focal_length=min(K[0, 0],
                                                                    K[1, 1])/2,
                                               max_focal_length=max(K[0, 0],
                                                                    K[1, 1])*2,
                                               fix_aspect_ratio=fix_aspect_ratio,
                                               fix_principal_point=True,
                                               fix_k1=fix_k1, fix_k2=True,
                                               fix_k3=True, fix_k4=True,
                                               fix_k5=True, fix_k6=True,
                                               max_runtime=10,
                                               best_ret=best_ret)

        err, K, dist, rvec, tvec = best_ret

    if not np.all([fix_k1, fix_k2, fix_k3, fix_k4, fix_k5, fix_k6]):
        print('Final Pass')
        best_ret = monte_carlo_calibrateCamera(min_focal_length=min(K[0, 0],
                                                                    K[1, 1])*0.9,
                                               max_focal_length=max(K[0, 0],
                                                                    K[1, 1])/0.9,
                                               fix_aspect_ratio=fix_aspect_ratio,
                                               fix_principal_point=True,
                                               fix_k1=fix_k1, fix_k2=fix_k2,
                                               fix_k3=fix_k3, fix_k4=fix_k4,
                                               fix_k5=fix_k5, fix_k6=fix_k6,
                                               max_runtime=10,
                                               best_ret=best_ret)

        err, K, dist, rvec, tvec = best_ret

    if not fix_principal_point:
        best_ret = monte_carlo_calibrateCamera(min_focal_length=min(K[0, 0],
                                                                    K[1, 1]),
                                               max_focal_length=max(K[0, 0],
                                                                    K[1, 1]),
                                               fix_aspect_ratio=fix_aspect_ratio,
                                               fix_principal_point=fix_principal_point,
                                               fix_k1=fix_k1, fix_k2=fix_k2,
                                               fix_k3=fix_k3, fix_k4=fix_k4,
                                               fix_k5=fix_k5, fix_k6=fix_k6,
                                               max_runtime=0.01,
                                               best_ret=best_ret)

        err, K, dist, rvec, tvec = best_ret

    def plot_proj_err(im_pts, wrld_pts, K, dist, rvec, tvec):
        im_pts2 = cv2.projectPoints(wrld_pts, rvec, tvec, K, dist)
        im_pts2 = np.squeeze(im_pts2[0])
        plt.imshow(ref_image)
        for i in range(len(im_pts)):
            plt.plot(im_pts[i, 0], im_pts[i, 1], 'bo')
            plt.plot(im_pts2[i, 0], im_pts2[i, 1], 'ro')
            plt.plot([im_pts[i, 0], im_pts2[i, 0]],
                     [im_pts[i, 1], im_pts2[i, 1]], 'k--')

    if plot_results:
        plt.figure()
        plot_proj_err(im_pts, wrld_pts, K, dist, rvec, tvec)

        plt.figure()
        plt.imshow(cv2.undistort(ref_image, K, dist))
        plt.axis('off')
        plt.title('Undistorted', fontsize=20)

    return K, dist, rvec, tvec
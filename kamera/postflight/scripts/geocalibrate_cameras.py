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
import cv2
import os
import glob
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Custom package imports.
from sensor_models import (
        quaternion_multiply,
        quaternion_from_matrix,
        quaternion_from_euler,
        quaternion_slerp,
        quaternion_inverse,
        quaternion_matrix
        )
from colmap_processing.camera_models import load_from_file, StandardCamera
from sensor_models.nav_state import NavStateINSJson
from sensor_models.nav_conversions import enu_to_llh, llh_to_enu


# Colmap text camera model directory.
flight_dir = './00'


def get_data(image_pts_fname, gcp_fname, lat0, lon0, h0):
    image_pts0 = np.loadtxt(image_pts_fname)
    image_pts0 = np.atleast_2d(image_pts0)
    gcp = np.loadtxt(gcp_fname)

    gcp_enu_dict = {}
    gcp_llh_dict = {}
    for i in range(len(gcp)):
        gcp_enu_dict[int(gcp[i, 0])] = llh_to_enu(gcp[i][1], gcp[i][2],
                                                  gcp[i][3], lat0, lon0, h0,
                                                  in_degrees=True)
        gcp_llh_dict[int(gcp[i, 0])] = gcp[i]

    im_pts0 = []
    llh = []
    enu = []
    for i in range(len(image_pts0)):
        try:
            enu.append(gcp_enu_dict[image_pts0[i][0]])
            im_pts0.append(image_pts0[i][1:4])
            llh.append(gcp_llh_dict[image_pts0[i][0]])

        except KeyError:
            # Latitude, longitude, and altitude were not defined for this
            # point.
            pass

    return np.array(im_pts0), np.array(llh), np.array(enu)


def calibrate_camera(camera_dir, camera_model_fname, cal_dir):
    # Read in the nav binary.
    nav_state_provider = NavStateINSJson('%s/*.json' % camera_dir)

    rgb_fnames = {}
    for rgb_fname in glob.glob('%s/*rgb.tif' % camera_dir):
        try:
            with open(rgb_fname.replace('rgb.tif', 'meta.json')) as fjson:
                d = json.load(fjson)

                # Time that the image was taken.
                rgb_fnames[d['evt']['time']] = rgb_fname
        except OSError:
            pass

    ir_fnames = {}
    for ir_fname in glob.glob('%s/*ir.tif' % camera_dir):
        try:
            with open(ir_fname.replace('ir.tif', 'meta.json')) as json_file:
                d = json.load(json_file)

                # Time that the image was taken.
                ir_fnames[d['evt']['time']] = rgb_fname
        except OSError:
            pass

    camera_model = load_from_file(camera_model_fname, nav_state_provider)
    nav_state_provider = camera_model.nav_state_provider

    frame_times = set(rgb_fnames.keys()).union(set(ir_fnames.keys()))
    frame_times = sorted(list(frame_times))
    rgb_fname_to_times = {os.path.split(rgb_fnames[key])[1]: key
                          for key in rgb_fnames}

    pose = nav_state_provider.pose_time_series
    if False:
        fig = plt.figure()
        fig.add_subplot(111, projection='3d')
        plt.plot(pose[:, 1], pose[:, 2], pose[:, 3])

    lat0 = nav_state_provider.lat0
    lon0 = nav_state_provider.lon0
    h0 = nav_state_provider.h0

    gcp_fname = '%s/ground_control_points.txt' % cal_dir
    frame_times = []
    im_pts = []
    enu = []
    for image_pts_fname in glob.glob('%s/*image_points.txt' % cal_dir):
        img_fname = os.path.split(image_pts_fname)[1].replace('_image_points.txt',
                                                              '.tif')
        ret = get_data(image_pts_fname, gcp_fname, lat0, lon0, h0)
        im_pts.append(ret[0])
        enu.append(ret[2])
        frame_times.append(np.ones(len(ret[0]))*rgb_fname_to_times[img_fname])

    frame_times = np.hstack(frame_times)
    im_pts = np.vstack(im_pts)
    enu = np.vstack(enu)

    tmp_cm = copy.deepcopy(camera_model)

    def proj_err(x):
        if len(x) > 4:
            xyz = enu.copy()
            xyz[:, 2] = xyz[:, 2] + x[4:]
        else:
            xyz = enu

        cam_quat = x[:4]
        cam_quat /= np.linalg.norm(cam_quat)
        tmp_cm.update_intrinsics(cam_quat=x)
        err = 0
        L = len(im_pts)
        for i in range(L):
            d = tmp_cm.project(xyz[i], frame_times[i]).ravel() - im_pts[i]
            err += np.linalg.norm(d)

        err /= L
        return err

    min_err = np.inf
    best_x = None
    for _ in range(1000):
        x = np.random.rand(4)*2-1
        x /= np.linalg.norm(x)
        err = proj_err(x)
        if err < min_err:
            min_err = err
            best_x = x

    cam_quat = minimize(proj_err, best_x).x

    x = np.hstack([cam_quat, np.zeros(len(enu))])
    x = minimize(proj_err, x).x
    cam_quat = x[:4]
    cam_quat /= np.linalg.norm(cam_quat)

    print('Final mean error:', proj_err(x), 'pixels')
    camera_model.update_intrinsics(cam_quat=cam_quat)
    camera_model.save_to_file(camera_model_fname)
    print('Saved updated camera model to', camera_model_fname)


# Process three RGB cameras.
camera_dir = '%s/CENT' % flight_dir
camera_model_fname = 'center_sys/rgb.yaml'
cal_dir = '%s/calibration_points' % camera_dir
calibrate_camera(camera_dir, camera_model_fname, cal_dir)

camera_dir = '%s/LEFT' % flight_dir
camera_model_fname = 'left_sys/rgb.yaml'
cal_dir = '%s/calibration_points' % camera_dir
calibrate_camera(camera_dir, camera_model_fname, cal_dir)

camera_dir = '%s/RIGHT' % flight_dir
camera_model_fname = 'right_sys/rgb.yaml'
cal_dir = '%s/calibration_points' % camera_dir
calibrate_camera(camera_dir, camera_model_fname, cal_dir)

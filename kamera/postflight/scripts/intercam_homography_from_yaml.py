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
import time
import os
import copy
import glob
import random
import json
import PIL
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
from colmap_processing.image_renderer import render_view


def process(camera_model_fname1, camera_model_fname2, out_fname):
    src_cm = load_from_file(camera_model_fname1)
    dst_cm = load_from_file(camera_model_fname2)

    x = np.linspace(0, dst_cm.width-1, dst_cm.width//2)
    y = np.linspace(0, dst_cm.height-1, dst_cm.height//2)
    X,Y = np.meshgrid(x, y)
    im_pts = np.vstack([X.ravel(),Y.ravel()])

    # Unproject rays into camera coordinate system.
    _, ray_dir = dst_cm.unproject(im_pts, 0)
    points = ray_dir*1e6

    im_pts_src = src_cm.project(points, 0).astype(np.float32)

    # Remove coordinates outside of src camera.
    ind = np.logical_and(im_pts_src[0] >= 0, im_pts_src[0] <= src_cm.width)
    ind = np.logical_and(ind, im_pts_src[1] >= 0)
    ind = np.logical_and(ind, im_pts_src[1] <= src_cm.height)

    im_pts = im_pts[:, ind]
    im_pts_src = im_pts_src[:, ind]

    h, status = cv2.findHomography(im_pts_src.T, im_pts.T)

    # Error in using homography to represent transformation.
    im_pts2 = np.dot(h, np.vstack([im_pts_src, np.ones(im_pts_src.shape[1])]))
    im_pts2 = im_pts2[:2]/im_pts2[2]
    err_forward = np.sqrt(np.sum((im_pts2 - im_pts)**2, axis=0))

    im_pts2 = np.dot(np.linalg.inv(h),
                     np.vstack([im_pts, np.ones(im_pts.shape[1])]))
    im_pts2 = im_pts2[:2]/im_pts2[2]
    err_reverse = np.sqrt(np.sum((im_pts2 - im_pts_src)**2, axis=0))

    base_dir, base_fname = os.path.split(out_fname)
    base_fname = os.path.splitext(base_fname)[0]

    try:
        os.makedirs(base_dir)
    except (IOError, OSError):
        pass

    np.savetxt(out_fname, h)

    fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
    plt.rc('font', **{'size': 40})
    plt.rc('axes', linewidth=4)
    plt.subplot(121)
    plt.plot(np.linspace(0, 100, len(err_forward)), np.sort(err_forward),
             linewidth=6)
    plt.xlabel('Percentile', fontsize=50)
    plt.ylabel('Error (pixels)', fontsize=50)
    plt.title('Forward', fontsize=50)
    ax = plt.subplot(122)
    plt.plot(np.linspace(0, 100, len(err_reverse)), np.sort(err_reverse),
             linewidth=6)
    plt.xlabel('Percentile', fontsize=50)
    plt.ylabel('Error (pixels)', fontsize=50)
    plt.title('Reverse', fontsize=50)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    fig.subplots_adjust(bottom=0.13)
    fig.subplots_adjust(top=0.93)
    fig.subplots_adjust(right=0.85)
    fig.subplots_adjust(left=0.12)
    plt.savefig('%s/%s_homog_approx_error.png' % (base_dir, base_fname))


# Process all cameras.
base_dir = '/host_filesystem/mnt/homenas2/kamera/Calibration/fl08/kamera_models'
for dirname in os.listdir(base_dir):
    if not os.path.isdir('%s/%s' % (base_dir, dirname)):
        continue

    fnames = glob.glob('%s/%s/*_rgb.yaml' % (base_dir, dirname))
    if len(fnames) != 1:
        continue

    rgb_camera_model_fname = fnames[0]

    fnames = glob.glob('%s/%s/*_uv.yaml' % (base_dir, dirname))
    if len(fnames) != 1:
        continue

    uv_camera_model_fname = fnames[0]

    out_fname = '%s/%s/%s_to_%s_homography.txt' % (base_dir, dirname, 'uv', 'rgb')
    process(uv_camera_model_fname, rgb_camera_model_fname, out_fname)

    out_fname = '%s/%s/%s_to_%s_homography.txt' % (base_dir, dirname, 'rgb', 'uv')
    process(rgb_camera_model_fname, uv_camera_model_fname, out_fname)

    fnames = glob.glob('%s/%s/*_ir.yaml' % (base_dir, dirname))
    if len(fnames) != 1:
        continue

    ir_camera_model_fname = fnames[0]

    print('Processing', dirname)

    out_fname = '%s/%s/%s_to_%s_homography.txt' % (base_dir, dirname, 'ir', 'rgb')
    process(ir_camera_model_fname, rgb_camera_model_fname, out_fname)

    out_fname = '%s/%s/%s_to_%s_homography.txt' % (base_dir, dirname, 'rgb', 'ir')
    process(rgb_camera_model_fname, ir_camera_model_fname, out_fname)

#!/usr/bin/env python
"""
Library handling projection operations of a standard camera model.
"""
from __future__ import division, print_function
import copy
import cv2
import time
import os
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


def calibrate_ir(rgb_camera_model_fname, ir_camera_model_fname,
                 image_point_pairs_fname):
    image_pts = np.loadtxt(image_point_pairs_fname)

    rgb_camera = load_from_file(rgb_camera_model_fname)
    ir_camera = load_from_file(ir_camera_model_fname)

    def get_new_cm(x):
        tmp_cm = copy.deepcopy(ir_camera)
        cam_quat = x[:4]
        cam_quat /= np.linalg.norm(cam_quat)
        tmp_cm.update_intrinsics(cam_quat=cam_quat)

        if len(x) > 4:
            tmp_cm.fx = x[4]

        if len(x) > 5:
            tmp_cm.fy = x[5]

        return tmp_cm

    def proj_err(x):
        tmp_cm = get_new_cm(x)

        wrld_pts = rgb_camera.unproject(image_pts[:, 2:].T)[1]
        im_pts = tmp_cm.project(wrld_pts)

        err = np.sqrt(np.sum(np.sum((image_pts[:, :2].T - im_pts)**2, 1)))
        err /= len(wrld_pts)
        print(err, x)
        return err

    min_err = np.inf
    best_x = None
    for _ in range(10000):
        x = np.random.rand(4)*2-1
        x /= np.linalg.norm(x)
        err = proj_err(x)
        if err < min_err:
            min_err = err
            best_x = x

    if True:
        x = np.hstack([best_x, ir_camera.fx, ir_camera.fy])
    else:
        x = best_x

    x = minimize(proj_err, x).x
    x = minimize(proj_err, x, method='Powell').x
    x = minimize(proj_err, x, method='BFGS').x
    x = minimize(proj_err, x).x

    tmp_cm = get_new_cm(x)

    print('Final mean error:', proj_err(x), 'pixels')
    tmp_cm.image_topic = ''
    tmp_cm.frame_id = ''
    tmp_cm.save_to_file(ir_camera_model_fname)
    print('Saved updated camera model to', ir_camera_model_fname)


# Process three RGB cameras.
rgb_camera_model_fname = ''
ir_camera_model_fname = ''
image_point_pairs_fname = ''
calibrate_ir(rgb_camera_model_fname, ir_camera_model_fname,
             image_point_pairs_fname)
# ---------------------------------- IR Gifs ---------------------------------

# Location to save KAMERA camera models.
rgb_img_dir = ''
ir_img_dir = ''

base_dir, base = os.path.split(ir_camera_model_fname)
out_dir = '%s/registration_gifs/%s' % (base_dir, os.path.splitext(base)[0])
num_gifs = 50

try:
    os.makedirs(out_dir)
except (IOError, OSError):
    pass


def stretch_contrast(img, clip_limit=3, stretch_percentiles=[0.1, 99.9]):
    img = img.astype(np.float32)
    img -= np.percentile(img.ravel(), stretch_percentiles[0])
    img[img < 0] = 0
    img /= np.percentile(img.ravel(), stretch_percentiles[1])/255
    img[img > 255] = 255
    img = np.round(img).astype(np.uint8)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
    hls[:, :, 1] = clahe.apply(hls[:, :, 1])

    img = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    return img

cm_rgb = load_from_file(rgb_camera_model_fname)
cm_ir = load_from_file(ir_camera_model_fname)

rgb_fnames = glob.glob('%s/*.jpg' % rgb_img_dir)
random.shuffle(rgb_fnames)
k = 0


for rgb_fname in rgb_fnames[:num_gifs]:
    if k == num_gifs:
        break

    ir_fname = '%s/%sir.tif' % (ir_img_dir, os.path.split(rgb_fname[:-7])[1])
    img2 = cv2.imread(ir_fname, cv2.IMREAD_COLOR)

    if img2 is None:
        continue

    img1 = cv2.imread(rgb_fname, cv2.IMREAD_COLOR)

    if img1 is None:
        continue

    k += 1

    img1 = img1[:, :, ::-1]
    img2 = img2[:, :, ::-1]

    img1 = stretch_contrast(img1, clip_limit=3, stretch_percentiles=[0.1, 99.9])
    img2 = stretch_contrast(img2, clip_limit=3, stretch_percentiles=[0.1, 99.9])

    img3, mask = render_view(cm_rgb, img1, 0, cm_ir, 0, block_size=10)

    img2_ = PIL.Image.fromarray(img2)
    img3_ = PIL.Image.fromarray(img3)
    fname_out = '%s/registration_%i.gif' % (out_dir, k+1)
    img2_.save(fname_out, save_all=True, append_images=[img3_],
               duration=250, loop=0)

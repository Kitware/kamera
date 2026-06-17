#! /usr/bin/python
from __future__ import division, print_function
import numpy as np
import os
import cv2
import subprocess
import matplotlib.pyplot as plt
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
project_dir = '/test'
pose_filename = '%s/image_poses.txt' % project_dir

# Path to the images.bin file.
images_bin_fname = '%s/images.bin' % project_dir

images = read_images_binary(images_bin_fname)

lines = []
for image_num in images:
    image = images[image_num]
    R = qvec2rotmat(image.qvec)
    tvec = image.tvec
    pos = -np.dot(R.T, tvec)
    line = [image.name]
    line = line + list(pos.ravel()) + list(R.ravel())
    line = [str(_) for _ in line]
    lines.append(line)


with open(pose_filename, 'w') as f:
    t = '# Image Name, Camera Position, Camera Rotation Matrix (unraveled by row).'
    f.write(t)
    for line in lines:
        line = '%s\n' % ', '.join(line)
        f.write(line)

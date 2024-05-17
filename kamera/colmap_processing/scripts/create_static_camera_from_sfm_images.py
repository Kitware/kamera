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
import trimesh
import math
import PIL
from osgeo import osr, gdal

# Colmap Processing imports.
from colmap_processing.geo_conversions import llh_to_enu
from colmap_processing.colmap_interface import read_images_binary, Image, \
    read_points3D_binary, read_cameras_binary, qvec2rotmat
from colmap_processing.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
import colmap_processing.vtk_util as vtk_util
from colmap_processing.geo_conversions import enu_to_llh, rmat_ecef_enu, \
    rmat_enu_ecef
from colmap_processing.static_camera_model import save_static_camera, \
    write_camera_krtd_file


# ----------------------------------------------------------------------------
mesh_fname = 'coarse.ply'
save_dir = 'reference_views'

# Base path to the colmap directory.
image_dir = 'images'

# Image to turn into a static camera.
image_fname = '102CANON/IMG_8520.JPG'

# Path to the images.bin file.
images_bin_fname = '%s/images.bin' % image_dir
camera_bin_fname = '%s/cameras.bin' % image_dir
points_3d_bin_fname = '%s/points3D.bin' % image_dir

if True:
    #MUTC.
    H_enu = np.identity(4)
    latitude0 = 0    # degrees
    longitude0 = 0  # degrees
    altitude0 = 0            # meters above WGS84 ellipsoid

H_enu = np.array(H_enu)

# VTK renderings are limited to monitor resolution (width x height).
monitor_resolution = (2200, 1200)
# ----------------------------------------------------------------------------


# Read model into VTK.
model_reader = vtk_util.load_world_model(mesh_fname)

# Read in the details of all images.
images = read_images_binary(images_bin_fname)
cameras = read_cameras_binary(camera_bin_fname)


def process(image_fname, save_dir):
    # Consider image with filename.
    fname_to_index = {images[_].name: _ for _ in images}
    image_id = fname_to_index[image_fname]
    image = images[image_id]
    camera = cameras[image.camera_id]
    p = camera.params

    # True camera matrix and distortion of the real image.
    K0 = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
    dist = p[4:]
    width = camera.width
    height = camera.height

    # Desired camera matrix has isotropic focal length.
    K = K0.copy()
    K[0, 0] = K[1, 1] = np.min([K0[0, 0], K0[1, 1]])
    K[0, 2] = width/2
    K[1, 2] = height/2

    # Pose of the real camera.
    R = qvec2rotmat(image.qvec)
    tvec = image.tvec
    cam_pos = np.dot(-R.T, image.tvec)

    # Update camera pose into the ENU coordinate system of the 3-D model.
    cam_pos = np.dot(H_enu[:3], np.hstack([cam_pos, 1]))
    ret = cv2.decomposeProjectionMatrix(H_enu[:3])
    R_enu = ret[1]
    R = np.dot(R, R_enu.T)

    # Define the rendering camera required to capture all of the image but not
    # exceeding monitor resolution.
    K_ = K.copy()
    if (width <= monitor_resolution[0] and height <= monitor_resolution[1] and
        np.all(dist == 0) and K_[0, 2] == width/2 and K_[0, 2] == height/2):
        needs_warping = False
        res_x, res_y = width, height
    else:
        # Need to find the undistorted camera model that circumscribes the
        # field of view of the real camera.
        K_[0, 2] = monitor_resolution[0]/2
        K_[1, 2] = monitor_resolution[1]/2
        K_[0, 0] = K_[1, 1] = np.min([K_[0, 0], K_[1, 1]])

        # Unproject points along 'camera_model' image border and project them
        # into the temporary camera defined above to determine how much it must
        # be expanded to avoid clipping.
        points = np.array([[0, width, width, 0], [0, 0, height, height]],
                          dtype=np.float32)
        ray_dir = cv2.undistortPoints(np.expand_dims(points.T, 0), K, dist,
                                      None)
        ray_dir = np.squeeze(ray_dir, 0).astype(np.float32).T
        ray_dir = np.vstack([ray_dir, np.ones(ray_dir.shape[1])])
        points2 = cv2.projectPoints(ray_dir.T, np.zeros(3, dtype=np.float32),
                                    np.zeros(3, dtype=np.float32), K_, None)[0]
        points2 = np.squeeze(points2, 1).T
        points2 = np.abs(points2 - np.atleast_2d(K_[:2, 2]).T)
        points3 = np.array([[0, width/2, width, width, width, width/2, 0, 0],
                            [0, 0, 0, height/2, height, height, height,
                             height/2]], dtype=np.float32)
        points3 = np.abs(points3 - np.atleast_2d(K_[:2, 2]).T)
        s = np.min(np.min(points3, 1) / np.max(points2, 1))
        K_[0, 0] = K_[1, 1] = K_[1, 1]*s
        needs_warping = True
        res_x, res_y = monitor_resolution

    vfov = np.arctan(res_y/2/K_[1, 1])*2*180/np.pi
    vtk_camera = vtk_util.Camera(res_x, res_y, vfov, cam_pos, R)

    clipping_range = [1, 2000]

    img = vtk_camera.render_image(model_reader, clipping_range=clipping_range,
                                  diffuse=0.6, ambient=0.6, specular=0.1,
                                  light_color=[1.0, 1.0, 1.0],
                                  light_pos=[0,0,1000])
    depth = vtk_camera.unproject_view(model_reader,
                                      clipping_range=clipping_range)[3]

    #img = cv2.resize(img, (width, height))
    #depth = cv2.resize(depth, (width, height))

    # Warp the rendered imagery.
    X, Y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    points = np.vstack([X.ravel(), Y.ravel()])
    ray_dir = cv2.undistortPoints(np.expand_dims(points.T, 0), K, None, None)
    ray_dir = np.squeeze(ray_dir, 0).astype(np.float32).T
    ray_dir = np.vstack([ray_dir, np.ones(ray_dir.shape[1])])
    points2 = cv2.projectPoints(ray_dir.T, np.zeros(3, dtype=np.float32),
                                np.zeros(3, dtype=np.float32), K_, None)[0]
    points2 = np.squeeze(points2, 1).T
    X2 = np.reshape(points2[0], X.shape).astype(np.float32)
    Y2 = np.reshape(points2[1], Y.shape).astype(np.float32)
    img = cv2.remap(img, X2, Y2, cv2.INTER_CUBIC)
    #E = cv2.remap(E, X2, Y2, cv2.INTER_CUBIC)
    #N = cv2.remap(N, X2, Y2, cv2.INTER_CUBIC)
    #U = cv2.remap(U, X2, Y2, cv2.INTER_CUBIC)
    depth = cv2.remap(depth, X2, Y2, cv2.INTER_CUBIC)

    # Read and undistort image.
    real_image = cv2.imread('%s/%s' % (image_dir, image_fname))[:, :, ::-1]

    if real_image.shape[0] == width:
        real_image = np.rot90(real_image, k=3).copy()

    cv2.imwrite('%s/ref_image_distored.png' % save_dir, real_image[:, :, ::-1])

    # Undistort real imagery.
    X, Y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    points = np.vstack([X.ravel(), Y.ravel()])
    ray_dir = cv2.undistortPoints(np.expand_dims(points.T, 0), K, None, None)
    ray_dir = np.squeeze(ray_dir, 0).astype(np.float32).T
    ray_dir = np.vstack([ray_dir, np.ones(ray_dir.shape[1])])
    points2 = cv2.projectPoints(ray_dir.T, np.zeros(3, dtype=np.float32),
                                np.zeros(3, dtype=np.float32), K0, dist)[0]
    points2 = np.squeeze(points2, 1).T
    X2 = np.reshape(points2[0], X.shape).astype(np.float32)
    Y2 = np.reshape(points2[1], Y.shape).astype(np.float32)
    real_image = cv2.remap(real_image, X2, Y2, cv2.INTER_CUBIC)

    #real_image = cv2.undistort(real_image, K0, dist, K)
    #plt.figure(); plt.imshow(real_image)

    try:
        os.makedirs(save_dir)
    except OSError:
        pass

    cv2.imwrite('%s/ref_image.png' % save_dir, real_image[:, :, ::-1])


    # -------------------------------------------------------------------
    # Identify holes in the model and then inpaint them.
    hole_mask = depth > clipping_range[-1] - 0.1

    output = cv2.connectedComponentsWithStats(hole_mask.astype(np.uint8),
                                              8, cv2.CV_32S)
    labels = output[1]

    # Remove components that touch outer boundary.
    edge_labels = set(labels[:, 0])
    edge_labels = edge_labels.union(set(labels[0, :]))
    edge_labels = edge_labels.union(set(labels[:, -1]))
    edge_labels = edge_labels.union(set(labels[-1, :]))

    for i in edge_labels:
        labels[labels == i] = 0

    mask = (labels > 0).astype(np.uint8)
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    depth = cv2.inpaint(depth.astype(np.float32), mask, 3, cv2.INPAINT_NS)
    cv2.imwrite('%s/ref_image_rendered.png' % save_dir, img[:, :, ::-1])
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Save visualization of depth map.
    depth_map_rgb = depth.copy()
    depth_map_rgb -= np.percentile(depth_map_rgb.ravel(), 0.1)
    depth_map_rgb[depth_map_rgb < 0] = 0
    r = depth_map_rgb.ravel()
    r = r[r < 900]
    depth_map_rgb /= np.percentile(r, 99.9)/255
    depth_map_rgb[depth_map_rgb > 255] = 255
    depth_map_rgb = 255 - depth_map_rgb
    depth_map_rgb = np.round(depth_map_rgb).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5,5))
    depth_map_rgb = clahe.apply(depth_map_rgb)
    depth_map_rgb = cv2.applyColorMap(depth_map_rgb, cv2.COLORMAP_JET)

    cv2.imwrite('%s/depth_map_visualization.jpg' % save_dir,
                depth_map_rgb[:, :, ::-1])
    # -------------------------------------------------------------------


    latitude, longitude, altitude = enu_to_llh(cam_pos[0], cam_pos[1],
                                               cam_pos[2], latitude0,
                                               longitude0, altitude0)

    # R currently is relative to ENU coordinate system at latitude0, longitude0,
    # altitude0, but we need it to be relative to latitude, longitude, altitude.
    Rp = np.dot(rmat_enu_ecef(latitude0, longitude0),
                rmat_ecef_enu(latitude, longitude))

    filename = '%s/camera_model.yaml' % save_dir
    dist = [0.0, 0.0, 0.0, 0.0]
    save_static_camera(filename, height, width, K, dist, np.dot(R, Rp), depth,
                       latitude, longitude, altitude)

    filename = '%s/camera_model.krtd' % save_dir
    write_camera_krtd_file([K, R, tvec, dist], filename)


image_fnames = [images[_].name for _ in images]
for key in images:
    image = images[key]
    save_dir_ = '%s/%s' % (save_dir, image.id)
    process(image.name, save_dir_)

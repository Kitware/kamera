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

Note: the image coordiante system has its origin at the center of the top left
pixel.

"""
from __future__ import division, print_function
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import time
import os
import glob
import matplotlib.pyplot as plt
import bisect
import json
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
from sensor_models.nav_state import NavStateINSBinary, NavStateINSJson


class ColmapImage(object):
    def __init__(self, image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name,
                 pts):
        self.image_id = image_id
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.cam_id = cam_id
        self.name = name
        self.pts = pts

    def get_camera_pose(self):
        R = quaternion_matrix([self.qx,self.qy,self.qz,self.qw])[:3,:3]
        return np.hstack([R,np.array([[self.tx,self.ty,self.tz]]).T])

    def pos(self):
        return np.dot(R.T,-t)


# Colmap text camera model directory.
flight_dir = '00'
colmap_dir = '00/colmap'
camera_model_dir = '/root/kamera/src/cfg/camera_models'

# Read in the nav binary.
#nav_state_provider = NavStateINSBinary(nav_binary_fname)

# Recover the mapping between filename and high-precision time.
image_globs = ['%s/CENT/*rgb.tif' % flight_dir,
               '%s/LEFT/*rgb.tif' % flight_dir,
               '%s/RIGHT/*rgb.tif' % flight_dir]
camera_model_fnames = ['%s/left_sys/rgb.yaml' % camera_model_dir,
                       '%s/center_sys/rgb.yaml' % camera_model_dir,
                       '%s/right_sys/rgb.yaml' % camera_model_dir]

img_fnames = {}
fname_to_time = {}
camera_models = []
for i in range(3):
    image_glob = image_globs[i]
    nav_dir = os.path.split(image_glob)[0]
    nav_state_provider = NavStateINSJson('%s/*meta.json' % nav_dir)

    for img_fname in glob.glob(image_glob):
        json_fname = img_fname
        json_fname = json_fname.replace('rgb.tif', 'meta.json')
        json_fname = json_fname.replace('ir.tif', 'meta.json')
        json_fname = json_fname.replace('uv.tif', 'meta.json')

        try:
            with open(json_fname) as json_file:
                d = json.load(json_file)

                # Time that the image was taken.
                img_fnames[d['evt']['time']] = img_fname
                fname = os.path.splitext(os.path.split(img_fname)[1])[0]
                fname_to_time[fname] = d['evt']['time']
        except OSError:
            pass

    lat0 = nav_state_provider.lat0
    lon0 = nav_state_provider.lon0
    h0 = nav_state_provider.h0

    camera_models.append(load_from_file(camera_model_fnames[i],
                                        nav_state_provider))



if False:
    camera_model_fnames = '%s/cameras.txt' % colmap_dir
    camera_models = []
    with open(camera_model_fnames, 'r') as infile:
        for line in infile:
            if not line.startswith('#'):
                p = np.array(line.split('\n')[0].split(' ')[2:], np.float)
                width, height, fx, fy, cx, cy, k1, k2, k3, k4 = p
                K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
                image_topic = None
                camera_models.append(StandardCamera(width, height, K,
                                                    (k1,k2,k3,k4), (0,0,0),
                                                    (1,0,0,0), image_topic,
                                                    frame_id=None,
                                                    nav_state_provider=nav_state_provider))


image_fname = '%s/output/images.txt' % colmap_dir
colmap_images = []
with open(image_fname, 'r') as infile:
    while True:
        line = infile.readline()
        if line == '':
            break

        if not line.startswith('#'):
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            #   POINTS2D[] as (X, Y, POINT3D_ID)
            p = line.split('\n')[0].split(' ')
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name = p
            qw, qx, qy, qz, tx, ty, tz = [float(_) for _ in (qw, qx, qy, qz, tx, ty, tz)]
            cam_id = int(cam_id)

            line = infile.readline()
            p = line.split('\n')[0].split(' ')
            pts = np.reshape(np.array([float(_) for _ in p]), (-1,3))

            colmap_image = ColmapImage(image_id, qw, qx, qy, qz, tx, ty, tz,
                                       cam_id, name, pts)
            colmap_images.append(colmap_image)


points_fname = '%s/points3D.txt' % colmap_dir
points = []
with open(points_fname, 'r') as infile:
    for line in infile:
        pass


nav_times = nav_state_provider.pose_time_series[:,0]


camera_model = camera_models[0]


def err_fun(cam_quat):
    #cam_quat = np.hstack([0.5, cam_quat])
    cam_quat /= np.linalg.norm(cam_quat)
    camera_model.update_intrinsics(cam_quat=cam_quat)
    theta = 0
    for i in range(len(colmap_images)):
        colmap_image = colmap_images[i]
        fname = os.path.splitext(os.path.split(colmap_image.name)[1])[0]
        t = fname_to_time[fname]
        P1 = camera_models[0].get_camera_pose(t)
        P2 = colmap_image.get_camera_pose()
        R1 = np.identity(4);    R1[:3,:3] = P1[:,:3]
        R2 = np.identity(4);    R2[:3,:3] = P2[:,:3]
        q1 = quaternion_from_matrix(R1)
        q2 = quaternion_from_matrix(R2)
        dq = quaternion_multiply(q1, quaternion_inverse(q2))
        theta += 2*np.arccos(max([min([dq[3],1]),-1]))

    theta /= len(colmap_images)
    print(cam_quat, theta)
    return theta

if False:
    min_err = np.inf
    for _ in range(10000):
        cam_quat = random_quaternion()
        err = err_fun(cam_quat)
        if err < min_err:
            min_err = err
            best_cam_quat = cam_quat
    else:
        cam_quat = camera_model.cam_quat

x = minimize(err_fun, cam_quat, tol=1-9).x


plt.plot(nav_state_provider.pose_time_series[:,3])

# ----------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = 5
times = nav_times
times = [image_times[_.name] for _ in colmap_images]
times = np.sort(times)
pos = np.array([nav_state_provider.pose(t)[0] for t in times]).T

plt.plot(pos[0], pos[1], pos[2], 'k-')
plt.plot(pos[0], pos[1], pos[2], 'ro')

for t in times:
    pos,quat = nav_state_provider.pose(t)
    R = quaternion_matrix(quaternion_inverse(quat))[:3,:3]

    s = ['r-','g-','b-']
    for i in range(3):
        plt.plot([pos[0],pos[0]+R[i][0]*r], [pos[1],pos[1]+R[i][1]*r],
                 [pos[2],pos[2]+R[i][2]*r], s[i], linewidth=3)

plt.xlabel('Easting')
plt.ylabel('Northing')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = 5
for i in range(len(colmap_images)):
    colmap_image = colmap_images[i]
    P = colmap_image.get_camera_pose()
    R = np.identity(4);    R[:3,:3] = P[:,:3]
    pos = colmap_image.pos()
    plt.plot([pos[0]], [pos[1]], [pos[2]], 'ro')

    s = ['r-','g-','b-']
    for i in range(3):
        plt.plot([pos[0],pos[0]+R[i][0]*r], [pos[1],pos[1]+R[i][1]*r],
                 [pos[2],pos[2]+R[i][2]*r], s[i], linewidth=3)

plt.xlabel('Easting')
plt.ylabel('Northing')

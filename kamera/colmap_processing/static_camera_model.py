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
import os, cv2, yaml, PIL
from PIL import Image


def to_str(v):
    """Convert numerical values (scalar or float) to string for saving to yaml

    """
    if hasattr(v,  "__len__"):
        if len(v) > 1:
            return repr(list(v))
        else:
            v = v[0]

    return repr(v)


def load_static_camera_from_file(filename):
    with open(filename, 'r') as f:
        calib = yaml.load(f)

        assert calib['model_type'] == 'static'

        # fill in CameraInfo fields
        width = calib['image_width']
        height = calib['image_height']
        dist = np.array(calib['distortion_coefficients'], dtype=np.float64)

        if isinstance(dist, str) and dist == 'None':
            dist = np.zeros(4, dtype=np.float64)

        fx = calib['fx']
        fy = calib['fy']
        cx = calib['cx']
        cy = calib['cy']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        R = np.reshape(np.array(calib['R']), (3, 3))
        latitude = calib['latitude']
        longitude = calib['longitude']
        altitude = calib['altitude']
#         image_topic = calib['image_topic']
#         frame_id = calib['frame_id']

        depth_map_fname = '%s_depth_map.tif' % os.path.splitext(filename)[0]
        try:
            depth_map = np.asarray(PIL.Image.open(depth_map_fname))
        except OSError:
            depth_map = None

    return height, width, K, dist, R, depth_map, latitude, longitude, altitude


def save_static_camera(filename, height, width, K, dist, R, depth_map,
                       latitude, longitude, altitude):
    dist = np.array(dist, dtype=np.float32).ravel()

    with open(filename, 'w') as f:
        f.write(''.join(['# The type of camera model.\n',
                         'model_type: static\n\n',
                         '# Image dimensions\n']))

        f.write(''.join(['image_width: ', to_str(width), '\n']))
        f.write(''.join(['image_height: ', to_str(height), '\n\n']))

        f.write('# Focal length along the image\'s x-axis.\n')
        f.write(''.join(['fx: ', to_str(K[0, 0]), '\n\n']))

        f.write('# Focal length along the image\'s y-axis.\n')
        f.write(''.join(['fy: ', to_str(K[1, 1]), '\n\n']))

        f.write('# Principal point is located at (cx,cy).\n')
        f.write(''.join(['cx: ', to_str(K[0, 2]), '\n']))
        f.write(''.join(['cy: ', to_str(K[1, 2]), '\n\n']))

        f.write(''.join(['# Distortion coefficients following OpenCv\'s ',
                'convention\n']))
        f.write(''.join(['distortion_coefficients: ',
                         to_str(dist), '\n\n']))

        f.write(''.join(['# Rotation matrix mapping vectors defined in an '
                         'east/north/up coordinate system\n# centered at '
                         'the camera into vectors defined in the camera'
                         'coordinate system.\n',
                         'R: [%0.10f, %0.10f, %0.10f,\n'
                         '    %0.10f, %0.10f, %0.10f,\n'
                         '    %0.10f, %0.10f, %0.10f]' %
                         tuple(R.ravel()), '\n\n']))

        f.write(''.join(['# Location of the camera\'s center of '
                         'projection. Latitude and longitude are in\n# '
                         'degrees, and altitude is meters above the WGS84 '
                         'ellipsoid.\n',
                         'latitude: %0.10f\n' % latitude,
                         'longitude: %0.10f\n' % longitude,
                         'altitude: %0.10f' % altitude,'\n\n']))

        f.write('# Topic on which this camera\'s image is published\n')
        f.write(''.join(['image_topic: \n\n']))

        f.write('# The frame_id embedded in the published image header.\n')
        f.write(''.join(['frame_id: ']))

    if depth_map is not None:
        im = PIL.Image.fromarray(depth_map, mode='F')  # float32
        depth_map_fname = '%s_depth_map.tif' % os.path.splitext(filename)[0]
        im.save(depth_map_fname)


def write_camera_krtd(camera, fout):
    """Write a single camera in ASCII KRTD format to the file object.
    """
    K, R, t, d = camera
    t = np.reshape(np.array(t), 3)
    fout.write('%.12g %.12g %.12g\n' % tuple(K.tolist()[0]))
    fout.write('%.12g %.12g %.12g\n' % tuple(K.tolist()[1]))
    fout.write('%.12g %.12g %.12g\n\n' % tuple(K.tolist()[2]))
    fout.write('%.12g %.12g %.12g\n' % tuple(R.tolist()[0]))
    fout.write('%.12g %.12g %.12g\n' % tuple(R.tolist()[1]))
    fout.write('%.12g %.12g %.12g\n\n' % tuple(R.tolist()[2]))
    fout.write('%.12g %.12g %.12g\n\n' % tuple(t.tolist()))
    for v in d:
        fout.write('%.12g ' % v)


def write_camera_krtd_file(camera, filename):
    """Write a camera to a krtd file
    """
    with open(filename,'w') as f:
        write_camera_krtd(camera, f)


def unproject_from_camera(im_pts, K, dist, R, cam_pos, depth_map):
    # Unproject rays into the camera coordinate system.
    ray_dir = np.ones((3, len(im_pts)), dtype=np.float)
    ray_dir0 = cv2.undistortPoints(np.expand_dims(im_pts, 0), K, dist, R=None)
    ray_dir[:2] = np.squeeze(ray_dir0, 0).T

    # We want the z-coordinate of the ray direction to be 1.

    enu0 = np.array(cam_pos, copy=True)

    # Rotate rays into the local east/north/up coordinate system.
    ray_dir = np.dot(R.T, ray_dir)

    height, width = depth_map.shape
    enu = np.zeros((len(im_pts), 3))
    for i in range(im_pts.shape[0]):
        x, y = im_pts[i]
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

    return enu.T


def project_to_camera(wrld_pts, K, dist, R, cam_pos):
    # Unproject rays into the camera coordinate system.
    tvec = -np.dot(R, cam_pos).ravel()
    rvec = cv2.Rodrigues(R)[0]
    im_pts = cv2.projectPoints(wrld_pts.T, rvec, tvec, K, dist)[0]
    im_pts = np.squeeze(im_pts)

    return im_pts

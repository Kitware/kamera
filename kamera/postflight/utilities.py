#!/usr/bin/env python3
"""
ckwg +31
Copyright 2021 by Kitware, Inc.
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
import os
from errno import EEXIST
import sys
from typing import Callable, Any, Dict, Tuple
import json
import time
import glob
import warnings
import threading
from shutil import copyfile
import exifread
import csv
from PIL import Image
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from datetime import datetime

import numpy as np
import cv2

import pygeodesy
from osgeo import osr, gdal
import simplekml
from shapely.geometry import Polygon, mapping
import shapefile

# Custom package imports.
import sys
sys.path.insert(0,'C:/Users/path_to/postflight_scripts/sensor_models/src')
from kamera.sensor_models import (
        quaternion_multiply,
        quaternion_from_matrix,
        quaternion_from_euler,
        quaternion_slerp,
        quaternion_inverse,
        quaternion_matrix
        )
from kamera.colmap_processing.camera_models import load_from_file
from kamera.sensor_models.nav_conversions import enu_to_llh, llh_to_enu
from kamera.sensor_models.nav_state import NavStateINSJson

# Adjust as necessary
GEOD_FNAME = "/src/kamera/kamera/assets/geods/egm84-15.pgm"

SUFFIX_RGB = 'rgb.jpg'
SUFFIX_IR = 'ir.tif'
SUFFIX_UV = 'uv.jpg'
SUFFIX_META = 'meta.json'

SUFFIX_DICT = {'rgb': SUFFIX_RGB, 'ir': SUFFIX_IR, 'uv': SUFFIX_UV}

# === === === === === === === one-off functions === === === === === === ===

class FileNotFoundError(IOError):
    def __init__(self, path, kind='file'):

        # Call the base class constructor with the parameters it needs
        super(FileNotFoundError, self).__init__("{} does not exist: '{}'".format(kind, path))

def get_subdirs(path):
    return next(os.walk(path))[1]

def make_path(path, from_file=False, verbose=False):
    """
    Make a path, ignoring already-exists error. Python 2/3 compliant.
    Catch any errors generated, and skip it if it's EEXIST.
    :param path: Path to create
    :type path: str, pathlib.Path
    :param from_file: if true, treat path as a file path and create the basedir
    :return:
    """
    path = str(path)  # coerce pathlib.Path
    if path == '':
        raise ValueError("Path is empty string, cannot make dir.")

    if from_file:
        path = os.path.dirname(path)
    try:
        os.makedirs(path)
        if verbose:
            print('Created path: {}'.format(path))
    except OSError as exception:
        if exception.errno != EEXIST:
            raise
        if verbose:
            print('Tried to create path, but exists: {}'.format(path))

FOV_CENTER = 'center'
FOV_LEFT = 'left'
FOV_RIGHT = 'right'

fov_map = {
    'center': FOV_CENTER,
    'left': FOV_LEFT,
    'right': FOV_RIGHT,
    'cent': FOV_CENTER,
    'rght': FOV_RIGHT
 }

fov_map.update({k.upper(): v for k, v in fov_map.items()})

def first_wordlike(name):
    return name.replace('-', '_').split('_')[0]


def reduce_fov(name, noisy=True):
    """Remap fov dir terms to standard ones. If noisy, throw error on failure"""
    fov = first_wordlike(name)
    res = fov_map.get(fov, None)
    if fov is None and noisy:
        raise KeyError("Could not remap FOV dir name: '{}'".format(name))
    return res


def get_dir_type(dpath):
    dpath = os.path.abspath(dpath)
    if os.path.isfile(dpath):
        return 'file'
    if not os.path.isdir(dpath):
        return 'special'
    subfiles = os.listdir(dpath)
    if 'sys_config.json' in subfiles:
        return 'sys_config'
    fn_logs = [el for el in subfiles if '_log.txt' in el]
    for fn_log in fn_logs:
        if os.path.basename(dpath) in os.path.basename(fn_log):
            return 'flight'
    return 'unknown'


def get_fov_dirs(flight_dir):
    """Return the actual FOV direcotry
    This actually expects a sys_cfg dir
    """
    #subdirs = get_subdirs(flight_dir)
    #actual_dirs = [name for name in subdirs if reduce_fov(name, noisy=False)]

    # These are the acceptable options for camera directories that we want to
    # consider.
    if not os.path.isdir(flight_dir):
        raise OSError("Called get_fov_dirs on a non-directory: {}".format(flight_dir))
    dtype = get_dir_type(flight_dir)
    if dtype != 'sys_config':
        raise RuntimeError("Called get_fov_dirs on {} directory: {}".format(dtype, flight_dir))

    acceptable_names = ['cent','center', 'left', 'right', 'center_view',
                        'left_view', 'right_view']

    actual_dirs = []

    for subdir in os.listdir(flight_dir):
        full_path = '%s/%s' % (flight_dir, subdir)

        if not os.path.isdir(full_path):
            continue

        if os.path.isdir(full_path) and subdir.lower() in acceptable_names:
            actual_dirs.append(subdir)

    return actual_dirs


def get_active_fovs(flight_dir):
    fov_dirs = get_fov_dirs(flight_dir)
    return [first_wordlike(n) for n in fov_dirs]


def dir_assert(dirname, kind='dir'):
    if not os.path.isdir(dirname):
        raise FileNotFoundError(dirname, kind=kind)


def file_assert(filename, kind='file'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename, kind=kind)




# === === === === === === === one-off functions === === === === === === ===


def update_progress(num, total, flight_dir, skip_n=10, msg=None, name='progress.jsonl'):
    if num % skip_n:
        return
    key = os.path.join(flight_dir, name)
    data = {"time": time.time(), "num": num, "total": total, "msg": msg}
    js = json.dumps(data) + '\n'
    with open(key, 'w') as fp:
        fp.write(js)


wgs84_cs = osr.SpatialReference()
wgs84_cs.SetWellKnownGeogCS("WGS84")
wgs84_wkt = wgs84_cs.ExportToWkt()


# Bayer pattern dictionary.
bayer_patterns = {}
bayer_patterns['bayer_rggb8'] = cv2.COLOR_BayerBG2RGB
bayer_patterns['bayer_grbg8'] = cv2.COLOR_BayerGB2RGB
bayer_patterns['bayer_bggr8'] = cv2.COLOR_BayerRG2RGB
bayer_patterns['bayer_gbrg8'] = cv2.COLOR_BayerGR2RGB
bayer_patterns['bayer_rggb16'] = cv2.COLOR_BayerBG2RGB
bayer_patterns['bayer_grbg16'] = cv2.COLOR_BayerGB2RGB
bayer_patterns['bayer_bggr16'] = cv2.COLOR_BayerRG2RGB
bayer_patterns['bayer_gbrg16'] = cv2.COLOR_BayerGR2RGB


kml_color_map = {'left_rgb': simplekml.Color.green,
                 'center_rgb': simplekml.Color.green,
                 'right_rgb': simplekml.Color.green,
                 'left_ir': simplekml.Color.red,
                 'center_ir': simplekml.Color.red,
                 'right_ir': simplekml.Color.red,
                 'left_uv': simplekml.Color.yellow,
                 'center_uv': simplekml.Color.yellow,
                 'right_uv': simplekml.Color.yellow}


def print2(*args):
    """Print and then flush stdout so that multi-threaded logging works.

    """
    print(*args)
    sys.stdout.flush()


class Detection(object):
    __slots__ = ['uid', 'image_fname', 'frame_id', 'image_bbox', 'lonlat_bbox',
                 'confidence', 'length', 'confidence_pairs', 'gsd',
                 'height_meters', 'width_meters', 'suppressed']

    def __init__(self, uid, image_fname, frame_id, image_bbox, lonlat_bbox,
                 confidence, length, confidence_pairs, gsd, height_meters,
                 width_meters, suppressed):
        self.uid = uid
        self.image_fname = image_fname
        self.frame_id = frame_id
        self.image_bbox = image_bbox
        self.lonlat_bbox = lonlat_bbox
        self.confidence = confidence
        self.length = length
        self.confidence_pairs = confidence_pairs
        self.gsd = gsd
        self.height_meters = height_meters
        self.width_meters = width_meters
        self.suppressed = suppressed


def decompose_affine(A):
    '''Decompose homogenous affine transformation matrix `A` into parts.

    The transforms3d package, including all examples, code snippets and
    attached documentation is covered by the 2-clause BSD license.

    Copyright (c) 2009-2017, Matthew Brett and Christoph Gohlke
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    The parts are translations, rotations, zooms, shears.
    `A` can be any square matrix, but is typically shape (4,4).
    Decomposes A into ``T, R, Z, S``, such that, if A is shape (4,4)::
       Smat = np.array([[1, S[0], S[1]],
                        [0,    1, S[2]],
                        [0,    0,    1]])
       RZS = np.dot(R, np.dot(np.diag(Z), Smat))
       A = np.eye(4)
       A[:3,:3] = RZS
       A[:-1,-1] = T
    The order of transformations is therefore shears, followed by
    zooms, followed by rotations, followed by translations.
    The case above (A.shape == (4,4)) is the most common, and
    corresponds to a 3D affine, but in fact A need only be square.
    Parameters
    ----------
    A : array shape (N,N)
    Returns
    -------
    T : array, shape (N-1,)
       Translation vector
    R : array shape (N-1, N-1)
        rotation matrix
    Z : array, shape (N-1,)
       Zoom vector.  May have one negative zoom to prevent need for negative
       determinant R matrix above
    S : array, shape (P,)
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix.  P is the (N-2)th Triangular
       number, which happens to be 3 for a 4x4 affine.
    Examples
    --------
    >>> T = [20, 30, 40] # translations
    >>> R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
    >>> Z = [2.0, 3.0, 4.0] # zooms
    >>> S = [0.2, 0.1, 0.3] # shears
    >>> # Now we make an affine matrix
    >>> A = np.eye(4)
    >>> Smat = np.array([[1, S[0], S[1]],
    ...                  [0,    1, S[2]],
    ...                  [0,    0,    1]])
    >>> RZS = np.dot(R, np.dot(np.diag(Z), Smat))
    >>> A[:3,:3] = RZS
    >>> A[:-1,-1] = T # set translations
    >>> Tdash, Rdash, Zdash, Sdash = decompose(A)
    >>> np.allclose(T, Tdash)
    True
    >>> np.allclose(R, Rdash)
    True
    >>> np.allclose(Z, Zdash)
    True
    >>> np.allclose(S, Sdash)
    True
    Notes
    -----
    We have used a nice trick from SPM to get the shears.  Let us call the
    starting N-1 by N-1 matrix ``RZS``, because it is the composition of the
    rotations on the zooms on the shears.  The rotation matrix ``R`` must have
    the property ``np.dot(R.T, R) == np.eye(N-1)``.  Thus ``np.dot(RZS.T,
    RZS)`` will, by the transpose rules, be equal to ``np.dot((ZS).T, (ZS))``.
    Because we are doing shears with the upper right part of the matrix, that
    means that the Cholesky decomposition of ``np.dot(RZS.T, RZS)`` will give
    us our ``ZS`` matrix, from which we take the zooms from the diagonal, and
    the shear values from the off-diagonal elements.
    '''
    A = np.asarray(A)
    T = A[:-1,-1]
    RZS = A[:-1,:-1]
    ZS = np.linalg.cholesky(np.dot(RZS.T,RZS)).T
    Z = np.diag(ZS).copy()
    shears = ZS / Z[:,np.newaxis]
    n = len(Z)
    S = shears[np.triu(np.ones((n,n)), 1).astype(bool)]
    R = np.dot(RZS, np.linalg.inv(ZS))
    if np.linalg.det(R) < 0:
        Z[0] *= -1
        ZS[0] *= -1
        R = np.dot(RZS, np.linalg.inv(ZS))
    return T, R, Z, S


def get_image_chip(image, left, right, top, bottom):
    l = np.maximum(left, 0)
    r = np.maximum(l, right)
    r = np.minimum(r, image.shape[1])
    t = np.maximum(top, 0)
    b = np.maximum(t, bottom)
    b = np.minimum(b, image.shape[0])

    if image.ndim == 3:
        return image[t:b, l:r, :]
    else:
        return image[t:b, l:r]


def points_along_image_border(width, height, num_points=4):
    """Return uniform array of points along the perimeter of the image.

    :param num_points: Number of points (approximately) to distribute
        around the perimeter of the image perimeter.
    :type num_points: int

    :return:
    :rtype: numpy.ndarry with shape (3,n)

    """
    perimeter = 2*(height + width)
    ds = num_points/float(perimeter)
    xn = np.max([2, int(ds*width)])
    yn = np.max([2, int(ds*height)])
    x = np.linspace(0, width, xn)
    y = np.linspace(0, height, yn)[1:-1]
    pts = np.vstack([np.hstack([x,
                                np.full(len(y), width,
                                        dtype=np.float64),
                                x[::-1],
                                np.zeros(len(y))]),
                     np.hstack([np.zeros(xn),
                                y,
                                np.full(xn, height,
                                        dtype=np.float64),
                                y[::-1]])])
    return pts


def meters_per_lon_lat(lon, lat):
    dtheta = 1e-5   # degrees
    meters_per_lat_deg = llh_to_enu(lat + dtheta, lon, 0, lat, lon, 0)[1]
    meters_per_lat_deg /= dtheta

    meters_per_lon_deg = llh_to_enu(lat, lon + dtheta, 0, lat, lon, 0)[0]
    meters_per_lon_deg /= dtheta

    return meters_per_lon_deg, meters_per_lat_deg


def match_image_to_json(img_fname):
    if img_fname[-7:] == 'rgb.tif':
        # Older RGB imagery was .tif.
        return img_fname.replace('rgb.tif', SUFFIX_META)

    json_fname = img_fname.replace(SUFFIX_RGB, SUFFIX_META) \
        .replace(SUFFIX_IR, SUFFIX_META)\
        .replace(SUFFIX_UV, SUFFIX_META)

    if not os.path.isfile(json_fname):
        warnings.warn('Could not find {} for {}'.format(SUFFIX_META, img_fname))
        return ''
    return json_fname


def parse_image_directory(image_dir, modality=None):
    # type: (str, str) -> Tuple[Dict, Dict, NavStateINSJson]
    """Parse imagery and metadata for subsystem image directory.

    """
    # Read in the nav binary.
    platform_pose_provider = NavStateINSJson('%s/*meta.json' % image_dir)

    img_fname_to_time = {}
    img_time_to_fname = {}
    image_files = []
    effort_type = {}
    trigger_type = {}

    print2('parse_image_directory({},\n  {})'.format(image_dir, modality))

    if modality is not None:
        suffixes = [SUFFIX_DICT[modality]]
    else:
        suffixes = [SUFFIX_RGB, SUFFIX_IR, SUFFIX_UV]

    for suffix in suffixes:
        globs = glob.glob('{}/*{}'.format(image_dir, suffix))

        if len(globs) == 0 and suffix == 'rgb.jpg':
            # Older RGB data was saved as tiff.
            globs = glob.glob('{}/*{}'.format(image_dir, 'rgb.tif'))

        print2('Found {} images of {}'.format(len(globs), suffix))
        image_files += globs

    for img_fname in image_files:
        json_fname = match_image_to_json(img_fname)

        try:
            with open(json_fname) as json_file:
                d = json.load(json_file)

                # Time that the image was taken.
                img_time_to_fname[d['evt']['time']] = img_fname
                img_fname_to_time[img_fname] = d['evt']['time']

                # Pull the effort type if available.
                try:
                    effort_type[img_fname] = str(d['effort'])
                except KeyError:
                    pass

                # Pull the trigger type if available.
                try:
                    trigger_type[img_fname] = str(d['collection_mode'])
                except KeyError:
                    pass
        except (OSError, IOError):
            pass

    return [img_fname_to_time, img_time_to_fname, platform_pose_provider,
            effort_type, trigger_type]


def get_image_boundary(camera_model, frame_time, geod_filename=GEOD_FNAME):
    """Return image coordinates and (latitude, longitude) for image border.

    This assumes the ground is located at mean sea level.

    """
    file_assert(geod_filename)
    geod = pygeodesy.geoids.GeoidPGM(geod_filename)

    im_pts = np.array([[0, 0], [camera_model.width, 0],
                       [camera_model.width, camera_model.height],
                       [0, camera_model.height]])

    # platform_pose_provider currently uses an assumption that we don't move too
    # far away from the origin, which is obviously violated here. Currently,
    # ypr is relative to the aircraft at its current state, which is fine in
    # this case.
    platform_pose_provider = camera_model.platform_pose_provider
    lat, lon, h = platform_pose_provider.ins_llh(frame_time)

    # Location of geod (i.e., mean sea level, which is generally the ground for
    # us) relative to theellipsoid. Positive value means that mean sea level is
    # above the WGS84 ellipsoid.
    offset = geod.height(lat, lon)
    alt_msl = h - offset

    ray_pos, ray_dir = camera_model.unproject(im_pts.T, frame_time)
    ray_pos[:2] = 0
    ray_pos[2] = h
    corner_enu = (ray_pos + ray_dir*(-alt_msl/ray_dir[2])).T

    im_pts = im_pts.astype(np.float64)

    corner_ll = []
    for i in range(len(corner_enu)):
        corner_ll.append(enu_to_llh(*corner_enu[i], lat0=lat, lon0=lon,
                                    h0=0)[:2])

    corner_ll = np.array(corner_ll, dtype=np.float64)

    return im_pts, corner_ll


def measure_image_to_image_homographies(img_fnames, homog_out_dir,
                                        ins_homog_dir=None, num_features=10000,
                                        min_matches=40, reproj_thresh=5,
                                        save_viz_gif=False):
    print("measure_image_to_image_homographies: {} img_fnames".format(len(img_fnames)))
    if len(img_fnames) == 0:
        return

    # Try to read homographies in parallel dir.
    lon_lat_homog = {}

    if ins_homog_dir is not None:
        for fname in glob.glob('%s/*.txt' % ins_homog_dir):
            try:
                h = np.loadtxt(fname)
                img_fname = os.path.splitext(os.path.split(fname)[1])[0]
                lon_lat_homog[img_fname] = np.reshape(h, (3, 3), order='C')
            except Exception as exc:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                msg = "{}:{}\n{}: {}".format(fname, exc_tb.tb_lineno, exc_type.__name__, exc)
                print(msg)

    for _fn in img_fnames:
        # Read first nonempty image to get img width/height
        img = cv2.imread(_fn)
        if img is not None:
            image_height, image_width = img.shape[:2]
            break

    # Find the keypoints and descriptors and match them.
    orb = cv2.ORB_create(nfeatures=num_features, edgeThreshold=21,
                         patchSize=31, nlevels=16,
                         scoreType=cv2.ORB_FAST_SCORE, fastThreshold=10)

    fname0 = img_fnames[0]
    homog_ij = {}
    for i in range(len(img_fnames)):
        fname1 = os.path.splitext(os.path.split(img_fnames[i])[1])[0]

        fname_base = '%s_to_%s' % (fname0, fname1)
        fname_out = '%s/%s.txt' % (homog_out_dir, fname_base)

        print2('Image %i/%i: extracting features from image \'%s\'' %
               (i + 1, len(img_fnames), fname1))

        h1 = lon_lat_homog.get(fname1, None)
        img1 = cv2.imread(img_fnames[i], -1)
        if img1 is None:
            continue

        # Empty image, don't process
        if img1 is None:
            print2("Image is empty, skipping.")
            continue

        if img1.dtype == np.uint16:
            img1 = img1.astype(np.float64)
            img1 -= np.percentile(img1.ravel(), 1)
            img1[img1 < 0] = 0
            img1 /= np.percentile(img1.ravel(), 99)/255
            img1[img1 > 225] = 255
            img1 = np.round(img1).astype(np.uint8)

        if img1.ndim == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1

        kp1, des1 = orb.detectAndCompute(img1_gray, None)

        if i == 0:
            fname0, h0, img0, kp0, des0 = fname1, h1, img1, kp1, des1
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des0, des1)

        if len(matches) < min_matches:
            print2('Only found %d feature matches, could not match images %i '
                   'to %i.' % (len(matches), i, i + 1))
            fname0, img0, kp0, des0 = fname1, img1, kp1, des1
            continue

        pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

        # Because this is an aircraft in forward flight, the optical flow
        # should predominantly be a translation. Let's say, conservatively,
        # that the displacement fits a translation within 1/20 the image's
        # larger dimension.
        if False:
            thresh = max([image_height, image_width])/20
            dxy = pts0 - pts1

            num_inliers = np.zeros(len(dxy))
            for k in range(len(dxy)):
                num_inliers[k] = sum(np.all(np.abs(dxy[i] - dxy) < thresh,
                                            axis=1))

            ind = np.all(dxy[np.argmax(num_inliers)] - dxy < thresh, axis=1)

            pts0 = pts0[ind]
            pts1 = pts1[ind]

            if len(pts0) < min_matches:
                print2('Only found %d matches, could not match images %i to '
                       '%i.' % (len(pts0), i, i + 1))
                fname0, img0, kp0, des0 = fname1, img1, kp1, des1
                continue

        # ----------------------- Consider INS estimate ----------------------
        if h0 is not None and h1 is not None:
            # Homography warping from image i - 1 to image i coordinates.
            h = np.dot(np.linalg.inv(h1), h0)

            pts0_to_1 = np.dot(h, np.hstack([pts0, np.ones((len(pts0), 1))]).T)
            pts0_to_1 = (pts0_to_1[:2]/pts0_to_1[2]).T

            err = np.sqrt(np.sum((pts0_to_1 - pts1)**2, axis=1))

            # Calculate GSD
            xc = image_width/2
            yc = image_height/2
            lon_lats = np.dot(h0, np.array([[xc, xc + 1, xc],
                                            [yc, yc, yc + 1],
                                            [1, 1, 1]]))
            lon_lats = lon_lats[:2]/lon_lats[2]
            dx = llh_to_enu(lon_lats[1, 1], lon_lats[0, 1], 0,
                            lon_lats[1, 0], lon_lats[0, 0], 0)
            dx = np.linalg.norm(dx)
            dy = llh_to_enu(lon_lats[1, 2], lon_lats[0, 2], 0,
                            lon_lats[1, 0], lon_lats[0, 0], 0)
            dy = np.linalg.norm(dy)
            gsd = np.mean([dx, dy])

            # Assume 100 meters of geo-accuracy.
            thresh = 100/gsd

            ind = err < thresh
            pts0 = pts0[ind]
            pts1 = pts1[ind]

        def affine_not_valid(h):
            """Is this affine matrix valid for our flight motion.

            """
            if h is None:
                return True

            if len(h) == 2:
                h = np.vstack([h, [0, 0, 1]])

            try:
                translation, R, scale, S = decompose_affine(h)
            except:
                return True

            #translation = h[:2, 2]
            #scale = np.sqrt(np.sum(h[:, :2]**2, axis=0))
            #angle = (np.arccos(h[0, 0] / scale[0]))*180/np.pi
            angle = np.arctan2(R[1, 0], R[0, 0])*180/np.pi

            shear_angle = float(np.arctan(S)*180/np.pi)

            if min(scale) < 0.8 or max(scale) > 1.3:
                return True

            if abs(angle) > 30:
                return True

            if np.linalg.norm(translation) < 50:
                return False

            if abs(shear_angle) > 10:
                return True

            return False

        # Use OpenCV homogrpahy fitting because it is fast, but it implicitly
        # uses a reprojection threshold of 2, which may be too strict.
        h = cv2.estimateAffinePartial2D(pts0, pts1, False)[0]
        pts0h = np.vstack([pts0.T, np.ones(len(pts0))])

        h_valid = not affine_not_valid(h)
        if h_valid:
            pts = np.dot(h, pts0h)
            pts = pts[:2]
            mask = np.sum((pts - pts1.T)**2, 0) < (2*reproj_thresh)**2
            h_valid = sum(mask) >= min_matches

        if not h_valid:
            # RANSAC loops
            L = len(pts0)
            h = None
            num_inliers = 0
            mask = np.zeros(len(pts0), bool)
            k = 0
            tic = time.time()
            k2 = 0
            while k < 20000:
                k += 1
                k2 += 1

                if k2 == 100:
                    k2 = 0
                    if time.time() > tic + 5:
                        break

                inds = np.random.randint(L, size=3)
                if len(set(inds)) < 3:
                    continue

                h_ = cv2.getAffineTransform(pts0[inds], pts1[inds])
                #h_ = cv2.getPerspectiveTransform(pts0[inds], pts1[inds])

                # Determine if this is an acceptable homography.
                if affine_not_valid(h_):
                    continue

                pts = np.dot(h_, pts0h)
                pts = pts[:2]
                ind_ = np.sum((pts - pts1.T)**2, 0) < (2*reproj_thresh)**2
                num_inliers_ = sum(ind_)
                if num_inliers_ > num_inliers:
                    num_inliers = num_inliers_
                    h = h_
                    mask = ind_

        pts0 = pts0[mask]
        pts1 = pts1[mask]

        if len(pts0) < min_matches:
            print2("Only found %d matches, could not match images %i to %i." %
                   (len(pts0), i, i + 1))
            fname0, img0, kp0, des0 = fname1, img1, kp1, des1
            continue

        if True:
            # Refine with full homography.
            mask = np.zeros(len(pts0))
            h, mask = cv2.findHomography(pts0.reshape(-1, 1, 2),
                                         pts1.reshape(-1, 1, 2),
                                         method=cv2.RANSAC,
                                         ransacReprojThreshold=reproj_thresh,
                                         mask=mask)

            if h is None:
                print2("Only found %d matches, could not match images %i to %i." %
                       (len(pts0), i, i + 1))
                fname0, img0, kp0, des0 = fname1, img1, kp1, des1
                continue

            mask = mask.ravel().astype(bool)

            # Verify whether homography is acceptable. If not, do RANSAC with
            # only acceptable test cases.
            det = np.linalg.det(h)

            pts0 = pts0[mask]
            pts1 = pts1[mask]

        if len(h) == 2:
            # Make a 3x3 affine homography.
            h = np.vstack([h, [0, 0, 1]])

        if False:
            plt.figure()
            plt.subplot('211')
            plt.imshow(img0, cmap='gray', interpolation='none')
            plt.plot(pts0.T[0], pts0.T[1], 'ro')
            plt.title(str(i - 1), fontsize=18)
            plt.subplot('212')
            plt.imshow(img1, cmap='gray', interpolation='none')
            plt.plot(pts1.T[0], pts1.T[1], 'bo')
            plt.title(str(i), fontsize=18)

        if save_viz_gif:
            dir_out = ('%s/refined_registration_viz' %
                       '/'.join(fname_out.split('/')[:-3]))
            viz_fname_out = '%s/%s_to_%s.gif' % (dir_out, fname0, fname1)
            save_registration_gif(img0, img1, h, viz_fname_out)

        if len(pts0) < min_matches:
            print2("Only found %d matches, could not match images %i to %i." %
                   (len(pts0), i, i + 1))
            fname0, img0, kp0, des0 = fname1, img1, kp1, des1
            continue

        try:
            os.makedirs(homog_out_dir)
        except (OSError, IOError):
            pass

        print2('Found %i consistent feature matches between images' %
               len(pts0))

        np.savetxt(fname_out, h.ravel(order='C'))

        # This current homography warps from the previous image to the current
        # one.
        homog_ij[(i - 1, i)] = h

        # Try to reconstruct mapping to previous images by composing
        # homographies. Look at most 10 images back.
        for k in range(1, 10):
            key_desired = (i - k, i)
            try:
                last_h = homog_ij[key_desired]
                last_key = key_desired

                # If the homography is in homog_ij, then it has already been
                # saved.
                continue
            except KeyError:
                pass

            intermediate_key = (i - k, i - k + 1)
            try:
                intermediate_h = homog_ij[intermediate_key]
            except KeyError:
                break

            assert intermediate_key[1] == last_key[0]
            assert key_desired[0] == intermediate_key[0]
            assert key_desired[1] == last_key[1]

            h_desired = np.dot(last_h, intermediate_h)

            # Check to see if there is actually any overlap.
            im_pts = points_along_image_border(image_width, image_height,
                                               num_points=100)
            im_pts = np.vstack([im_pts, np.ones(im_pts.shape[1])])
            im_pts = np.dot(h_desired, im_pts)
            im_pts = im_pts[:2]/im_pts[2]
            ind = np.logical_and(im_pts[0] > 0, im_pts[0] < image_width)
            ind = np.logical_and(ind, im_pts[1] > 0)
            ind = np.logical_and(ind, im_pts[1] < image_height)

            if not np.any(ind):
                break

            homog_ij[key_desired] = h_desired

            # Save this reconstructed homography.
            i1, i2 = key_desired
            fname_from = os.path.splitext(os.path.split(img_fnames[i1])[1])[0]
            fname_to = os.path.splitext(os.path.split(img_fnames[i2])[1])[0]
            fname_base = '%s_to_%s' % (fname_from, fname_to)
            fname_out = '%s/%s.txt' % (homog_out_dir, fname_base)
            np.savetxt(fname_out, h.ravel(order='C'))

            last_h = h_desired
            last_key = key_desired

        fname0, img0, kp0, des0 = fname1, img1, kp1, des1


def measure_image_to_image_homographies_flight_dir(flight_dir,
                                                   multi_threaded=True,
                                                   save_viz_gif=False):
    """Process all image_to_image homographies in the flight directory.

    Triggered by "Fine Tune Tracking" menu item
    A flight directory contains a folder structure where different
    configurations of the cameras is its own subdirectory <sys_config>. For
    example, the camera mount angles may be different or the focal lengths may
    be different. The directory structure for one <sys_config> will look like.

    <flight_dir>/<sys_config>/sys_config.json
    <flight_dir>/<sys_config>/detections
    <flight_dir>/<sys_config>/processed_results
    <flight_dir>/<sys_config>/left_view
    <flight_dir>/<sys_config>/right_view
    <flight_dir>/<sys_config>/center_view

    """
    print('measure_image_to_image_homographies_flight_dir: {}'.format(flight_dir))
    dtype = get_dir_type(flight_dir)
    if dtype != 'flight':
        raise RuntimeError("Called measure_image_to_image_homographies_flight_dir on `{}` directory: {}".format(dtype, flight_dir))

    threads = []
    for sys_config in os.listdir(flight_dir):
        sys_config_dir = '%s/%s' % (flight_dir, sys_config)
        if not os.path.isdir(sys_config_dir):
            warnings.warn("skipping because it's not a real sys_config_dir: {}".format(sys_config_dir))
            continue

        sys_config_dtype = get_dir_type(sys_config_dir)
        if sys_config_dtype != 'sys_config':
            warnings.warn("skipping because it's not a real sys_config_dir: {}".format(sys_config_dir))
            continue

        print('measure_image_to_image_homographies_flight_dir: sys_config_dir: {}'.format(sys_config_dir))
        homog_dir0 = '%s/processed_results/homographies_img_to_img' % (flight_dir)

        for modality in ['rgb', 'ir', 'uv']:
            for sys_str in get_fov_dirs(sys_config_dir):
                image_dir = '%s/%s' % (sys_config_dir, sys_str)
                print('Image directory:', image_dir)

                image_glob = '%s/*%s.*' % (image_dir, modality)
                img_fnames = glob.glob(image_glob)
                img_fnames = sorted(img_fnames)

                homog_dir = '%s/%s_%s' % (homog_dir0, sys_str.lower(), modality)

                print2('Creating homographies for', image_glob)

                if multi_threaded:
                    thread = threading.Thread(target=measure_image_to_image_homographies,
                                              args=(img_fnames, homog_dir, None,
                                                    20000, 40, 5, save_viz_gif))
                    thread.daemon = True
                    thread.start()
                    threads.append(thread)
                else:
                    measure_image_to_image_homographies(img_fnames, homog_dir,
                                                        ins_homog_dir=None,
                                                        num_features=20000,
                                                        min_matches=40,
                                                        reproj_thresh=5,
                                                        save_viz_gif=save_viz_gif)

    if multi_threaded:
        # Block until all threads finished (if any).
        any_alive = True
        while any_alive:
            any_alive = False
            for thread in threads:
                if thread.is_alive():
                    any_alive = True


def debayer_image(image, bayer_pattern):
    """Debayer image or return image if already debayered.

    :param image: Image to debayer or to return if already debayered.
    :type image: Numpy single or mutli-channel array

    :param bayer_pattern:
    :type bayer_pattern: str {'bayer_rggb8', 'bayer_grbg8', 'bayer_bggr8',
                              'bayer_gbrg8', 'bayer_rggb16', 'bayer_grbg16',
                              'bayer_bggr16', 'bayer_gbrg16'}

    """
    if image.ndim == 2:
        return cv2.cvtColor(image, bayer_patterns[bayer_pattern])
    if (np.all(image[:, :, 0] == image[:, :, 1]) and
        np.all(image[:, :, 0] == image[:, :, 2])):
        # Bayered image encoded as three seperate identical channels.
        return cv2.cvtColor(image[:, :, 0], bayer_patterns[bayer_pattern])
    else:
        return image


def debayer_image_list(img_fnames, img_fnames_out):
    """Process all images in list of filenames.

    :param img_fnames: List of image paths to read Bayered imagery from.
    :type img_fnames: list

    :param img_fname_out: List of image paths to save deBayered imagery to.
    :type img_fname_out: list

    """
    for i in range(len(img_fnames)):
        img_fname = img_fnames[i]
        img_fname_out = img_fnames_out[i]

        with open(img_fname) as f:
            tags = exifread.process_file(f)
            try:
                if tags['Image SamplesPerPixel'].field_type == 3:
                    if img_fname_out != img_fname:
                        print2('Copying \'%s\' to \'%s\'' % (img_fname,
                                                             img_fname_out))
                        copyfile(img_fname, img_fname_out)
                    else:
                        print2('Image \'%s\' already debayered' %
                               img_fname_out)

                continue
            except KeyError:
                pass

        print2('Reading:', img_fname)
        img = cv2.imread(img_fname)

        if img is None:
            continue

        try:
            os.makedirs(os.path.split(img_fname_out)[0])
        except OSError:
            pass

        img2 = debayer_image(img, 'bayer_gbrg8')

        if img2 is not img:
            print2('Saving:', img_fname_out)
            cv2.imwrite(img_fname_out, img2)
        else:
            if img_fname_out != img_fname:
                print2('Copying \'%s\' to \'%s\'' % (img_fname,
                                                     img_fname_out))
                copyfile(img_fname, img_fname_out)
            else:
                print2('Image \'%s\' already debayered' % img_fname_out)

    print2('Thread Finished')


def debayer_dir_tree(src_dir, dst_dir=None, num_threads=1):
    """Debayer images from RGB camera into 3-channel RGB.

    :param src_dir: Every file with pattern '*rgb.tif' will be considered for
        deBayering. If the image is already debayered, it won't be changed.
    :type src_dir: str

    :param dst_dir: Location to save results (defaults to same source folder).
    :type dst_dir: str

    """

    if dst_dir is None:
        dst_dir = src_dir

    img_fnames = []
    img_fnames_out = []
    for img_fname in glob.glob('%s/*rgb.tif' % src_dir):
        img_fnames.append(img_fname)
        img_fnames_out.append(img_fname.replace(src_dir, dst_dir))

    for root, dirnames, filenames in os.walk(src_dir):
        for dirname in dirnames:
            curr_dir = '%s/%s/*rgb.tif' % (root, dirname)
            for img_fname in glob.glob(curr_dir):
                img_fnames.append(img_fname)
                img_fnames_out.append(img_fname.replace(src_dir, dst_dir))

    print2('Found %i RGB images to consider' % len(img_fnames))

    num_threads = min([num_threads, len(img_fnames)])

    if num_threads == 1:
        debayer_image_list(img_fnames, img_fnames_out)
    else:
        ind = np.round(np.linspace(0, len(img_fnames) - 1, num_threads))
        ind = ind.astype(np.int)
        threads = []
        for i in range(len(ind) - 1):
            thread = threading.Thread(target=debayer_image_list,
                                      args=(img_fnames[ind[i]:ind[i+1]],
                                            img_fnames_out[ind[i]:ind[i+1]]))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # Block until all threads finished (if any).
        any_alive = True
        while any_alive:
            any_alive = False
            for thread in threads:
                if thread.is_alive():
                    any_alive = True


def stretch_constrast(img):
    img = img.astype(np.float32)
    img -= np.percentile(img.ravel(), 0.1)
    img[img < 0] = 0
    img /= np.percentile(img.ravel(), 99.9)/255
    img[img > 225] = 255
    img = np.round(img).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(5, 5))
    if img.ndim == 3:
        HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        HLS[:, :, 1] = clahe.apply(HLS[:, :, 1])
        img = cv2.cvtColor(HLS, cv2.COLOR_HLS2BGR)
    else:
        img = clahe.apply(img)

    return img


def save_geotiff(img, camera_model, frame_time, geotiff_fname,
                 compression_setting=None, verbosity=0):
    gdal_drv = gdal.GetDriverByName('GTiff')

    if compression_setting is not None:
        # Compress the RGB and UV.
        gdal_settings = ['COMPRESS=JPEG',
                         'JPEG_QUALITY=%i' % compression_setting]
    else:
        gdal_settings = []

    ds = gdal_drv.Create(geotiff_fname, img.shape[1], img.shape[0], img.ndim,
                         gdal.GDT_Byte, gdal_settings)
    ds.SetProjection(wgs84_cs.ExportToWkt())

    im_pts, corner_ll = get_image_boundary(camera_model, frame_time)
    corner_ll = corner_ll[:, ::-1]

    # Affine transform warping image coordinates to latitude/longitude.
    A, inliers = cv2.estimateAffine2D(np.reshape(im_pts, (1, -1, 2)),
                                   np.reshape(corner_ll, (1, -1, 2)), True)

    if A is None:
        # Failure can happen when the aircraft is just above the ground.
        center = np.mean(corner_ll, axis=0)
        corners = corner_ll - center
        scale = np.max(np.abs(corners))
        corners /= scale

        A, inliers = cv2.estimateAffine2D(np.reshape(im_pts, (1, -1, 2)),
                                       np.reshape(corners, (1, -1, 2)),
                                       True)
        A = np.vstack([A, [0, 0, 1]])
        S = np.identity(3)
        S[0, 0] = S[1, 1] = scale
        A = np.dot(S, A)
        T = np.identity(3)
        T[:2, 2] = center
        A = np.dot(T, A)[:2]

        if False:
            corner_ll2 = np.dot(A, np.vstack([im_pts.T, [1, 1, 1, 1]])).T
            plt.plot(corner_ll.T[0], corner_ll.T[1])
            plt.plot(corner_ll2.T[0], corner_ll2.T[1])

    # Xp = padfTransform[0] + P*padfTransform[1] + L*padfTransform[2]
    # Yp = padfTransform[3] + P*padfTransform[4] + L*padfTransform[5]
    geotrans = [A[0, 2], A[0, 0], A[0, 1], A[1, 2], A[1, 0], A[1, 1]]

    ds.SetGeoTransform(geotrans)

    if ds.RasterCount == 1:
        ds.GetRasterBand(1).WriteArray(img[:, :], 0, 0)
    else:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i+1).WriteArray(img[:, :, i], 0, 0)

    ds.FlushCache()  # Write to disk.
    ds = None
    if verbosity >= 3:
        print2('Saved \'%s\'' % geotiff_fname)


def create_geotiffs_glob(image_dir, modality, output_dir, camera_model_fname,
                         compression_quality=75, stretch_constrast=None,
                         verbosity=0):
    # type: (str, str, str, str, int, Callable, int) -> None
    """Create geotiffs for images from one camera.

    :param image_glob: Global expression indicating all images to consider. For
        example, 'BACKUP/fl03/CENT/*rgb.tif would indicate all of the RGB
        frames for one image system.
    :type image_glob: str

    :param output_dir: System path to directory to save resulting GeoTIFF.
    :type output_dir: str

    :param camera_model_fname: System path to the camera model yaml.
    :type camera_model_fname:  str

    :param compression_quality: JPEG compression quality.
    :type compression_quality: int [0,100]

    """
    file_assert(camera_model_fname, 'Cannot load camera model. File')

    dir_assert(image_dir, 'Image dir')

    # This will do some duplication of NavState parsing but I do not have time to fix
    ret = parse_image_directory(image_dir, modality=modality)
    img_fname_to_time = ret[0]
    img_time_to_fname = ret[1]
    platform_pose_provider = ret[2]
    effort_type = ret[3]
    trigger_type = ret[4]

    camera_model = load_from_file(camera_model_fname, platform_pose_provider)

    frame_times = sorted(list(set(img_time_to_fname.keys())))

    if False:
        frame_times = [_ for _ in frame_times
                       if _ >= 1557373173.6 and _ <= 1557373193.2]

    if verbosity >= 1:
        print2('create_greotiffs_glob: {}'.format(modality))
    for frame_time in frame_times:
        src_img_fname = img_time_to_fname[frame_time]
        if verbosity >= 2:
            print2('Processing \'%s\'' % src_img_fname)
        img = cv2.imread(src_img_fname)

        if img is None:
            continue

        if img.ndim == 3:
            img = img[:, :, ::-1]
        elif 'rgb.tif' in src_img_fname:
            # Wasn't debayered yet.
            img = debayer_image(img, 'bayer_gbrg8')

        if stretch_constrast is not None:
            img = stretch_constrast(img)

        base_fname = os.path.splitext(os.path.split(src_img_fname)[1])[0]
        fname = '%s/%s.tif' % (output_dir, base_fname)

        if img.shape[1]*img.shape[0] > 1500*1500:
            # Compress the RGB and UV.
            save_geotiff(img, camera_model, frame_time, fname,
                         compression_quality)
        else:
            # Don't compress the IR.
            save_geotiff(img, camera_model, frame_time, fname, None)

def fac_geotiff_thunk(image_dir, modality, output_dir, camera_model_fname,
                         compression_quality=75, stretch_constrast=None,
                         verbosity=0):
    # type: (str, str, str, str, int, Callable, int) -> Callable
    """Create a callback to be passed to thread"""

    def thread_thunk():
        #try:
        create_geotiffs_glob(image_dir=image_dir, modality=modality,
                             output_dir=output_dir,
                             camera_model_fname=camera_model_fname,
                             compression_quality=compression_quality,
                             verbosity=verbosity)
    #    except FileNotFoundError as exc:
    #        print2(repr(exc))

    #    except Exception as exc:
    #        exc_type, value, traceback = sys.exc_info()
    #        print2(
    #            'Unexpected exception: {}\n{}\n{}'.format(exc_type, value, traceback))
    return thread_thunk


def create_all_geotiff(flight_dir, output_dir=None, quality=75,
                       multi_threaded=False, verbosity=0):
    """Create geotiffs for all images in the flight directory.

    A flight directory contains a folder structure where different
    configurations of the cameras is its own subdirectory <sys_config>. For
    example, the camera mount angles may be different or the focal lengths may
    be different. The directory structure for one <sys_config> will look like.

    <flight_dir>/<sys_config>/sys_config.json
    <flight_dir>/<sys_config>/detections
    <flight_dir>/<sys_config>/processed_results
    <flight_dir>/<sys_config>/left_view
    <flight_dir>/<sys_config>/right_view
    <flight_dir>/<sys_config>/center_view

    """
    if output_dir is None:
        output_dir = '%s/processed_results/geotiffs' % flight_dir

    make_path(output_dir)

    jobs = []
    for sys_config in os.listdir(flight_dir):
        sys_config_dir = '%s/%s' % (flight_dir, sys_config)
        if not os.path.isdir(sys_config_dir):
            pass

        try:
            sys_config_fname = '%s/sys_config.json' % sys_config_dir
            with open(sys_config_fname, "r") as input_file:
                camera_model_paths = json.load(input_file)
        except (OSError, IOError):
            print('Could not read \'%s\', skipping summary for system '
                  'configuration \'%s\'' % (sys_config_fname, sys_config_dir))
            continue

        for modality in ['ir', 'uv','rgb']:
            for fov_dirname in get_fov_dirs(sys_config_dir):
                sys_str = first_wordlike(fov_dirname)
                image_dir = os.path.join(sys_config_dir, fov_dirname)

                sys_str = sys_str.lower()

                try:
                    key = '%s_%s_yaml_path' % (sys_str, modality)
                    camera_model_fname = camera_model_paths[key]
                except KeyError:
                    print('Camera model path specification file \'%s\' does'
                          'not include a specification for \'%s\', skipping '
                          % (sys_config_fname, key))
                    continue

                try:
                    load_from_file(camera_model_fname)
                except IOError:
                    warnings.warn('Could not load the camera model '
                                  'yaml specified to be located: %s. '
                                  'Skipping' % camera_model_fname)
                    continue

                image_glob = '%s/*%s.tif' % (image_dir, modality)
                print2('Creating GeoTIFFs for', image_glob)

                thunk = fac_geotiff_thunk(
                            image_dir=image_dir,
                            modality=modality,
                            output_dir=output_dir,
                            camera_model_fname=camera_model_fname,
                            compression_quality=quality,
                            verbosity=verbosity)

                jobs.append(thunk)

    if not multi_threaded:
        for thunk in jobs:
            thunk()
        return

    threads = [threading.Thread(target=thunk) for thunk in jobs]
    print('Threads: {}'.format(len(threads)))
    for thread in threads:
        thread.daemon = True
        thread.start()

    # Block until all threads finished (if any).
    any_alive = True
    while any_alive:
        any_alive = False
        for thread in threads:
            if thread.is_alive():
                any_alive = True


def create_flight_summary(flight_dir, save_shapefile_per_image=False):
    """Create flight summary for a flight directory.

    A flight directory contains a folder structure where different
    configurations of the cameras is its own subdirectory <sys_config>. For
    example, the camera mount angles may be different or the focal lengths may
    be different. The directory structure for one <sys_config> will look like.

    <flight_dir>/<sys_config>/sys_config.json
    <flight_dir>/<sys_config>/detections
    <flight_dir>/<sys_config>/processed_results
    <flight_dir>/<sys_config>/left_view
    <flight_dir>/<sys_config>/right_view
    <flight_dir>/<sys_config>/center_view

    """
    flight_id = os.path.basename(flight_dir)
    project_id = os.path.basename(os.path.dirname(flight_dir))
    print("FLIGHT_ID: %s" % flight_id)
    print("PROJECT_ID: %s" % project_id)
    process_summary = {'fovs': {}, 'models': [], 'shapefile_count': 0, 'fails': {}}

    camera_models = {}
    img_fname_to_time = {}
    img_time_to_fname = {}
    fnames_by_system = {}
    effort_type = defaultdict(lambda: '')
    trigger_type = defaultdict(lambda: '')

    # ------------------------------------------------------------------------
    # We loop over <sys_config> within the flight directory to find the camera
    # models in:
    # <flight_dir>/<sys_config>/sys_config.json
    # and consider each image in:
    # <flight_dir>/<sys_config>/left_view
    # <flight_dir>/<sys_config>/right_view
    # <flight_dir>/<sys_config>/center_view

    fn_glob = os.path.join(flight_dir, '*/*/*meta.json')
    count = 0
    est_metas = glob.glob(fn_glob)
    total = len(est_metas) * 3

    for sys_config in os.listdir(flight_dir):
        sys_config_dir = '%s/%s' % (flight_dir, sys_config)
        if not os.path.isdir(sys_config_dir):
            pass

        try:
            sys_config_fname = '%s/sys_config.json' % sys_config_dir
            with open(sys_config_fname, "r") as input_file:
                camera_model_paths = json.load(input_file)
        except (OSError, IOError):
            print('Could not read \'%s\', skipping summary for system '
                  'configuration \'%s\'' % (sys_config_fname, sys_config_dir))
            continue

        # Extract camera models.
        for sys_str in get_fov_dirs(sys_config_dir):
            image_dir = '%s/%s' % (sys_config_dir, sys_str)

            try:
                ret = parse_image_directory(image_dir)
            except:
                print('Not considering: \'%s\'' % image_dir)
                continue

            img_fname_to_time.update(ret[0])
            img_time_to_fname.update(ret[1])
            effort_type.update(ret[3])
            trigger_type.update(ret[4])

            platform_pose_provider = ret[2]

            sys_str = reduce_fov(sys_str)

            for cam_str in ['rgb', 'ir', 'uv']:
                img_fnames = glob.glob('%s/*%s.*' % (image_dir, cam_str))
                key = '%s_%s_yaml_path' % (sys_str, cam_str)

                try:
                    camera_model_fname = camera_model_paths[key]
                except KeyError:
                    print('Camera model path specification file \'%s\' does'
                          'not include a specification for \'%s\', skipping '
                          % (sys_config_fname, key))
                    continue

                sys_str2 = '%s_%s' % (sys_str, cam_str)

                if sys_str2 not in fnames_by_system:
                    fnames_by_system[sys_str2] = []

                fnames_by_system[sys_str2] = fnames_by_system[sys_str2] + img_fnames

                if len(img_fnames) > 0:
                    # There are images for this camera.
                    try:
                        camera_model = load_from_file(camera_model_fname,
                                                      platform_pose_provider)
                    except IOError:
                        warnings.warn('Could not load the camera model '
                                      'yaml specified to be located: %s. '
                                      'Skipping' % camera_model_fname)
                        continue

                    for img_fname in img_fnames:
                        camera_models[img_fname] = camera_model
                        if camera_model_fname not in process_summary['models']:
                            process_summary['models'].append(camera_model_fname)
                        process_summary['fovs'].update({cam_str: len(img_fnames)})
                        count += 1
                        # update_progress(count, total, flight_dir=flight_dir, msg="create_flight_summary")
    # ------------------------------------------------------------------------

    # Calculate each image's boundary in latitude and longitude.
    print2('Calculating each image\'s footprint in latitude and longitude by '
           'leveraging INS-reported pose and assuming the ground is at mean '
           'sea level.')
    corner_ll = {}
    aircraft_state = {}
    for sys_str in fnames_by_system:
        img_fnames = fnames_by_system[sys_str]

        if len(img_fnames) == 0:
            warnings.warn('No images found for {}'.format(sys_str))
            continue

        for img_fname in img_fnames:
            try:
                frame_time = img_fname_to_time[img_fname]
            except KeyError:
                warnings.warn('Missing time for image: {}'.format(img_fname))
                failcount = process_summary['fails'].get(sys_str, 0)
                process_summary['fails'][sys_str] = failcount + 1

                continue

            try:
                camera_model = camera_models[img_fname]
            except KeyError:
                warnings.warn('No camera model for {}'.format(img_fname))
                continue

            ret = get_image_boundary(camera_model, frame_time)
            corner_ll[img_fname] = ret

            # Extract aircraft state information.
            nsp = camera_model.platform_pose_provider
            lat, lon, h = nsp.llh(frame_time)
            heading, pitch, roll, = nsp.ins_heading_pitch_roll(frame_time)

            aircraft_state[img_fname] = [frame_time, lat, lon, h, heading,
                                         pitch, roll]

    # ------------------------------------------------------------------------
    # Save homographies estimated by INS.
    homog_dir = '%s/processed_results/homographies_img_to_lonlat' % (flight_dir)

    for sys_str in fnames_by_system:
        homog_dir2 = '%s/%s' % (homog_dir, sys_str)
        img_fnames = fnames_by_system[sys_str]
        for img_fname in img_fnames:
            try:
                im_pts, ll = corner_ll[img_fname]
            except KeyError:
                continue

            h, status = cv2.findHomography(im_pts, ll[:, ::-1])

            if False:
                ll2 = np.dot(h, np.vstack([im_pts.T, np.ones(4)]))
                ll2 = ll2[:2]/ll2[2]
                print(ll2)
                print(ll[:, ::-1].T)

            if h is None:
                continue

            try:
                os.makedirs(homog_dir2)
            except OSError:
                pass

            fname_base = os.path.split(os.path.splitext(img_fname)[0])[1]
            fname = '%s/%s.txt' % (homog_dir2, fname_base)
            np.savetxt(fname, h.ravel(order='C'))

    # ------------------------------------------------------------------------
    for sys_str in fnames_by_system:
        flid_sys_str = "%s_%s_%s" % (project_id, flight_id, sys_str)
        print2('Processing image footprints for system:', sys_str)
        img_fnames = fnames_by_system[sys_str]

        if len(img_fnames) == 0:
            continue

        kml = simplekml.Kml(name=flid_sys_str)

        shp_shapes = []
        shp_shapes_fnames = []
        shape_img_basenames = []
        for img_fname in img_fnames:
            try:
                ret = corner_ll[img_fname][1]
            except KeyError:
                continue

            lats = ret[:, 0]
            lons = ret[:, 1]
            lls = zip(lons.ravel(), lats.ravel())
            lls = list(lls)
            lls.append(lls[0])

            # Check area in m^2.
            s = [llh_to_enu(_[1], _[0], 0, lls[0][1], lls[0][0], 0)
                 for _ in lls]
            area = Polygon(s).area

            if area > 1000*1000:
                # Must be a bad FOV (e.g., pitched too far up).
                continue

            """
            boundary0 = boundary
            if boundary is None:
                boundary = Polygon(corner_ll)
            else:
                boundary = boundary.union(Polygon(corner_ll))

            if len(mapping(boundary)['coordinates']) > 1 or i == len(frame_times) - 1:
                lats, lons = np.array(mapping(boundary0)['coordinates']).T
                pol = kml.newpolygon(name='Coverage')
                pol.outerboundaryis = zip(lons.ravel(), lats.ravel())
                pol.style.linestyle.color = simplekml.Color.red
                pol.style.linestyle.width = 2
                pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.red)
                boundary = None
            """

            pol = kml.newpolygon(name=os.path.split(img_fname)[1])
            pol.outerboundaryis = lls

            c = kml_color_map[sys_str]
            pol.style.linestyle.color = c
            pol.style.linestyle.width = 2
            pol.style.polystyle.color = simplekml.Color.changealphaint(50, c)
            pol.timestamp = simplekml.TimeStamp(str(img_fname_to_time[img_fname]))

            shp_shapes.append(lls)
            shape_img_basenames.append(os.path.split(img_fname)[1])
            shp_shapes_fnames.append(img_fname)
        if len(shp_shapes) > 0:
            shapefile_dir = ('%s/processed_results/fov_shapefiles/' %
                             flight_dir)

            try:
                os.makedirs(shapefile_dir)
            except OSError:
                pass

            kml.savekmz('%s/%s.kml' % (shapefile_dir, flid_sys_str))

            w = shapefile.Writer('%s/%s.shp' % (shapefile_dir, flid_sys_str),
                                 shapeType=shapefile.POLYGON)
            w.autoBalance = 1
            w.field('image_file', 'C', size=255)
            w.field('time', 'N', decimal=3)
            w.field('latitude', 'N', decimal=10)
            w.field('longitude', 'N', decimal=10)
            w.field('altitude', 'N', decimal=3)
            w.field('heading', 'N', decimal=5)
            w.field('pitch', 'N', decimal=5)
            w.field('roll', 'N', decimal=5)
            w.field('effort', 'C', size=255)
            w.field('trigger', 'C', size=255)
            w.field('reviewed', 'C', size=5)
            w.field('fate', 'C', size=255)

            for i in range(len(shp_shapes)):
                try:
                    w.poly([shp_shapes[i]])
                    tmp = aircraft_state[shp_shapes_fnames[i]]
                    frame_time, lat, lon, h, heading, pitch, roll = tmp
                    w.record(shape_img_basenames[i], frame_time, lat, lon, h, heading,
                         pitch, roll, effort_type[shp_shapes_fnames[i]],
                         trigger_type[shp_shapes_fnames[i]], 'False', '')
                except Exception as e:
                    warnings.warn('{}: {}'.format(e.__class__.__name__, e))

            w.close()

            with open('%s/%s.prj' % (shapefile_dir, flid_sys_str), "w") as fout:
                fout.write(wgs84_wkt)

            if save_shapefile_per_image:
                # Write each individual frame as a seperate shapefile.
                shapefile_dir = ('%s/processed_results/fov_shapefiles/%s_fovs'
                                 % (flight_dir, sys_str))

                try:
                    os.makedirs(shapefile_dir)
                except OSError:
                    pass

                for i in range(len(shp_shapes)):
                    base = os.path.splitext(shape_img_basenames[i])[0]
                    w = shapefile.Writer('%s/%s.shp' % (shapefile_dir, base),
                                         shapeType=shapefile.POLYGON)
                    w.autoBalance = 1
                    w.field('image_file', 'C', size=255)
                    w.field('time', 'N', decimal=3)
                    w.field('latitude', 'N', decimal=10)
                    w.field('longitude', 'N', decimal=10)
                    w.field('altitude', 'N', decimal=3)
                    w.field('heading', 'N', decimal=5)
                    w.field('pitch', 'N', decimal=5)
                    w.field('roll', 'N', decimal=5)
                    w.field('effort', 'C', size=255)
                    w.field('trigger', 'C', size=255)
                    w.field('reviewed', 'C', size=5)
                    w.field('fate', 'C', size=255)

                    w.poly([shp_shapes[i]])

                    tmp = aircraft_state[shp_shapes_fnames[i]]
                    frame_time, lat, lon, h, heading, pitch, roll = tmp

                    w.record(shape_img_basenames[i], frame_time, lat, lon, h,
                             heading, pitch, roll, effort_type[shp_shapes_fnames[i]],
                             trigger_type[shp_shapes_fnames[i]], 'False', '')

                    w.close()

                    with open('%s/%s.prj' % (shapefile_dir, base), "w") as fout:
                        fout.write(wgs84_wkt)

                process_summary['shapefile_count'] += 1

    return process_summary


def visualize_registration_homographies(flight_dir, sys_str='rgb'):
    """

    :param homography_dir: Directory containing subdirectories for
        different cameras containing text files with names matching the
        associated image filename where the homography maps image
        coordinates to (longitude, latitude).
    :type homography_dir: str

    :param animal_min_meters: Minimum length of animal in meters.
    :type animal_min_meters: float

    :param animal_max_meters: Maximum length of animal in meters.
    :type animal_max_meters: float

    :param geo_registration_error: Assumed geo-registration error.
    :type geo_registration_error: float

    """
    img_to_lonlat_homog_dir = ('%s/processed_results/'
                               'homographies_img_to_lonlat' % flight_dir)

    img_to_img_homog_dir = ('%s/processed_results/'
                            'homographies_img_to_img' % flight_dir)

    dir_out = '%s/processed_results/ins_registration_viz' % flight_dir

    try:
        os.makedirs(dir_out)
    except OSError:
        pass

    # Find homographies
    img_to_lonlat_homog = {}
    for root, dirnames, filenames in os.walk(img_to_lonlat_homog_dir):
        for fname in glob.glob('%s/*.txt' % root):
            try:
                h = np.loadtxt(fname)
                img_fname = os.path.splitext(os.path.split(fname)[1])[0]
                if img_fname[-len(sys_str):] != sys_str:
                    continue

                img_to_lonlat_homog[img_fname] = np.reshape(h, (3, 3),
                                                            order='C')
            except (IOError, OSError):
                pass

    img_fnames = sorted(list(img_to_lonlat_homog.keys()))

    def get_image(fname):
        for fov in get_fov_dirs(flight_dir):
            base_dir = os.path.join(flight_dir, fov)
            fnames = glob.glob('%s/%s.tif' % (base_dir, fname))
            if len(fnames) > 0:
                img = cv2.imread(fnames[0])
                if img is not None:
                    return img

    for _fn in img_fnames:
        img = get_image(_fn)
        if img is not None:
            image_height, image_width = img.shape[:2]
            break

    for i in range(1, len(img_fnames)):
        fname1 = img_fnames[i - 1]
        fname2 = img_fnames[i]
        h1 = img_to_lonlat_homog[fname1]
        h2 = img_to_lonlat_homog[fname2]
        h12 = np.dot(np.linalg.inv(h2), h1)

        # Check to see if there is actually any overlap.
        im_pts = points_along_image_border(image_width, image_height,
                                           num_points=100)
        im_pts = np.vstack([im_pts, np.ones(im_pts.shape[1])])
        im_pts = np.dot(h12, im_pts)
        im_pts = im_pts[:2]/im_pts[2]
        ind = np.logical_and(im_pts[0] > 0, im_pts[0] < image_width)
        ind = np.logical_and(ind, im_pts[1] > 0)
        ind = np.logical_and(ind, im_pts[1] < image_height)

        if not np.any(ind):
            continue

        img1 = get_image(fname1)
        img2 = get_image(fname2)

        if img1 is None or img2 is None:
            continue

        fname_out = '%s/%s_to_%s.gif' % (dir_out, fname1, fname2)

        print2('Saving image \'%s\'' % fname_out)

        save_registration_gif(img1, img2, h12, fname_out)

    # ------------------------------------------------------------------------
    dir_out = '%s/processed_results/refined_registration_viz' % flight_dir

    try:
        os.makedirs(dir_out)
    except OSError:
        pass

    img_to_img_homog = {}
    for root, dirnames, filenames in os.walk(img_to_img_homog_dir):
        for fname in glob.glob('%s/*.txt' % root):
            try:
                h = np.loadtxt(fname)

                if len(h) == 6:
                    h = np.reshape(h, (2, 3), order='C')
                    h = np.vstack([h, [0, 0, 1]])
                else:
                    h = np.reshape(h, (3, 3), order='C')

                img_fname = os.path.splitext(os.path.split(fname)[1])[0]
                img_to_img_homog[img_fname] = h
            except (IOError, OSError):
                pass

    img_pair_fnames = sorted(list(img_to_img_homog.keys()))
    for img_pair_fname in img_pair_fnames:

        fname1, fname2 = img_pair_fname.split('_to_')
        h12 = img_to_img_homog[img_pair_fname]
        img1 = get_image(fname1)
        img2 = get_image(fname2)

        if img1 is None or img2 is None:
            continue

        fname_out = '%s/%s_to_%s.gif' % (dir_out, fname1, fname2)

        print2('Saving image \'%s\'' % fname_out)

        save_registration_gif(img1, img2, h12, fname_out)


def save_registration_gif(img1, img2, h_1_to_2, fname):
    """
    :param img1: First image.
    :type img1: Numpy array

    :param img2: Second image.
    :type img2: Numpy array

    :param h_1_to_2: Homography that warps img1 coordinates to img2
        coordinates.
    :type h_1_to_2: 3x3 array

    """
    img_rect = cv2.warpPerspective(img1, h_1_to_2, img1.shape[:2][::-1])

    if max(img_rect.shape) > 3000:
        s = 3000/max(img_rect.shape)
        img_rect = cv2.resize(img_rect, None, fx=s, fy=s)
        img2 = cv2.resize(img2, None, fx=s, fy=s)

    images = [Image.fromarray(img2), Image.fromarray(img_rect)]
    images[0].save(fname, format='GIF', append_images=images[1:],
                   save_all=True, duration=300, loop=0)


def detection_summary(flight_dir=None, detection_csvs=None,
                      img_to_lonlat_homog_dir=None, img_to_img_homog_dir=None,
                      animal_min_meters=0.1, animal_max_meters=7,
                      geo_registration_error=10, save_annotations=False):
    """
    :param flight_dir: Path to flight directory containing subdirectories:
        LEFT, RIGHT, CENTER.
    :type flight_dir: str

    :param detection_csvs: Path to the detection csv to be processed or a list
        of paths to csv to be processed.
    :type detection_csvs: str | list of str

    :param img_to_lonlat_homog_dir: Path to the directory containing
        homographies to warp from image coordinates to longitude and latitude
        that are created by the 'create_flight_summary' function call.
        Typically, this will be in '<flight_dir>/processed_results/
        homographies_img_to_lonlat', which is used by default if None is
        passed.
    :type img_to_lonlat_homog_dir: str | None

    :param img_to_img_homog_dir: Path to the directory containing
        homographies to warp from image coordinates in one image to image
        coordinates in sucessive images from the same camera. These
        homographies are generated by the function call
        'measure_image_to_image_homographies_flight_dir'.
        Typically, this will be in '<flight_dir>/processed_results/
        homographies_img_to_img', which is used by default if None is passed.
    :type img_to_img_homog_dir: str | None

    :param animal_min_meters: Minimum length of animal in meters.
    :type animal_min_meters: float

    :param animal_max_meters: Maximum length of animal in meters.
    :type animal_max_meters: float

    :param geo_registration_error: Assumed geo-registration error to use when
        determining whether two detections are redundant. If the center of a
        detection bounding box in one image can be mapped to the center of a
        detection bounding box in another nearby-in-time image within this
        distance, then the detection is assumed to of the same entity. If
        fine-tuned registration homographies have been created, they will be
        used to more-precisely determine whether the detections are redundant.
    :type geo_registration_error: float

    """
    if any([flight_dir is None, detection_csvs is None]):
        raise ValueError("Must supply all of flight_dir, detection_csv ")

    if isinstance(detection_csvs, str):
        # Even if just one path, make a list of length one.
        detection_csvs = [detection_csvs]

    if img_to_lonlat_homog_dir is None:
        img_to_lonlat_homog_dir = ('%s/processed_results/'
                                   'homographies_img_to_lonlat' % flight_dir)

    if img_to_img_homog_dir is None:
        img_to_img_homog_dir = ('%s/processed_results/'
                                'homographies_img_to_img' % flight_dir)

    if not os.path.isdir(img_to_lonlat_homog_dir):
        raise Exception('This function requires that homographies have '
                        'already been generated. Try running '
                        'create_flight_summary on the flight directory '
                        'first.')

    # Find homographies
    img_to_lonlat_homog = {}
    for root, dirnames, filenames in os.walk(img_to_lonlat_homog_dir):
        for fname in glob.glob('%s/*.txt' % root):
            try:
                h = np.loadtxt(fname)
                img_fname = os.path.splitext(os.path.split(fname)[1])[0]
                img_to_lonlat_homog[img_fname] = np.reshape(h, (3, 3),
                                                            order='C')
            except (IOError, OSError):
                pass

    print2('Found %i image-to-lon/lat homographies' % len(img_to_lonlat_homog))

    img_to_img_homog = {}
    for root, dirnames, filenames in os.walk(img_to_img_homog_dir):
        for fname in glob.glob('%s/*.txt' % root):
            try:
                h = np.loadtxt(fname)

                if len(h) == 6:
                    h = np.reshape(h, (2, 3), order='C')
                    h = np.vstack([h, [0, 0, 1]])
                else:
                    h = np.reshape(h, (3, 3), order='C')

                img_fname = os.path.splitext(os.path.split(fname)[1])[0]
                img_to_img_homog[img_fname] = h
            except (IOError, OSError):
                pass

    print2('Found %i image-to-image fine-tuned homographies' %
           len(img_to_img_homog))

    for detection_csv in detection_csvs:
        print('Processing \'%s\'' % detection_csv)
        __process_detection_csv(flight_dir, detection_csv, img_to_lonlat_homog,
                                img_to_img_homog, animal_min_meters,
                                animal_max_meters, geo_registration_error,
                                save_annotations)


def __process_detection_csv(flight_dir, detection_csv, img_to_lonlat_homog,
                            img_to_img_homog, animal_min_meters,
                            animal_max_meters, geo_registration_error,
                            save_annotations):
    """Process one csv.

    """
    # Dictionary that accepts the image filename and returns a list of
    # Detection objects.
    detections = {}
    img_fnames_set = set()
    img_fnames = []
    num_suppressed = 0
    num_dets = 0

    # Read the detection csv and populate 'detections'.
    with open(detection_csv) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='#')
        for row in csv_reader:
            if len(row) == 1:
                continue

            uid = row[0]   # Detection or Track-id
            image_fname = row[1]    # Video or Image Identifier
            frame_id = row[2]       # Unique Frame Identifier

            # Img-bbox(TL_x,TL_y,BR_x,BR_y)
            left = float(row[3])
            top = float(row[4])
            right = float(row[5])
            bottom = float(row[6])
            image_bbox = np.array([(left, top), (right, bottom)])

            confidence = row[7]         # Detection confidence
            length = row[8]             # Fish Length (0 or -1 if invalid)
            confidence_pairs = row[9:]

            img_fname = os.path.splitext(os.path.split(image_fname)[1])[0]

            try:
                h = img_to_lonlat_homog[img_fname]
            except KeyError:
                print2('Could not find a homography for \'%s\'. Skipping '
                       'processing of detections from that image' %
                       image_fname)
                continue

            xc, yc = np.mean(image_bbox, axis=0)

            lonlat_bbox = np.dot(h, np.array([[left, right, right, left],
                                              [top, top, bottom, bottom],
                                              [1, 1, 1, 1]]))
            lonlat_bbox = (lonlat_bbox[:2]/lonlat_bbox[2]).T

            # Calculate GSD
            lon_lats = np.dot(h, np.array([[xc, xc + 1, xc],
                                           [yc, yc, yc + 1],
                                           [1, 1, 1]]))
            lon_lats = lon_lats[:2]/lon_lats[2]
            dx = llh_to_enu(lon_lats[1, 1], lon_lats[0, 1], 0,
                            lon_lats[1, 0], lon_lats[0, 0], 0)
            dx = np.linalg.norm(dx)
            dy = llh_to_enu(lon_lats[1, 2], lon_lats[0, 2], 0,
                            lon_lats[1, 0], lon_lats[0, 0], 0)
            dy = np.linalg.norm(dy)

            if False:
                lonlat_bbox
                plt.plot(lon_lats[0], lon_lats[1])
                plt.plot(lonlat_bbox[:, 0], lonlat_bbox[:, 1])

            # Calculate the diagonal of the bounding box in meters
            dxy = (np.diff(image_bbox, axis=0)*np.array([dx, dy])).ravel()
            width_meters, height_meters = dxy

            if max(dxy) < animal_min_meters:
                suppressed = True
                num_suppressed += 1
            elif min(dxy) > animal_max_meters:
                suppressed = True
                num_suppressed += 1
            else:
                suppressed = False

            det = Detection(uid=uid, image_fname=image_fname,
                            frame_id=frame_id, image_bbox=image_bbox,
                            lonlat_bbox=lonlat_bbox,
                            confidence=confidence, length=length,
                            confidence_pairs=confidence_pairs,
                            gsd=(dx, dy), height_meters=height_meters,
                            width_meters=width_meters,
                            suppressed=suppressed)

            if img_fname not in detections:
                detections[img_fname] = []

            detections[img_fname].append(det)
            num_dets += 1

            if img_fname not in img_fnames_set:
                img_fnames_set.add(img_fname)
                img_fnames.append(img_fname)

    print2('Suppressed %i out of %i detections due to constraints on the '
           'minimum and maximum size of the animal in meters' %
           (num_suppressed, num_dets))

    def get_image(fname):
        for fov in get_fov_dirs(flight_dir):
            base_dir = os.path.join(flight_dir, fov)
            fnames = glob.glob('%s/%s.tif' % (base_dir, fname))
            if len(fnames) > 0:
                img = cv2.imread(fnames[0])
                if img is not None:
                    return img

    # Track redundant detections.
    print2('Comparing detections between frames to identify redundant '
           'detections...')
    num_suppressed = 0
    img_fnames = sorted(img_fnames)

    # Sanity check.
    det_uid = set()
    for fname in detections:
        for det in detections[fname]:
            det_uid.add(det.uid)

    print('File initially had %i unique detection uid' % len(det_uid))

    # ------------------------------------------------------------------------
    print('Running detection tracker to identify redundant detections across '
          '%i images' % len(img_fnames))
    for i in range(len(img_fnames)):
        # Start with detections in this img_fnames[i] and find possible
        # redundancies in subsequent images.
        detsi = detections[img_fnames[i]]

        # Figure out the mapping between longitude/latitude displacements
        # and displacements in meters.
        lon, lat = np.mean(detsi[0].lonlat_bbox, axis=0)
        meters_per_lon, meters_per_lat = meters_per_lon_lat(lon, lat)

        for j in range(i + 1, min(i + 10, len(img_fnames))):
            detsj = detections[img_fnames[j]]

            if False:
                print2('Comparing %i detections from image %i/%i against '
                       '%i detections from nearby image %i' %
                       (len(detsi), i+1, len(img_fnames), len(detsj), j))

            # Check whether there is an available image-to-image fine-tuned
            # homography available.
            h_ij = None
            try:
                # Check if ther eis a homography that warps from image i to
                # image i.
                h_ij = img_to_img_homog['%s_to_%s' % (img_fnames[i],
                                                      img_fnames[j])]
            except KeyError:
                # Check if ther eis a homography that warps from image j to
                # image i.
                try:
                    h_ij = img_to_img_homog['%s_to_%s' % (img_fnames[j],
                                                          img_fnames[i])]
                    h_ij = np.linalg.inv(h_ij)
                except KeyError:
                    pass

            # We now have all the detections 'detsi' from frame i and all the
            # detections 'detsj' from frame j.
            dist = np.full((len(detsi), len(detsj)), 1e5)
            for i_, deti in enumerate(detsi):
                if deti.suppressed:
                    continue

                for j_, detj in enumerate(detsj):
                    if detj.suppressed:
                        continue

                    if h_ij is not None:
                        cxyi = np.mean(deti.image_bbox, axis=0)
                        cxyi = np.hstack([cxyi, 1])
                        cxyi_on_j = np.dot(h_ij, cxyi)
                        cxyi_on_j = cxyi_on_j[:2]/cxyi_on_j[2]
                        cxyj = np.mean(detj.image_bbox, axis=0)

                        # Distance between images in pixels.
                        d = np.linalg.norm(cxyi_on_j - cxyj)

                        d = d*np.mean(detj.gsd)

                        # If the distance is less than 2 meters, supress.
                        if d < 2:
                            dist[i_, j_] = d
                    else:
                        lon_lati = np.mean(deti.lonlat_bbox, axis=0)
                        lon_latj = np.mean(detj.lonlat_bbox, axis=0)

                        dxy = (lon_lati - lon_latj)
                        dxy = dxy*(meters_per_lon, meters_per_lat)
                        d = np.sqrt(dxy[0]**2 + dxy[1]**2)

                        if d < geo_registration_error:
                            dist[i_, j_] = d

            # dist[ii, jj] is a distance matrix for the estimated distance
            # between dets_i[ii] and dets_j[jj]. The value 1e5 represents too
            # far. It is possible that most of this distance matrix is too far.
            # Therefore, we remove rows and columns that don't have any values
            # less than 1e5 to produce the new distance matrix dist_. But, we
            # need to keep track of dist_[i_, j_] correspond to dist[ii, jj].

            maski = np.any(dist < 1e5, axis=1)
            dist_ = np.atleast_2d(dist[maski])
            maskj = np.any(dist < 1e5, axis=0)
            dist_ = np.atleast_2d(dist_[:, maskj])

            if len(dist_) == 0:
                continue

            # Indices mapping from the variables ending in _ back respect to
            # the original variables.
            inds_i = np.arange(len(detsi))[maski]
            inds_j = np.arange(len(detsj))[maskj]

            # Use Hungarian matching algorithm for optimal assignment.
            row_ind0, col_ind0 = linear_sum_assignment(dist_)

            # Convert back into indices that reference relative to the dets_i
            # and dets_j.
            row_ind = [inds_i[_] for _ in row_ind0]
            col_ind = [inds_j[_] for _ in col_ind0]

            if False:
                # Sanity check.
                dist[row_ind, col_ind]

            for i_, j_ in zip(row_ind, col_ind):
                if dist[i_, j_] >= 1e5:
                    continue

                num_suppressed += 1

                deti = detsi[i_]
                detj = detsj[j_]

                # We want deti and detj to share the same uid. The
                # safest thing to do is to assign deti.uid to detj.uid
                # because frame j comes after frame i, so if we always
                # propogate the earlier detection forward in time, this
                # should gaurentee a consistent chain if the detection
                # of the same entity occurs in multiple successive
                # frames.
                detj.uid = deti.uid

                if deti.confidence > detj.confidence:
                    detj.suppressed = True
                else:
                    deti.suppressed = True

    # Sanity check.
    det_uid = set()
    for fname in detections:
        for det in detections[fname]:
            det_uid.add(det.uid)

    print('Tracking reduced down to %i unique detection uid' % len(det_uid))

    print2('Suppressed %i out of %i detections overlapping detections' %
           (num_suppressed, num_dets))

    # ------------------------------------------------------------------------
    # Save new detection csv with suppresions indicated.
    os.path.splitext(os.path.split(detection_csv)[1])[0]
    det_csv_dir = '%s/processed_results/detection_csv' % flight_dir

    try:
        os.makedirs(det_csv_dir)
    except OSError:
        pass

    csv_fname_out = '%s/%s' % (det_csv_dir, os.path.split(detection_csv)[1])

    print2('Saving new detection csv: \'%s\'' % csv_fname_out)

    with open(csv_fname_out, 'w') as out_file:
        out_file.write('# 1: Detection or Track-id,  2: Video or Image '
                       'Identifier,  3: Unique Frame Identifier,  4-7: '
                       'Img-bbox(TL_x,TL_y,BR_x,BR_y),  8: Detection '
                       'confidence,  9: Fish Length (0 or -1 if invalid),  '
                       '10-11+: Repeated Species, Confidence Pairs\n')

        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        what_wrote = 'detection summary'
        out_file.write('# Written on: %s   by: %s\n' % (date_time, what_wrote))

        for i in range(len(img_fnames)):
            for det in detections[img_fnames[i]]:
                (left, top), (right, bottom) = det.image_bbox
                out_file.write('%s,%s,%s,%i,%i,%i,%i,%s,%0.5f,%s\n' %
                               (det.uid, det.image_fname, det.frame_id,
                                int(left), int(top), int(right), int(bottom),
                                det.confidence, max([det.height_meters,
                                                     det.width_meters]),
                                ','.join(det.confidence_pairs)))


    # ------------------------------------------------------------------------
    # Save shapefile of detections.
    shapefile_dir = ('%s/processed_results/detection_shapefiles' % flight_dir)

    try:
        os.makedirs(shapefile_dir)
    except OSError:
        pass

    fname_base = os.path.splitext(os.path.split(detection_csv)[1])[0]

    # Find maximum length of all required fields.
    len_conf_pairs = 0
    len_img_filename = 0
    len_img_frame_id = 0
    len_track_id = 0
    for i in range(len(img_fnames)):
        for det in detections[img_fnames[i]]:
            len_conf_pairs = max([len(', '.join(det.confidence_pairs)),
                                  len_conf_pairs])
            len_img_filename = max([len_img_filename, len(det.image_fname)])
            len_img_frame_id = max([len_img_frame_id, len(str(det.frame_id))])
            len_track_id = max([len_track_id, len(str(det.uid))])

    print2('Saving detection shapefiles')
    w = shapefile.Writer('%s/%s.shp' % (shapefile_dir, fname_base),
                         shapeType=shapefile.POLYGON)
    w.autoBalance = 1
    w.field('img_filename', 'C', size=len_img_filename)
    w.field('frame_id', 'C', size=len_img_frame_id)
    w.field('track_id', 'C', size=len_track_id)
    w.field('img_left', 'N', decimal=2)
    w.field('img_right', 'N', decimal=2)
    w.field('img_top', 'N', decimal=2)
    w.field('img_bottom', 'N', decimal=2)
    w.field('confidence', 'N', decimal=10)
    w.field('length', 'N', decimal=10)
    w.field('conf_pairs', 'C', size=len_conf_pairs)
    w.field('gsd_m', 'N', decimal=5)
    w.field('height_m', 'N', decimal=5)
    w.field('width_m', 'N', decimal=5)
    w.field('latitude', 'N', decimal=7)
    w.field('longitude', 'N', decimal=7)
    w.field('suppressed', 'C', decimal=5)

    for i in range(len(img_fnames)):
        for det in detections[img_fnames[i]]:
            confidence_pairs = ', '.join(det.confidence_pairs)
            lats = det.lonlat_bbox[:, 1]
            lons = det.lonlat_bbox[:, 0]
            lls = zip(lons, lats)
            lls = list(lls)
            lls.append(lls[0])

            w.poly([lls])
            xl = int(det.image_bbox[0, 0])
            xr = int(det.image_bbox[1, 0])
            yt = int(det.image_bbox[0, 1])
            yb = int(det.image_bbox[1, 1])
            w.record(det.image_fname,       # img_filename
                     det.frame_id,          # frame_id
                     det.uid,               # track_id
                     det.image_bbox[0, 0],  # image_left
                     det.image_bbox[1, 0],  # image_right
                     det.image_bbox[0, 1],  # image_top
                     det.image_bbox[1, 1],  # image_bottom
                     det.confidence,        # confidence
                     det.length,            # length
                     confidence_pairs,      # confidence_pairs
                     np.mean(det.gsd),      # gsd
                     det.height_meters,     # height_meters
                     det.width_meters,      # width_meters
                     np.mean(lats),         # latitude
                     np.mean(lons),         # longitude
                     str(det.suppressed).lower())

    w.close()

    with open('%s/%s.prj' % (shapefile_dir, fname_base), "w") as fo:
        fo.write(wgs84_wkt)

    # ------------------------------------------------------------------------
    if save_annotations:
        print2('Saving images with detections annotations superimposed')
        save_dir = ('%s/processed_results/annotated_detections' % flight_dir)

        try:
            os.makedirs(save_dir)
        except OSError:
            pass

        for i in range(len(img_fnames)):
            img = get_image(img_fnames[i])
            if img is None:
                continue
            for det in detections[img_fnames[i]]:
                xl = int(det.image_bbox[0, 0])
                xr = int(det.image_bbox[1, 0])
                yt = int(det.image_bbox[0, 1])
                yb = int(det.image_bbox[1, 1])

                if det.suppressed:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(img, (xl, yt), (xr, yb), color=color,
                              thickness=2)

            fname = '%s/%s.jpg' % (save_dir, img_fnames[i])
            print2('Saving \'%s\'' % fname)
            cv2.imwrite(fname, img[:, :, ::-1])

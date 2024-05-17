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
from math import sqrt

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Repository imports.
from colmap_processing.image_renderer import stitch_images
from colmap_processing.platform_pose import PlatformPoseFixed
from colmap_processing.geo_conversions import enu_to_llh, llh_to_enu
import colmap_processing.dp as dp
from colmap_processing.rotations import euler_from_quaternion, \
    quaternion_multiply, quaternion_matrix, quaternion_from_euler, \
    quaternion_inverse, quaternion_from_matrix


def to_str(v):
    """Convert numerical values (scalar or float) to string for saving to yaml

    """
    if hasattr(v,  "__len__"):
        if len(v) > 1:
            return repr(list(v))
        else:
            v = v[0]

    return repr(v)


class CamToCamTform(object):
    """Model to transform image coordinates from one camera to another.

    This model assumes that both cameras are rigidly mounted relative to each
    other. If the platform that both cameras are mounted to is moving relative
    to the world, then the time associated with the source and destination
    image coordinates are assumed to be equivalent such that the navigation
    state associated with each is equivalent.

    Additionally, the distance between the centers of projection of the cameras
    is assumed to be small relative to the distance of the nearest object in
    the view such that we can ignore parallax during transformation.

    """
    def __init__(self, src_cm, dst_cm):
        if src_cm.platform_pose_provider != dst_cm.platform_pose_provider and   \
           not isinstance(src_cm.platform_pose_provider, PlatformPoseFixed) and \
           not isinstance(dst_cm.platform_pose_provider, PlatformPoseFixed):
            raise Exception('src_cm and dst_cm must have the same '
                            'platform_pose_provider indicating that the cameras '
                            'are rigidly mounted to the same platform')

        self._src_cm = src_cm
        self._dst_cm = dst_cm

    def fit(self, tol=0.1, k=1):
        """
        :param tol: Accuracy of the transformation (pixels).
        :type tol: float

        :param k: Degree of the bivariate spline.
        :type k: int

        """
        w = self._src_cm.width
        h = self._src_cm.height

        # TODO: we can do something fancier here, but for now just hard code
        # the number of tiles.
        N = 10
        while True:
            dx = np.sqrt(w*h/N)
            x = np.linspace(0, w, int(np.ceil(w/dx)))
            y = np.linspace(0, h, int(np.ceil(h/dx)))
            X,Y = np.meshgrid(x, y)
            points = np.vstack([X.ravel(),Y.ravel()])

            out_points = self.tform_rigorous(points)

            out_x = np.reshape(out_points[0], X.shape)
            out_y = np.reshape(out_points[1], Y.shape)

            self._model_x = RectBivariateSpline(x, y, out_x.T, kx=k, ky=k)
            self._model_y = RectBivariateSpline(x, y, out_y.T, kx=k, ky=k)

            # Test
            x = np.linspace(0, w, int(np.ceil(w/dx))*2)
            y = np.linspace(0, h, int(np.ceil(h/dx))*2)
            X,Y = np.meshgrid(x, y)
            points = np.vstack([X.ravel(),Y.ravel()])

            points_out = self.tform(points)
            points_out_truth = self.tform_rigorous(points)
            err = np.sqrt(np.sum((points_out_truth - points_out)**2, 0))
            if np.max(err) < tol or N > 2000:
                break

            N *= 2

    def tform(self, points):
        """
        :param points: Coordinates of a point or points within the image
            coordinate system. The coordinate may be Cartesian or homogenous.
        :type points: array with shape (2), (2,N), (3) or (3,N)

        :return: Image coordinates in the destination camera coordinate system
            associated with the source camera `points`.
        :rtype: numpy.ndarray of size (2,n)

        """
        if not hasattr(self, '_model_x'):
            raise Exception('Must call \'fit\' before calling \'tform\'')

        out_points = np.zeros_like(points)
        out_points[0] = self._model_x.ev(points[0], points[1])
        out_points[1] = self._model_y.ev(points[0], points[1])
        return out_points

    def tform_rigorous(self, points):
        """Rigorously accurate transformation.

        :param points: Coordinates of a point or points within the source
        camera image coordinate system. The coordinate may be Cartesian or
        homogenous.
        :type points: array with shape (2), (2,N), (3) or (3,N)

        :return: Image coordinates in the destination camera coordinate system
            associated with the source camera `points`.
        :rtype: numpy.ndarray of size (2,n)

        """
        ray_pos, ray_dir = self._src_cm.unproject(points, -np.inf)

        # We don't have a world model to intersect with, so we send it out
        # to "infinity".
        point = (ray_pos + ray_dir*1e5)
        return self._dst_cm.project(point, -np.inf)


def rt_from_quat_pos(position, quaternion):
    """Return rotation plus translation transformation matrix.

    Given the position and orientation of a coordinate system relative to a
    reference coordinate system, this function returns [R|T] padded to a 4x4.

    :param position: Position of the moving coordinate system within the
        reference coordinate system.
    :type position: 3-array

    :param quaternion: Quaternion (x, y, z, w) specifying the orientation of the
        moving coordinate system relative to the reference coordinate system.
        The quaternion represents a coordinate system rotation that takes the
        reference coordinate system and rotates it into the moving coordinate
        system.
    :type quaternion: 4-array

    :return: A 4x4 matrix that accepts a homogeneous 4-vector defining a 3-D
        point in the reference coordinate system and returns a homogeneous
        4-vector in the moving coordinate system pointing from the origin to
        the point.
    :rtype: 4x4 array

    """
    # The ROS convention for the rotation is that it transforms the
    # coordinate system. The computer vision community convention is that
    # the operator transforms a vector to get it into the moving coordinate
    # system. So, we invert each quaternion.
    quaternion = quaternion_inverse(quaternion)

    p = quaternion_matrix(quaternion)        # R
    p[:3,3] = -np.dot(p[:3,:3], position)    # T
    return p


def load_from_file(filename, platform_pose_provider=None):
    """Load from configuration yaml for any of the Camera subclasses.

    """
    with open(filename, 'r') as f:
        calib = yaml.safe_load(f)

    if calib['model_type'] == 'standard':
        return StandardCamera.load_from_file(filename, platform_pose_provider)

    if calib['model_type'] == 'rolling_shutter':
        return RollingShutterCamera.load_from_file(filename, platform_pose_provider)

    if calib['model_type'] == 'depth':
        return DepthCamera.load_from_file(filename, platform_pose_provider)

    if calib['model_type'] == 'static':
        return GeoStaticCamera.load_from_file(filename, platform_pose_provider)

    if calib['model_type'] == 'azel':
        return AzelCamera.load_from_file(filename, platform_pose_provider)

    raise Exception()


def ray_intersect_plane(plane_point, plane_normal, ray_pos, ray_dir,
                        epsilon=1e-6):
    """From https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python

    :param ray_pos: Ray starting positions.
    :type ray_pos : array with shape ((3) or (3,N)

    :param ray_dir: Ray directions.
    :type ray_dir: array with shape ((3) or (3,N)

    """
    ndotu = np.dot(plane_normal, ray_dir)

    if ray_dir.ndim == 1:
        if abs(ndotu) < epsilon:
            return np.nan

        w = ray_pos - np.array(plane_point)
        si = -plane_normal.dot(w) / ndotu
        Psi = w + si * ray_dir + plane_point
        return Psi

    psi = np.zeros_like(ray_pos)
    psi[:, ndotu < epsilon] = np.nan

    plane_point = np.atleast_2d(plane_point)
    plane_point = np.reshape(plane_point, (3, 1))
    w = ray_pos - plane_point
    si = -np.dot(plane_normal, w) / ndotu
    Psi = w + si * ray_dir + plane_point
    return Psi


class Camera(object):
    """Base class for all imaging sensors.

    The imaging sensor is attached to a navigation coordinate system (i.e., the
    frame of the INS), which may move relative to the East/North/Up world
    coordinate system. The pose (position and orientation) of this navigation
    coordinate system within the ENU coordinate system is provided by the
    platform_pose_provider attribute, which is an instance of a subclass of
    nav_state.NavStateProvider.

    The Camera object captures all of the imaging properties of the sensor.
    Intrinsic and derived parameters can be queried, and projection operations
    (pixels to world coordinates and vice versa) are provided.

    Most operations require specification of time in order determine values of
    any time-varying parameters (e.g., navigation coordinate system state).

    """
    def __init__(self, width, height, platform_pose_provider=None):
        """
        :param width: Width of the image provided by the imaging sensor,
        :type width: int

        :param height: Height of the image provided by the imaging sensor,
        :type height: int

        :param platform_pose_provider: Object that returns the state of the
            navigation coordinate system as a function of time. If None is
            passed, the navigation coordinate system will always have
            its x-axis aligned with world y, its y-axis aligned with world x,
            and its z-axis pointing down (negative world z).
        :type platform_pose_provider: subclass of NavStateProvider

        """
        self._width = width
        self._height = height

        if platform_pose_provider is None:
            self._platform_pose_provider = PlatformPoseFixed()
        else:
            self._platform_pose_provider = platform_pose_provider

        self._depth_map = None

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = int(value)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = int(value)

    @property
    def depth_map(self):
        return self._depth_map

    @depth_map.setter
    def depth_map(self, value):
        self._depth_map = value

    @property
    def platform_pose_provider(self):
        """Instance of a subclass of NavStateProvider

        """
        return self._platform_pose_provider

    @platform_pose_provider.setter
    def platform_pose_provider(self, value):
        """Instance of a subclass of NavStateProvider

        """
        self._platform_pose_provider = value

    def __str__(self):
        string = [''.join(['image_width: ',repr(self._width),'\n'])]
        string.append(''.join(['image_height: ',repr(self._height),'\n']))
        string.append(''.join(['platform_pose_provider: ',
                               repr(self._platform_pose_provider)]))

        try:
            # Some time-dependent cameras may not have a queue of values.
            string.append(''.join(['\nifov: ',
                                   '({:.6g},{:.6g})'.format(*self.ifov(np.inf)),
                                   '\n']))
            string.append(''.join(['fov: ',
                                   '({:.6},{:.6},{:.6})'.format(*self.fov(np.inf))]))
        except:
            pass

        return ''.join(string)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def load_from_file(cls, filename, platform_pose_provider=None):
        raise NotImplementedError

    def save_to_file(filename):
        raise NotImplementedError

    def get_param_array(self, param_list):
        """Return set of parameters as array.

        :param ptype: Parameters.
        :type ptype: str or list of str

        :return: Pameters as an array.
        :rtype: numpy.ndarray

        """
        params = np.zeros(0)
        for param in param_list:
            params = np.hstack([params,getattr(self, param)])

        return params

    def set_param_array(self, param_list, params):
        """Return set of parameters as array.

        :param param_list: List of parameter names.
        :type param_list: list of str

        :param params: Pameters as an array.
        :type params: numpy.ndarray

        """
        ind = 0
        for param in param_list:
            p0 = getattr(self, param)
            if hasattr(p0, '__len__') and len(p0) > 1:
                setattr(self, param, params[ind:ind+len(p0)])
                ind += len(p0)
            else:
                setattr(self, param, params[ind])
                ind += 1

    def project(self, points, t=None):
        """Project world points into the image at a particular time.

        :param points: Coordinates of a point or points within the world
            coordinate system. The coordinate may be Cartesian or homogenous.
        :type points: array with shape (3), (4), (3,n), (4,n)

        :param t: Time at which to project the point(s) (time in seconds since
            Unix epoch).
        :type t: float

        :return: Image coordinates associated with points.
        :rtype: numpy.ndarray of size (2,n)

        """
        raise NotImplementedError

    def unproject(self, points, t=None, normalize_ray_dir=True):
        """Unproject image points into the world at a particular time.

        :param points: Coordinates of a point or points within the image
            coordinate system. The coordinate may be Cartesian or homogenous.
        :type points: array with shape (2), (2,N), (3) or (3,N)

        :param t: Time at which to unproject the point(s) (time in seconds
            since Unix epoch).
        :type t: float

        :param normalize_dir: If set to True, the ray directions will be
            normalized to a unit a lenght. If set to False, the projection of
            the ray direction vectors onto the optical axis of the camera
            (the z-axis of the camera coordinate system) will have unit
            magnitude, useful when a depth-map projection distance is to be
            applied.
        :type normalize_dir: bool

        :return: Ray position and direction corresponding to provided image
            points. The direction points from the center of projection to the
            points. The direction vectors are not necassarily normalized.
        :rtype: [ray_pos, ray_dir] where both of type numpy.ndarry with shape
            (3,n).

        """
        raise NotImplementedError

    def unproject_to_depth(self, points, t=None):
        """Unproject image points into the world at a particular time.

        :param points: Coordinates of a point or points within the image
            coordinate system. The coordinate may be Cartesian or homogenous.
        :type points: array with shape (2), (2,N), (3) or (3,N)

        :param t: Time at which to unproject the point(s) (time in seconds
            since Unix epoch).
        :type t: float

        :return: Position of each ray's after project to its depth (i.e.,
            intersection with a world surface).
        :rtype: Numpy.ndarry with shape (3,n).

        """
        raise NotImplementedError

    def unproject_to_llh(self, points, t=None, cov=None):
        """Get latitude, longitude, height associated with image coordinate(s).

        :param points: Coordinates of a point or points within the image
            coordinate system.
        :type points: array with shape (2), (2,N)

        :param t: Time at which to query the camera's pose (time in seconds
            since Unix epoch).
        :type t: float | None

        :param cov: List of 2x2 covariance matrices indicating the
            positional uncertainty, in image space, of the points.
        :type cov: list of 2x2 array | None

        :return: Latitude (degrees), longitude (degrees), and height above
            WGS84 ellipsoid (meters). If 'points_cov' is not None, the second
            element of the return is a list of 3x3 arrays, one for each output
            point, indicating the localization uncertainty in a local-tangent
            east, north, up coordinate system centered at the point.
        :rtype: numpy.ndarray of size (2,n) and (optionally) a list of 3x3
            arrays

        """
        if cov is not None and not isinstance(cov, (list, tuple)):
            raise TypeError()

        lat0 = self.platform_pose_provider.lat0
        lon0 = self.platform_pose_provider.lon0
        h0 = self.platform_pose_provider.h0

        if lat0 is None or lon0 is None or h0 is None:
            raise Exception('\'platform_pose_provider\' must have \'lat0\', '
                            '\'lon0\', and \'ho\' defined.')

        points = np.array(points)
        if points.ndim == 1:
            was_1d = True
            points = np.atleast_2d(points).T
        else:
            was_1d = False
            points = np.array(points)
            points = np.reshape(points, (2,-1))

        llh = []
        geo_cov = []

        for i in range(points.shape[1]):
            xyz = self.unproject_to_depth(points[:,i], t)
            if np.all(np.isfinite(xyz)):
                llh.append(enu_to_llh(xyz[0], xyz[1], xyz[2], lat0, lon0, h0))
            else:
                llh.append(np.full(3, np.nan))

            if cov is not None:
                # Sample 10 random points and project each into enu coordinate
                # system
                rpoints = np.random.multivariate_normal(points[:,i], cov[i],
                                                        20)

                # Points must be inside image.
                ind = np.logical_and(rpoints[:,0] > 0, rpoints[:,1] > 0)
                ind = np.logical_and(ind, rpoints[:,0] < self.width)
                ind = np.logical_and(ind, rpoints[:,1] < self.height)
                rpoints = rpoints[ind]
                enu_pts = ([self.unproject_to_depth(_, t).ravel()
                             for _ in rpoints])

                enu_pts = [_ for _ in enu_pts if np.all(np.isfinite(enu_pts))]

                if len(enu_pts) < 2:
                    geo_cov.append(None)
                    continue

                enu_pts = np.array(enu_pts)

                if xyz[0]**2 + xyz[1]**2 > 6250000:
                    # If the point is further than 2.5km from the camera, we
                    # want the covariance defined in an east/north/up
                    # coordinate system centered at xyz, the most-likely
                    # location of the entity.

                    # TODO, directly calculate the rotation matrix that can be
                    # applied to the covariance matrix.
                    llh0 = enu_to_llh(xyz[0], xyz[1], xyz[2], lat0, lon0, h0)

                    for i in range(len(enu_pts)):
                        llhi = enu_to_llh(enu_pts[i,0], enu_pts[i,1],
                                          enu_pts[i,2], llh0[0], llh0[1],
                                          llh0[2])
                        enu_pts[i,:] = llh_to_enu(llhi[0], llhi[1], llhi[2],
                                                  llh0[0], llh0[1], llh0[2])


                geo_cov.append(np.cov(enu_pts.T))

        if was_1d:
            llh = llh[0]

        llh = np.array(llh).T

        if cov is not None:
            return llh,geo_cov
        else:
            return llh

    def points_along_image_border(self, num_points=4):
        """Return uniform array of points along the perimeter of the image.

        :param num_points: Number of points (approximately) to distribute
            around the perimeter of the image perimeter.
        :type num_points: int

        :return:
        :rtype: numpy.ndarry with shape (3,n)

        """
        perimeter = 2*(self.height + self.width)
        ds = num_points/float(perimeter)
        xn = np.max([2,int(ds*self.width)])
        yn = np.max([2,int(ds*self.height)])
        x = np.linspace(0, self.width, xn)
        y = np.linspace(0, self.height, yn)[1:-1]
        pts = np.vstack([np.hstack([x,
                                    np.full(len(y), self.width,
                                            dtype=np.float64),
                                    x[::-1],
                                    np.zeros(len(y))]),
                         np.hstack([np.zeros(xn),
                                    y,
                                    np.full(xn, self.height,
                                            dtype=np.float64),
                                    y[::-1]])])
        return pts


    def ifov(self, t=None):
        """Instantaneous field of view (ifov) at the image center.

        ifov is the angular extend spanned by a single pixel.

        :param t: Time at which to calculate the ifov (time in seconds since
            Unix epoch). Only relevant for sensors where zoom can change.
        :type t: float

        :return: The ifov along the horizontal and vertical directions of the
            image, evaluated at the center of the image (radians).

        """
        if t is None:
            t = time.time()

        cx = self.width/2
        cy = self.height/2
        ray1 = self.unproject([cx,cy], t)[1]
        ray1 /= np.sqrt(np.sum(ray1**2, 0))
        ray2 = self.unproject([cx,cy+1], t)[1]
        ray2 /= np.sqrt(np.sum(ray2**2, 0))
        ray3 = self.unproject([cx+1,cy], t)[1]
        ray3 /= np.sqrt(np.sum(ray3**2, 0))

        ifovx = np.arccos(np.dot(ray1.ravel(), ray3.ravel()))
        ifovy = np.arccos(np.dot(ray1.ravel(), ray2.ravel()))

        return ifovx, ifovy

    def fov(self, t=None):
        """Field of view (fov).

        :param t: Time at which to calculate the ifov (time in seconds since
            Unix epoch). Only relevant for sensors where zoom can change.
        :type t: float

        :return: The horizontal, vertical, and diagonal field of view of the
            camera (degrees).
        :rtype: tuple of (ifov_h, ifov_v, ifov_d)

        """
        if t is None:
            t = time.time()

        cx = self.width/2
        cy = self.height/2

        ray1 = self.unproject([cx,0], t)[1]
        ray1 /= np.sqrt(np.sum(ray1**2, 0))
        ray2 = self.unproject([cx,self.height], t)[1]
        ray2 /= np.sqrt(np.sum(ray2**2, 0))
        fov_v = np.arccos(np.dot(ray1.ravel(), ray2.ravel()))*180/np.pi

        ray1 = self.unproject([0,cy], t)[1]
        ray1 /= np.sqrt(np.sum(ray1**2, 0))
        ray2 = self.unproject([self.width, cy], t)[1]
        ray2 /= np.sqrt(np.sum(ray2**2, 0))
        fov_h = np.arccos(np.dot(ray1.ravel(), ray2.ravel()))*180/np.pi

        ray1 = self.unproject([0,0], t)[1]
        ray1 /= np.sqrt(np.sum(ray1**2, 0))
        ray2 = self.unproject([self.width,self.height], t)[1]
        ray2 /= np.sqrt(np.sum(ray2**2, 0))
        fov_d = np.arccos(np.dot(ray1.ravel(), ray2.ravel()))*180/np.pi

        return fov_h, fov_v, fov_d

    def add_image_to_list(self, raw_image, t=None):
        """Add image to image buffer.

        """
        if t is None:
            t = time.time()

        if hasattr(self, '_buffer_size'):
            buffer_size = self._buffer_size
        else:
            buffer_size = np.inf

        with lock:
            # Make sure list will not exceed buffer size after addition.
            while len(self.images) >= buffer_size:
                self.images.pop(0)
                self._image_times = np.delete(self.image_times, 0)

            self._images.append(raw_image)
            self._image_times = np.hstack([self._image_times,t])

    def unproject_to_depth(self, points, t=None):
        """See Camera.unproject_to_depth documentation.

        """
        points = self._unproject_to_depth(points, self.depth_map, t=t)
        return points

    def save_depth_viz(self, fname):
        depth_image = self.depth_map.copy()
        v = depth_image[np.isfinite(depth_image)]
        if len(v) > 0:
            vmin = np.percentile(v, 1)
            vmax = np.percentile(v, 99)
            depth_image -= vmin
            depth_image[depth_image < 0] = 0
            v = vmax - vmin
            if v > 0:
                depth_image /= v/255

        depth_image = np.round(depth_image).astype(np.uint8)

        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        cv2.imwrite(fname, depth_image[:, :, ::-1])


class StandardCamera(Camera):
    """Standard camera model.

    This is a model for a camera that is rigidly mounted to the navigation
    coordinate system. The camera model specification follows that of Opencv.

    See addition parameter definitions in base class Camera.

    :param K: Camera intrinsic matrix.
    :type K: 3x3 numpy.ndarray | None

    :param cam_pos: Position of the camera's center of projection in the
        navigation coordinate system.
    :type cam_pos: numpy.ndarray | None

    :param cam_quat: Quaternion (x, y, z, w) specifying the orientation of the
        camera relative to the navigation coordinate system. The quaternion
        represents a coordinate system rotation that takes the navigation
        coordinate system and rotates it into the camera coordinate system.
    :type cam_quat: numpy.ndarray | None

    :param dist: Input vector of distortion coefficients (k1, k2, p1, p2, k3,
        k4, k5, k6) of 4, 5, or 8 elements.
    :type dist: numpy.ndarray

    """
    def __init__(self, width, height, K, dist, cam_pos, cam_quat,
                 platform_pose_provider=None):
        """
        See additional documentation from base class above.

        """
        super(StandardCamera, self).__init__(width, height,
              platform_pose_provider)

        self._K = np.array(K, dtype=np.float64)
        self._dist = np.atleast_1d(dist).astype(np.float32)
        self._cam_pos = np.array(cam_pos, dtype=np.float64)
        self._cam_quat = np.array(cam_quat, dtype=np.float64)
        self._cam_quat /= np.linalg.norm(self._cam_quat)
        self._min_ray_cos = None

    def __str__(self):
        string = ['model_type: standard\n']
        string.append(super(StandardCamera, self).__str__())
        string.append('\n')
        string.append(''.join(['fx: ',repr(self._K[0,0]),'\n']))
        string.append(''.join(['fy: ',repr(self._K[1,1]),'\n']))
        string.append(''.join(['cx: ',repr(self._K[0,2]),'\n']))
        string.append(''.join(['cy: ',repr(self._K[1,2]),'\n']))
        string.append(''.join(['distortion_coefficients: ',
                               repr(tuple(self._dist)),
                               '\n']))
        string.append(''.join(['camera_quaternion: ',
                               repr(tuple(self._cam_quat)),'\n']))
        string.append(''.join(['camera_position: ',repr(tuple(self._cam_pos)),
                               '\n']))
        return ''.join(string)

    @classmethod
    def load_from_file(cls, filename, platform_pose_provider=None):
        """See base class Camera documentation.

        """
        with open(filename, 'r') as f:
            calib = yaml.safe_load(f)

        assert calib['model_type'] == 'standard'

        # fill in CameraInfo fields
        width = calib['image_width']
        height = calib['image_height']
        dist = calib['distortion_coefficients']

        if dist == 'None':
            dist = np.zeros(4)

        fx = calib['fx']
        fy = calib['fy']
        cx = calib['cx']
        cy = calib['cy']
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

        cam_quat = calib['camera_quaternion']
        cam_pos = calib['camera_position']

        return cls(width, height, K, dist, cam_pos, cam_quat,
                   platform_pose_provider)

    @classmethod
    def load_from_krtd(cls, filename):
        """See base class Camera documentation.

        """
        data = []
        with open(filename) as f:
            for line in f.readlines():
                data.append(line.strip('\n'))

        fx = float(data[0].split(' ' )[0])
        fy = float(data[1].split(' ' )[1])
        cx = float(data[0].split(' ' )[2])
        cy = float(data[1].split(' ' )[2])

        R = np.zeros((3, 3))

        for i in range(3):
            R[i] = [float(d) for d in data[4 + i].split(' ')]

        tvec = [float(d) for d in data[8].split(' ')]

        cam_pos = -np.dot(R.T, tvec).ravel()

        cam_quat = quaternion_inverse(quaternion_from_matrix(R))

        # fill in CameraInfo fields
        width = None
        height = None

        dist = [float(d) for d in data[10].split(' ') if len(d) > 0]
        dist = np.array(dist)

        K = np.array([[fx, 0, cx], [0, fy, cy],[0, 0, 1]])

        return cls(width, height, K, dist, cam_pos, cam_quat)

    def save_to_file(self, filename):
        """See base class Camera documentation.

        """
        with open(filename, 'w') as f:
            f.write(''.join(['# The type of camera model.\n',
                             'model_type: standard\n\n',
                             '# Image dimensions\n']))

            f.write(''.join(['image_width: ',to_str(self.width),'\n']))
            f.write(''.join(['image_height: ',to_str(self.height),'\n\n']))

            f.write('# Focal length along the image\'s x-axis.\n')
            f.write(''.join(['fx: ',to_str(self.K[0,0]),'\n\n']))

            f.write('# Focal length along the image\'s y-axis.\n')
            f.write(''.join(['fy: ',to_str(self.K[1,1]),'\n\n']))

            f.write('# Principal point is located at (cx,cy).\n')
            f.write(''.join(['cx: ',to_str(self.K[0,2]),'\n']))
            f.write(''.join(['cy: ',to_str(self.K[1,2]),'\n\n']))

            f.write(''.join(['# Distortion coefficients following OpenCv\'s ',
                    'convention\n']))

            dist = self.dist
            if np.all(dist == 0):
                dist = 'None'

            f.write(''.join(['distortion_coefficients: ',
                             to_str(self.dist),'\n\n']))

            f.write(''.join(['# Quaternion (x, y, z, w) specifying the ',
                             'orientation of the camera relative to\n# the ',
                             'platform coordinate system. The quaternion ',
                             'represents a coordinate\n# system rotation that ',
                             'takes the platform coordinate system and ',
                             'rotates it\n# into the camera coordinate ',
                             'system.\ncamera_quaternion: ',
                             to_str(self.cam_quat),'\n\n']))

            f.write(''.join(['# Position of the camera\'s center of ',
                             'projection within the navigation\n# coordinate ',
                             'system.\n',
                             'camera_position: ',to_str(self.cam_pos),
                             '\n\n']))

    @property
    def K(self):
        return self._K

    @property
    def K_no_skew(self):
        """Returns a compact version of K assuming no skew.

        """
        K = self.K
        return np.array([K[0,0],K[1,1],K[0,2],K[1,2]])

    @K_no_skew.setter
    def K_no_skew(self, value):
        """fx, fy, cx, cy
        """
        K = np.zeros((3,3), dtype=np.float64)
        K[0,0] = value[0]
        K[1,1] = value[1]
        K[0,2] = value[2]
        K[1,2] = value[3]
        self._K = K
        self._min_ray_cos = None

    @property
    def focal_length(self):
        return self._K[0,0]

    @focal_length.setter
    def focal_length(self, value):
        self._K[0,0] = value
        self._K[1,1] = value
        self._min_ray_cos = None

    @property
    def fx(self):
        return self._K[0,0]

    @property
    def fy(self):
        return self._K[1,1]

    @fx.setter
    def fx(self, value):
        self._K[0,0] = value
        self._min_ray_cos = None

    @fy.setter
    def fy(self, value):
        self._K[1,1] = value
        self._min_ray_cos = None

    @property
    def cx(self):
        return self._K[0,2]

    @property
    def cy(self):
        return self._K[1,2]

    @cx.setter
    def cx(self, value):
        self._K[0,2] = value
        self._min_ray_cos = None

    @cy.setter
    def cy(self, value):
        self._K[1,2] = value
        self._min_ray_cos = None

    @property
    def aspect_ratio(self):
        return self._K[0,0]/self._K[1,1]

    @aspect_ratio.setter
    def aspect_ratio(self, value):
        self._K[1,1] = self._K[0,0]*value

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, value):
        if value is None or value is 0:
            value = np.zeros(4)

        self._dist = np.atleast_1d(value)
        self._min_ray_cos = None

    @property
    def cam_pos(self):
        return self._cam_pos

    @property
    def cam_quat(self):
        return self._cam_quat

    @cam_quat.setter
    def cam_quat(self, value):
        self._cam_quat = np.atleast_1d(value)
        self._cam_quat /= np.linalg.norm(self._cam_quat)

    @property
    def min_ray_cos(self):
        """Minimum cosine of ray angle with optical axis valid for 'project'.

        For various sets of parameters for radial distortion, the function
        might only be invertible over a limited range of ray angles.
        """
        if self._min_ray_cos is not None:
            return self._min_ray_cos

        if np.all(self.dist == 0):
            self._min_ray_cos = 0
            return self._min_ray_cos

        # Make sure we aren't seeing behind the camera.
        t = time.time()
        center = self.cx, self.cy
        ray0 = self.unproject(center, t, normalize_ray_dir=True)[1].ravel()
        w, h = self.width, self.height
        self._min_ray_cos = 1
        for x,y in [[0,0],[w,0],[w,h],[0,h]]:
            ray1 = self.unproject([x,  y], t, normalize_ray_dir=True)[1]
            ray1 = ray1.ravel()
            ray_cosi = np.dot(ray0, ray1)
            self._min_ray_cos = np.minimum(self._min_ray_cos, ray_cosi)

        self._min_ray_cos -= 1e-8

        return self._min_ray_cos

    def update_intrinsics(self, K=None, cam_quat=None, dist=None):
        """
        """
        if K is not None:
            self._K = K.astype(np.float64)
        if cam_quat is not None:
            self._cam_quat = cam_quat
        if dist is not None:
            self._dist = dist

    def get_camera_pose(self, t=None):
        """Returns 3x4 matrix mapping world points to camera vectors.

        :param t: Time at which to query the camera's pose (time in seconds
            since Unix epoch).

        :return: A 3x4 matrix that accepts a homogeneous 4-vector defining a
            3-D point in the world and returns a Cartesian 3-vector in the
            camera's coordinate system pointing from the camera's center of
            projection to the word point (i.e., the negative of the principal
            ray coming from this world point).
        :rtype: 3x4 array

        """
        ins_pos, ins_quat = self.platform_pose_provider.pose(t)

        cam_pos = self._cam_pos
        cam_quat = self._cam_quat

        p_ins = rt_from_quat_pos(ins_pos, ins_quat)
        p_cam = rt_from_quat_pos(cam_pos, cam_quat)

        return np.dot(p_cam, p_ins)[:3]

    def project(self, points, t=None):
        """See Camera.project documentation.

        """
        points = np.array(points, dtype=np.float64)
        if points.ndim == 1:
            points = np.atleast_2d(points).T

        if t is None:
            t = time.time()

        pose_mat = self.get_camera_pose(t)

        # Project rays into camera coordinate system.
        rvec = cv2.Rodrigues(pose_mat[:3,:3])[0].ravel()
        tvec = pose_mat[:,3]
        im_pts = cv2.projectPoints(points.T, rvec, tvec, self._K,
                                   self._dist)[0]
        im_pts = np.squeeze(im_pts, 1).T

        # Make homogeneous
        points = np.vstack([points, np.ones(points.shape[1])])
        points = np.dot(pose_mat, points)
        points /= np.sqrt(np.sum(points**2, 0))

        # Add the 1e-8 to avoid "falling off the focal plane" due to rounding
        # error.
        ind = points[2] <= self.min_ray_cos

        im_pts[:, ind] = -100000

        return im_pts

    def unproject(self, points, t=None, normalize_ray_dir=True):
        """See Camera.unproject documentation.

        """
        points = np.array(points, dtype=np.float64)
        if points.ndim == 1:
            points = np.atleast_2d(points).T
            points = np.reshape(points, (2,-1))

        if t is None:
            t = time.time()

        ins_pos, ins_quat = self.platform_pose_provider.pose(t)
        #print('ins_pos', ins_pos)
        #print('ins_quat', ins_quat)

        # Unproject rays into the camera coordinate system.
        ray_dir = np.ones((3,points.shape[1]), dtype=points.dtype)
        ray_dir0 = cv2.undistortPoints(np.expand_dims(points.T, 1),
                                       self._K, self._dist, R=None)
        ray_dir[:2] = np.squeeze(ray_dir0, 1).T

        # Rotate rays into the navigation coordinate system.
        ray_dir = np.dot(quaternion_matrix(self._cam_quat)[:3,:3], ray_dir)

        # Translate ray positions into their navigation coordinate system
        # definition.
        ray_pos = np.zeros_like(ray_dir)
        ray_pos[0] = self._cam_pos[0]
        ray_pos[1] = self._cam_pos[1]
        ray_pos[2] = self._cam_pos[2]

        # Rotate and translate rays into the world coordinate system.
        R_ins_to_world = quaternion_matrix(ins_quat)[:3,:3]
        ray_dir = np.dot(R_ins_to_world, ray_dir)
        ray_pos = np.dot(R_ins_to_world, ray_pos) + np.atleast_2d(ins_pos).T

        if normalize_ray_dir:
            # Normalize
            ray_dir /= np.sqrt(np.sum(ray_dir**2, 0))

        return ray_pos, ray_dir


class RollingShutterCamera(StandardCamera):
    """Standard camera model.

    This is a model for a camera that is rigidly mounted to the navigation
    coordinate system. The camera model specification follows that of Opencv.

    See addition parameter definitions in base class Camera.

    :param K: Camera intrinsic matrix.
    :type K: 3x3 numpy.ndarray | None

    :param cam_pos: Position of the camera's center of projection in the
        navigation coordinate system.
    :type cam_pos: numpy.ndarray | None

    :param cam_quat: Quaternion (x, y, z, w) specifying the orientation of the
        camera relative to the navigation coordinate system. The quaternion
        represents a coordinate system rotation that takes the navigation
        coordinate system and rotates it into the camera coordinate system.
    :type cam_quat: numpy.ndarray | None

    :param dist: Input vector of distortion coefficients (k1, k2, p1, p2, k3,
        k4, k5, k6) of 4, 5, or 8 elements.
    :type dist: numpy.ndarray

    """
    def __init__(self, width, height, K, dist, cam_pos, cam_quat,
                 shutter_roll_time, platform_pose_provider=None):
        """
        See additional documentation from base class above.

        """
        super(RollingShutterCamera, self).__init__(width, height, K, dist,
                                                   cam_pos, cam_quat,
                                                   platform_pose_provider)
        self.shutter_roll_time = shutter_roll_time

    def __str__(self):
        string = ['model_type: rolling_shutter\n']
        string.append(super(RollingShutterCamera, self).__str__())
        string.append('shutter_roll_time: %s\n' %self.shutter_roll_time)
        return ''.join(string)

    @classmethod
    def load_from_file(cls, filename, platform_pose_provider=None):
        """See base class Camera documentation.

        """
        with open(filename, 'r') as f:
            calib = yaml.safe_load(f)

        assert calib['model_type'] == 'rolling_shutter'

        # fill in CameraInfo fields
        width = calib['image_width']
        height = calib['image_height']
        shutter_roll_time = calib['shutter_roll_time']
        dist = calib['distortion_coefficients']

        if dist == 'None':
            dist = np.zeros(4)

        fx = calib['fx']
        fy = calib['fy']
        cx = calib['cx']
        cy = calib['cy']
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

        cam_quat = calib['camera_quaternion']
        cam_pos = calib['camera_position']

        return cls(width, height, K, dist, cam_pos, cam_quat,
                   shutter_roll_time, platform_pose_provider)

    def save_to_file(self, filename):
        """See base class Camera documentation.

        """
        with open(filename, 'w') as f:
            f.write(''.join(['# The type of camera model.\n',
                             'model_type: rolling_shutter\n\n',
                             '# Image dimensions\n']))

            f.write(''.join(['image_width: ',to_str(self.width),'\n']))
            f.write(''.join(['image_height: ',to_str(self.height),'\n\n']))

            f.write(''.join(['shutter_roll_time: ',
                             to_str(self.shutter_roll_time),'\n\n']))

            f.write('# Focal length along the image\'s x-axis.\n')
            f.write(''.join(['fx: ',to_str(self.K[0,0]),'\n\n']))

            f.write('# Focal length along the image\'s y-axis.\n')
            f.write(''.join(['fy: ',to_str(self.K[1,1]),'\n\n']))

            f.write('# Principal point is located at (cx,cy).\n')
            f.write(''.join(['cx: ',to_str(self.K[0,2]),'\n']))
            f.write(''.join(['cy: ',to_str(self.K[1,2]),'\n\n']))

            f.write(''.join(['# Distortion coefficients following OpenCv\'s ',
                    'convention\n']))

            dist = self.dist
            if np.all(dist == 0):
                dist = 'None'

            f.write(''.join(['distortion_coefficients: ',
                             to_str(self.dist),'\n\n']))

            f.write(''.join(['# Quaternion (x, y, z, w) specifying the ',
                             'orientation of the camera relative to\n# the ',
                             'platform coordinate system. The quaternion ',
                             'represents a coordinate\n# system rotation that ',
                             'takes the platform coordinate system and ',
                             'rotates it\n# into the camera coordinate ',
                             'system.\ncamera_quaternion: ',
                             to_str(self.cam_quat),'\n\n']))

            f.write(''.join(['# Position of the camera\'s center of ',
                             'projection within the navigation\n# coordinate ',
                             'system.\n',
                             'camera_position: ',to_str(self.cam_pos),
                             '\n\n']))

    def project(self, points, t=None):
        """See Camera.project documentation.

        """
        # The challenge projecting into a rolling shutter camera is that every
        # row of the image is exposed at a different time. So, if you assume a
        # particular time to evaluate the pose at and then project into the
        # image, the y coordinate of that result may indicate a different time.
        # So, you must search for the time such that the y-coordinate of the
        # projection agrees with the time.
        proj_fun = super(RollingShutterCamera, self).project

        # We start by projecting assuming all points are at the time associated
        # with the center of the field of view.
        im_pts = proj_fun(points, t + 0.5*self.shutter_roll_time)

        if False:
            # Slower but more accurate.
            L = im_pts.shape[1]

            # 'ind' encodes the indices into im_pts that need to be further
            # checked that refining the
            ind = np.ones(L, dtype=bool)

            cont = True
            for _ in range(10):
                cont = False
                for i in range(L):
                    if ind[i]:
                        # The fraction of the rolling shutter time this y
                        # coordinate has accumulated.
                        alpha = np.clip(im_pts[1, i]/self.height, 0, 1)
                        t_ = t + alpha*self.shutter_roll_time
                        im_pt_ = proj_fun(points[:, i:i+1], t_)
                        d = sqrt(np.sum((im_pt_ - im_pts[:, i:i+1])**2))

                        if d > 0.01:
                            cont = True
                        else:
                            ind[i] = False

                        im_pts[:, i:i+1] = im_pt_

                if not cont:
                    break
        else:
            N = 10
            alphas = np.linspace(0, 1, N)
            im_pts_list = [proj_fun(points, t + alpha*self.shutter_roll_time).T
                           for alpha in alphas]
            im_pts_list = np.array(im_pts_list).T
            alphas2 = np.clip(im_pts_list[1]/self.height, 0, 1)
            alpha_err = alphas2 - alphas

            # We want to interpolate to the zero-valued alpha error.
            # ind1 is the index of the smallest-magnitude alpha_error.
            # ind2 is the index of the second-smallest-magnitude alpha_error.
            ind = np.argpartition(np.abs(alpha_err), 2, axis=1)[:, :2]

            ind1 = ind[:, 0:1]
            ind2 = ind[:, 1:2]
            alpha_err1 = np.take_along_axis(alpha_err, ind1, axis=1).ravel()
            alpha_err2 = np.take_along_axis(alpha_err, ind2, axis=1).ravel()

            # Linearly interpolate the im_pts. w encodes the fraction of ind1
            # to take.
            # w*(alpha_err2 - alpha_err1) + alpha_err1 = 0
            # w*alpha_err2 + (1-w)*alpha_err1 = 0
            delta = alpha_err1 - alpha_err2
            w = np.ones(len(alpha_err1))
            ind = delta != 0
            w[ind] = (alpha_err1[ind])/delta[ind]

            im_pts1 = np.hstack([np.take_along_axis(im_pts_list[0], ind1, axis=1),
                                 np.take_along_axis(im_pts_list[1], ind1, axis=1)]).T

            im_pts2 = np.hstack([np.take_along_axis(im_pts_list[0], ind2, axis=1),
                                 np.take_along_axis(im_pts_list[1], ind2, axis=1)]).T

            im_pts = w*im_pts2 + (1-w)*im_pts1

        return im_pts

    def unproject(self, points, t, normalize_ray_dir=True):
        """See Camera.unproject documentation.

        """
        points = np.array(points, dtype=np.float64)
        if points.ndim == 1:
            points = np.atleast_2d(points).T
            points = np.reshape(points, (2,-1))

        alphas = np.clip(points[1]/self.height, 0, 1)
        ts_ = t + (alphas*self.shutter_roll_time).astype(np.float64)

        ret = [super(RollingShutterCamera, self).unproject(points[:, i:i+1], ts_[i])
               for i in range(len(ts_))]
        ray_pos = np.hstack([ret_[0] for ret_ in ret])
        ray_dir = np.hstack([ret_[1] for ret_ in ret])

        return ray_pos, ray_dir


class DepthCamera(StandardCamera):
    """Camera with depth map.

    """
    def __init__(self, width, height, K, dist, cam_pos, cam_quat, depth_map,
                 platform_pose_provider=None):
        """
        See additional documentation from base class above.

        """
        super(DepthCamera, self).__init__(width=width, height=height, K=K,
                                          dist=dist, cam_pos=cam_pos,
                                          cam_quat=cam_quat,
                                          platform_pose_provider=platform_pose_provider)
        self._depth_map = depth_map

    @classmethod
    def load_from_file(cls, filename, platform_pose_provider=None):
        """See base class Camera documentation.

        """
        with open(filename, 'r') as f:
            calib = yaml.safe_load(f)

        assert calib['model_type'] == 'depth'

        # fill in CameraInfo fields
        width = calib['image_width']
        height = calib['image_height']
        dist = calib['distortion_coefficients']

        if dist == 'None':
            dist = np.zeros(4)

        fx = calib['fx']
        fy = calib['fy']
        cx = calib['cx']
        cy = calib['cy']
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

        cam_quat = calib['camera_quaternion']
        cam_pos = calib['camera_position']

        return cls(width, height, K, dist, cam_pos, cam_quat,
                   platform_pose_provider)

    def save_to_file(self, filename, save_depth_viz=True):
        """See base class Camera documentation.

        """
        with open(filename, 'w') as f:
            f.write(''.join(['# The type of camera model.\n',
                             'model_type: depth\n\n',
                             '# Image dimensions\n']))

            f.write(''.join(['image_width: ',to_str(self.width),'\n']))
            f.write(''.join(['image_height: ',to_str(self.height),'\n\n']))

            f.write('# Focal length along the image\'s x-axis.\n')
            f.write(''.join(['fx: ',to_str(self.K[0,0]),'\n\n']))

            f.write('# Focal length along the image\'s y-axis.\n')
            f.write(''.join(['fy: ',to_str(self.K[1,1]),'\n\n']))

            f.write('# Principal point is located at (cx,cy).\n')
            f.write(''.join(['cx: ',to_str(self.K[0,2]),'\n']))
            f.write(''.join(['cy: ',to_str(self.K[1,2]),'\n\n']))

            f.write(''.join(['# Distortion coefficients following OpenCv\'s ',
                    'convention\n']))

            dist = self.dist
            if np.all(dist == 0):
                dist = 'None'

            f.write(''.join(['distortion_coefficients: ',
                             to_str(self.dist),'\n\n']))

            f.write(''.join(['# Quaternion (x, y, z, w) specifying the ',
                             'orientation of the camera relative to\n# the ',
                             'navigation coordinate system. The quaternion ',
                             'represents a coordinate\n# system rotation that ',
                             'takes the navigation coordinate system and ',
                             'rotates it\n# into the camera coordinate ',
                             'system.\n camera_quaternion: ',
                             to_str(self.cam_quat),'\n\n']))

            f.write(''.join(['# Position of the camera\'s center of ',
                             'projection within the navigation\n# coordinate ',
                             'system.\n',
                             'camera_position: ',to_str(self.cam_pos),
                             '\n\n']))

        if self.depth_map is not None:
            im = PIL.Image.fromarray(self.depth_map.astype(np.float32),
                                     mode='F') # float32
            depth_map_fname = '%s_depth_map.tif' % os.path.splitext(filename)[0]
            im.save(depth_map_fname)

            if save_depth_viz:
                depth_viz_fname = ('%s/depth_vizualization.png' %
                                   os.path.split(filename)[0])
                self.save_depth_viz(depth_viz_fname)

    def __str__(self):
        string = ['model_type: depth\n']
        string.append(super(DepthCamera, self).__str__())
        string.append('\n')
        string.append(''.join(['fx: ',repr(self._K[0,0]),'\n']))
        string.append(''.join(['fy: ',repr(self._K[1,1]),'\n']))
        string.append(''.join(['cx: ',repr(self._K[0,2]),'\n']))
        string.append(''.join(['cy: ',repr(self._K[1,2]),'\n']))
        string.append(''.join(['distortion_coefficients: ',
                               repr(tuple(self._dist)),
                               '\n']))
        string.append(''.join(['camera_quaternion: ',
                               repr(tuple(self._cam_quat)),'\n']))
        string.append(''.join(['camera_position: ',repr(tuple(self._cam_pos)),
                               '\n']))
        return ''.join(string)

    def _unproject_to_depth(self, points, depth_map, t=None):
        """Unproject image points into the world at a particular time.

        :param points: Coordinates of a point or points within the image
            coordinate system. The coordinate may be Cartesian or homogenous.
        :type points: array with shape (2), (2,N), (3) or (3,N)

        :param depth_map: Depth image providing depth at each pixel.
            :type depth_map: Numpy 2-D array

        :param t: Time at which to unproject the point(s) (time in seconds
            since Unix epoch).
        :type t: float

        :return: Position of each ray's after project to its depth (i.e.,
            intersection with a world surface).
        :rtype: Numpy.ndarry with shape (3,n).

        """
        points = np.atleast_2d(points)
        points = np.reshape(points, (2,-1))
        ray_pos, ray_dir = self.unproject(points, t=t, normalize_ray_dir=False)

        for i in range(points.shape[1]):
            x,y = points[:,i]
            # Get ray distance traveled until intersection. Therefore, we need
            # to evaluate the depth map at x,y. We need to convert from image
            # coordinates (i.e., upper-left corner of upper-left pixel is 0,0)
            # to image indices. Explicitly handle potential rounding
            # variability at edges of image domain.
            if x == 0:
                ix = 0
            elif x == self.width:
                ix = int(self.width - 1)
            else:
                ix = int(round(x - 0.5))

            if y == 0:
                iy = 0
            elif y == self.height:
                iy = int(self.height - 1)
            else:
                iy = int(round(y - 0.5))

            if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
                print(x == self.width)
                print(y == self.height)
                raise ValueError('Coordinates (%0.1f,%0.f) are outside the '
                                 '%ix%i image' %
                                 (x,y,self.width,self.height))

            ray_pos[:,i] += ray_dir[:,i]*depth_map[iy,ix]

        return ray_pos


class GeoStaticCamera(DepthCamera):
    """Stationary camera with a fixed pose at some geo-fixed location.

    width, height, K, dist, lat, lon, altitude, cam_quat

    """
    def __init__(self, width, height, K, dist, depth_map, latitude, longitude,
                 altitude, R):
        """
        See additional documentation from base class above.

        :param latitude: Latitude in degrees of the camera's position.

        :param longitude: Longitude in degrees of the camera's position.
        :type longitude: float

        :param altitude: Height of the camera in meters above the WGS84
            ellipsoid.
        :type altitude: float

        """
        R = np.array(R)
        R /= np.linalg.det(R)
        cam_pos = np.array([0,0,0])
        cam_quat = np.array([0,0,0,1])

        # Quaternion for level system (z down) with x-axis pointing north.
        enu_quat = np.array([1/np.sqrt(2),1/np.sqrt(2),0,0])

        platform_pose_provider = PlatformPoseFixed(pos=np.array([0,0,0]),
                                           quat=enu_quat,
                                           lat0=latitude, lon0=longitude,
                                           h0=altitude)

        super(GeoStaticCamera, self).__init__(width=width, height=height, K=K,
                                           dist=dist, cam_pos=cam_pos,
                                           cam_quat=cam_quat,
                                           depth_map=depth_map,
                                           platform_pose_provider=platform_pose_provider)
        self._R = R

        self._depth_map = depth_map

        # The local ENU coordinate system is located at the camera.
        self._tvec = np.array([[0],[0],[0]], dtype=np.float64)
        self._camera_pose = np.hstack([R,self._tvec])

    def __str__(self):
        string = ['model_type: static\n']
        string.append(super(GeoStaticCamera, self).__str__())
        string.append('\n')
        string.append(''.join(['fx: ',repr(self._K[0,0]),'\n']))
        string.append(''.join(['fy: ',repr(self._K[1,1]),'\n']))
        string.append(''.join(['cx: ',repr(self._K[0,2]),'\n']))
        string.append(''.join(['cy: ',repr(self._K[1,2]),'\n']))
        string.append(''.join(['distortion_coefficients: ',
                               repr(tuple(self._dist)),
                               '\n']))
        string.append(''.join(['latitude: %0.8f' % self.latitude,
                               '\n']))
        string.append(''.join(['longitude: %0.8f' % self.longitude,
                               '\n']))
        string.append(''.join(['altitude: %0.8f' % self.altitude,
                               '\n']))
        string.append(''.join(['R: ',
                               repr(tuple(self.R)),'\n']))
        return ''.join(string)

    @classmethod
    def load_from_file(cls, filename, platform_pose_provider=None):
        """See base class Camera documentation.

        """
        with open(filename, 'r') as f:
            calib = yaml.safe_load(f)

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
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        R = np.reshape(np.array(calib['R']), (3,3))
        latitude = calib['latitude']
        longitude = calib['longitude']
        altitude = calib['altitude']

        depth_map_fname = '%s_depth_map.tif' % os.path.splitext(filename)[0]
        try:
            depth_map = np.asarray(PIL.Image.open(depth_map_fname))
        except OSError:
            depth_map = None

        return cls(width, height, K, dist, depth_map, latitude, longitude,
                   altitude, R)

    def save_to_file(self, filename):
        """See base class Camera documentation.

        """
        with open(filename, 'w') as f:
            f.write(''.join(['# The type of camera model.\n',
                             'model_type: static\n\n',
                             '# Image dimensions\n']))

            f.write(''.join(['image_width: ',to_str(self.width),'\n']))
            f.write(''.join(['image_height: ',to_str(self.height),'\n\n']))

            f.write('# Focal length along the image\'s x-axis.\n')
            f.write(''.join(['fx: ',to_str(self._K[0,0]),'\n\n']))

            f.write('# Focal length along the image\'s y-axis.\n')
            f.write(''.join(['fy: ',to_str(self._K[1,1]),'\n\n']))

            f.write('# Principal point is located at (cx,cy).\n')
            f.write(''.join(['cx: ',to_str(self._K[0,2]),'\n']))
            f.write(''.join(['cy: ',to_str(self._K[1,2]),'\n\n']))

            f.write(''.join(['# Distortion coefficients following OpenCv\'s ',
                    'convention\n']))

            dist = self._dist
            if np.all(dist == 0):
                dist = 'None'

            f.write(''.join(['distortion_coefficients: ',
                             to_str(self._dist),'\n\n']))

            f.write(''.join(['# Rotation matrix mapping vectors defined in an '
                             'east/north/up coordinate system\n# centered at '
                             'the camera into vectors defined in the camera'
                             'coordinate system.\n',
                             'R: [%0.10f, %0.10f, %0.10f,\n'
                             '    %0.10f, %0.10f, %0.10f,\n'
                             '    %0.10f, %0.10f, %0.10f]' %
                             tuple(self.R.ravel()), '\n\n']))

            f.write(''.join(['# Location of the camera\'s center of '
                             'projection. Latitude and longitude are in\n# '
                             'degrees, and altitude is meters above the WGS84 '
                             'ellipsoid.\n',
                             'latitude: %0.10f\n' % self.latitude,
                             'longitude: %0.10f\n' % self.longitude,
                             'altitude: %0.10f' % self.altitude,'\n\n']))

        if self.depth_map is not None:
            im = PIL.Image.fromarray(self.depth_map, mode='F') # float32
            depth_map_fname = '%s_depth_map.tif' % os.path.splitext(filename)[0]
            im.save(depth_map_fname)

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R /= value/np.linalg.det(value)
        self._rvec = cv2.Rodrigues(self.R)[0].ravel()

    @property
    def latitude(self):
        return self.platform_pose_provider.lat0

    @property
    def longitude(self):
        return self.platform_pose_provider.lon0

    @property
    def altitude(self):
        return self.platform_pose_provider.h0

    def get_camera_pose(self, t=None):
        """Returns 3x4 matrix mapping world points to camera vectors.

        :param t: Time at which to query the camera's pose (time in seconds
            since Unix epoch).

        :return: A 3x4 matrix that accepts a homogeneous 4-vector defining a
            3-D point in the world and returns a Cartesian 3-vector in the
            camera's coordinate system pointing from the camera's center of
            projection to the word point (i.e., the negative of the principal
            ray coming from this world point).
        :rtype: 3x4 array

        """
        return self._camera_pose

    def project(self, points, t=None):
        """See Camera.project documentation.

        The world points being projected into the camera are defined relative
        to a local east/north/up coordinate system with origin at the camera's
        center of projection.

        The projection is independent of time, but we retain the function
        prototype of the base Camera class.

        """
        points = np.array(points, dtype=np.float64)
        if points.ndim == 1:
            points = np.atleast_2d(points).T

        # Project rays into camera coordinate system.
        im_pts = cv2.projectPoints(points.T, self._rvec, self._tvec, self.K,
                                   self.dist)[0]
        return np.squeeze(im_pts, 1).T

    def unproject(self, points, t=None, normalize_ray_dir=True):
        """See Camera.unproject documentation.

        The ray position and direction are defined relative to a local
        east/north/up coordinate system with origin at the camera's center of
        projection.

        The projection is independent of time, but we retain the function
        prototype of the base Camera class.

        """
        points = np.array(points, dtype=np.float64)
        if points.ndim == 1:
            points = np.atleast_2d(points).T
            points = np.reshape(points, (2,-1))

        # Unproject rays into the camera coordinate system.
        ray_dir = np.ones((3,points.shape[1]), dtype=points.dtype)
        ray_dir0 = cv2.undistortPoints(np.expand_dims(points.T, 1),
                                       self.K, self.dist, R=None)
        ray_dir[:2] = np.squeeze(ray_dir0, 1).T

        # Rotate rays into the local east/north/up coordinate system.
        ray_dir = np.dot(self.R.T, ray_dir)

        if normalize_ray_dir:
            ray_dir /= np.sqrt(np.sum(ray_dir**2, 0))

        ray_pos = np.zeros_like(ray_dir)

        return ray_pos, ray_dir


class MapCamera(Camera):
    """Camera producing map images.

    This object is primarily built around GDAL.

    """
    def __init__(self, base_layer):
        super(MapCamera, self).__init__(width=base_layer.res_x,
                                        height=base_layer.res_y)
        self.base_layer = base_layer

    def project(self, points, t=None):
        return np.array([self.base_layer.meters_to_raster(point) for point in points.T]).T

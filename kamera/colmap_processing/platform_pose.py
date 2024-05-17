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

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import threading
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import bisect

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# colmap_processing imports
from colmap_processing.geo_conversions import llh_to_enu, enu_to_llh
from colmap_processing.rotations import quaternion_slerp, quaternion_matrix, \
    quaternion_inverse, quaternion_multiply, euler_from_quaternion


lock = threading.Lock()


ODOMETRY_HEADERS = {}
ODOMETRY_HEADERS['time (s)  latitude (deg)  longitude (deg)  height (m)  quatx  quaty  quatz  quatw'] = 1
ODOMETRY_HEADERS['time (s)  latitude (deg)  longitude (deg)  height (m)  quatx  quaty  quatz  quatw std_easting (m) std_northing (m) std_up (m) std_heading (rad) std_pitch (rad) std_roll (rad)'] = 2
ODOMETRY_HEADERS['time (s)  easting (m)  northing (m)  up (m)  quatx  quaty  quatz  quatw'] = 3
ODOMETRY_HEADERS['time (s)  easting (m)  northing (m)  up (m)  quatx  quaty  quatz  quatw std_easting (m) std_northing (m) std_up (m) std_heading (rad) std_pitch (rad) std_roll (rad)'] = 4


class PlatformPoseProvider(object):
    """Camera manager object.

    This object supports requests for the navigation coordinate system state,
    at a particular time, relative to a local east/north/up coordinate system.

    Attributes:
    :param lat0: Latitude of the origin (deg).
    :type lat0: float

    :param lon0: Longitude of the origin (deg).
    :type lon0: float

    :param h0: Height above the WGS84 ellipsoid of the origin (meters).
    :type h0: float

    """
    def __init__(self):
        raise NotImplementedError

    def pos(self, t):
        """
        :param t: Time at which to query the platform state (time in seconds
            since Unix epoch).
        :type t: float

        :return: Position of the platform coordinate system relative to a
            local level east/north/up coordinate system.
        :rtype: 3-pos

        """
        raise NotImplementedError

    def quat(self, t):
        """
        :param t: Time at which to query the platform state (time in seconds
            since Unix epoch).
        :type t: float

        :return: Quaternion (qx, qy, qz, qw) specifying the orientation of the
            navigation coordinate system relative to a local level
            east/north/up (ENU) coordinate system. The quaternion represent a
            coordinate system rotation from ENU to the navigation coordinate
            system.
        :rtype: 4-array

        """
        raise NotImplementedError

    def pose(self, t):
        """
        :param t: Time at which to query the INS state (time in seconds since
            Unix epoch).
        :type t: float

        :return: List with the first element the 3-array position (see pos) and
            the second element the 4-array orientation quaternion (see quat).
        :rtype: list

        """
        raise NotImplementedError

    def __str__(self):
        return str(type(self))

    def __repr__(self):
        return self.__str__()


class PlatformPoseInterp(PlatformPoseProvider):
    """Interpolated pose from a time series.

    """
    def __init__(self, lat0=None, lon0=None, h0=None, max_length=np.inf):
        """
        :param lat0: Geodetic latitude of origin (degrees).
        :type lat0: None | float

        :param lon0: Longitude of origin (degrees).
        :type lon0: None | float

        :param h0: Height above WGS84 ellipsoid at the origin (meters).
        :type h0: None | float

        :param max_length: Maximum number of states to maintain in the history.
            If None, all states will be retained.
        :type max_length: None | int
        """
        self._pose_time_series = np.zeros((0,8))
        self.lat0 = lat0
        self.lon0 = lon0
        self.h0 = h0
        self.max_length = max_length
        self._has_std = False

    @classmethod
    def from_odometry_llh_txt(cls, fname, lat0=None, lon0=None, h0=None,
                              max_length=None):
        """Load from an odometry text file encoding position at latitude and
        longitude.

        The odometry text file provides a state per time via a line including
        the following elements:
        0 - time (s)
        1 - latitude (deg)
        2 - longitude (deg)
        3 - height (m)
        4 - quatx
        5 - quaty
        6 - quatz
        7 - quatw
        8 [optional] - std_easting (m)
        9 [optional] - std_northing (m)
        10 [optional] - std_up (m)
        11 [optional] - std_heading (rad)
        12 [optional] - std_pitch (rad)
        13 [optional] - std_roll (rad)

        :param fname: Path to odometry text file.
        :type fname: str

        :param lat0: Geodetic latitude of origin (degrees). If None, the
            median latitude will be used.
        :type lat0: None | float

        :param lon0: Longitude of origin (degrees). If None, the median
            longitude will be used.
        :type lon0: None | float

        :param h0: Height above WGS84 ellipsoid at the origin (meters). If
            None, the minimum value will be used.
        :type h0: None | float

        :param max_length: Maximum number of states to maintain in the history.
            If None, all states will be retained.
        :type max_length: None | int
        """
        self = cls(lat0, lon0, h0, max_length)

        # Read the header to determine the contents of the file.
        with open(fname, 'r') as f:
            header = f.readline()

        header = header[2:-1]

        try:
            ver = ODOMETRY_HEADERS[header]
        except KeyError:
            raise ValueError('Unrecognized header format in file \'%s\': %s'
                             % (fname, header))

        odometry = np.loadtxt(fname, comments='#')

        # Determine whether x, y coordinates are in latitude and longitude.
        inllh = ver in [1, 2]

        if inllh:
            if self.lat0 is None:
                self.lat0 = np.median(odometry[:, 1])

            if self.lon0 is None:
                self.lon0 = np.median(odometry[:, 2])

            if self.h0 is None:
                self.h0 = np.min(odometry[:, 3])

            enu = [llh_to_enu(d[1], d[2], d[3], self.lat0, self.lon0, self.h0)
                   for d in odometry]

            odometry[:, 1:4] = enu

        self._pose_time_series = odometry

        if ver in [2]:
            self._has_std = True
        else:
            self._has_std = False

        # Check for duplicate state definitions.
        ind = np.where(np.diff(self._pose_time_series[:, 0], axis=0) > 0)[0] + 1
        ind = np.hstack([0, ind])
        self._pose_time_series = self._pose_time_series[ind]

        return self

    @property
    def pose_time_series(self):
        with lock:
            return self._pose_time_series

    def to_odometry_llh_txt(self, odometry_txt_fname):
        odometry = self._pose_time_series.copy()

        if self.lat0 is None or self.lon0 is None or self.h0 is None:
            if self.lat0 is not None or self.lon0 is not None or self.h0 is not None:
                raise Exception('If any of \'lat0\', \'lon0\', \'h0\' is None, '
                                'they should all be None')

        if self.lat0 is None:
            if self._pose_time_series.shape[1] == 8:
                header = 'time (s)  easting (m)  northing (m)  up (m)  quatx  quaty  quatz  quatw'
            elif self._pose_time_series.shape[1] == 14:
                header = 'time (s)  easting (m)  northing (m)  up (m)  quatx  quaty  quatz  quatw std_easting (m) std_northing (m) std_up (m) std_heading (rad) std_pitch (rad) std_roll (rad)'
            else:
                raise Exception('Unhandled case for \'_pose_time_series\' '
                                'of length %i' % self._pose_time_series.shape[1])
        else:
            if self._pose_time_series.shape[1] == 8:
                header = 'time (s)  latitude (deg)  longitude (deg)  height (m)  quatx  quaty  quatz  quatw'
            elif self._pose_time_series.shape[1] == 14:
                header = 'time (s)  latitude (deg)  longitude (deg)  height (m)  quatx  quaty  quatz  quatw std_easting (m) std_northing (m) std_up (m) std_heading (rad) std_pitch (rad) std_roll (rad)'
            else:
                raise Exception('Unhandled case for \'_pose_time_series\' '
                                'of length %i' % self._pose_time_series.shape[1])

            for i in range(len(odometry)):
                odometry[i, 1:4] = enu_to_llh(odometry[i, 1],
                                              odometry[i, 2],
                                              odometry[i, 3], self.lat0, self.lon0,
                                              self.h0)

        np.savetxt(odometry_txt_fname, odometry, header=header)

    def add_to_pose_time_series(self, t, pos, quat, std=None):
        """Adds to pose time series such that time is monotonically
        increasiing.

        :param std: Estimated standard deviation in the easting, northing, and
            up values (meters) and the heading pitch roll (radians).
        :type std: array of length 6
        """
        if std is not None and not self._has_std:
            if len(self._pose_time_series) > 0:
                raise ValueError('This instance has already has received '
                                 'updates without standard deviation supplied,'
                                 ' so it cannot now accomodate standard '
                                 'deviations with updates')
            else:
                self._pose_time_series = np.hstack([t, pos, quat, std])
                self._has_std = True
                return

        if std is None and self._has_std:
            raise ValueError('This instance has already has received standard '
                             'deviation  values, so it requires that updates '
                             'also provide standard deviations')

        if std is not None:
            pose = np.hstack([t, pos, quat, std])
        else:
            pose = np.hstack([t, pos, quat])

        with lock:
            if len(self._pose_time_series) == 0:
                self._pose_time_series = np.insert(self._pose_time_series, 0,
                                                   pose, axis=0)
                return

            ind = bisect.bisect_left(self._pose_time_series[:, 0], t)
            self._pose_time_series = np.insert(self._pose_time_series, ind,
                                               pose, axis=0)

            if (self.max_length is not None and
                len(self._pose_time_series) > self.max_length):
                self._pose_time_series = self._pose_time_series[-self.max_length:]

    def pose(self, t, return_std=False):
        """See PlatformPose documentation.

        :param return_std: Also return estimated standard deviation in values.
        :type return_std: bool

        Return:
        pos
        quat
        std (optional): Estimated standard deviation in the easting, northing,
            and up values (meters) and the heading pitch roll (radians).
        """
        if return_std:
            assert self._has_std

        with lock:
            ind = bisect.bisect(self._pose_time_series[:, 0], t)
            if ind == len(self._pose_time_series):
                ind -= 1
            elif ind == 0:
                ind += 1

            t1 = self._pose_time_series[ind - 1, 0]
            t2 = self._pose_time_series[ind, 0]

            if np.abs(t - t1) < 1e-6:
                if return_std:
                    return self._pose_time_series[ind - 1, 1:4], self._pose_time_series[ind - 1, 4:8], self._pose_time_series[ind - 1, 8:]
                else:
                    return self._pose_time_series[ind - 1, 1:4], self._pose_time_series[ind - 1, 4:8]
            elif np.abs(t - t2) < 1e-6:
                if return_std:
                    return self._pose_time_series[ind, 1:4], self._pose_time_series[ind, 4:8], self._pose_time_series[ind, 8:]
                else:
                    return self._pose_time_series[ind, 1:4], self._pose_time_series[ind, 4:8]

            pos1 = self._pose_time_series[ind - 1, 1:4]
            quat1 = self._pose_time_series[ind - 1, 4:8]
            pos2 = self._pose_time_series[ind, 1:4]
            quat2 = self._pose_time_series[ind, 4:8]

            weight = (t - t1) / (t2 - t1)
            quat = quaternion_slerp(quat1, quat2, weight, spin=0,
                                    shortestpath=True)

            # Interpolate position.
            pos = pos1*(1 - weight) + pos2*weight

            if return_std:
                std1 = self._pose_time_series[ind - 1, 8:]
                std2 = self._pose_time_series[ind, 8:]
                std = std1*(1 - weight) + std2*weight
                return pos, quat, std
            else:
                return pos, quat

    @property
    def times(self):
        """Return times at which pose is exactly specified.

        """
        return self._pose_time_series[:,0]

    def pos(self, t):
        """See PlatformPose documentation.

        """
        return self.pose(t)[0]
        #return self.average_pos()

    def quat(self, t):
        """See PlatformPose documentation.

        """
        return self.pose(t)[1]
        #return self.average_quat()

    def average_quat(self):
        if len(self.pose_time_series) > 0:
            return np.mean(self.pose_time_series[:,4:8], 0)

    def average_pos(self):
        if len(self.pose_time_series) > 0:
            return np.mean(self.pose_time_series[:,1:4], 0)

    def plot(self, max_states=None, show_uncertainty=False):
        data = self.pose_time_series

        if max_states is not None:
            ind = np.round(np.linspace(0, len(data) - 1, max_states)).astype(int)
            data = data[ind]

        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        plt.plot(data[:, 1], data[:, 2])

        if show_uncertainty and self._has_std:
            ax = plt.gca()
            for i in range(len(data)):
                ax.add_patch(Ellipse(data[i, 1:3],
                                     width=data[i, 8]*2,
                                     height=data[i, 9]*2,
                                     linewidth=2, fill=False))

        plt.xlabel('X', fontsize=40)
        plt.ylabel('Y', fontsize=40)
        plt.axis('image')
        plt.tight_layout()

        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        plt.plot(data[:, 1], data[:, 3])
        plt.plot(data[:, 1], data[:, 3], '.')

        if show_uncertainty and self._has_std:
            ax = plt.gca()
            for i in range(len(data)):
                ax.add_patch(Ellipse((data[i, 1], data[i, 3]),
                                     width=data[i, 8]*2,
                                     height=data[i, 10]*2,
                                     linewidth=2, fill=False))

        plt.xlabel('X', fontsize=40)
        plt.ylabel('Z', fontsize=40)
        plt.tight_layout()

        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        plt.plot(data[:, 2], data[:, 3])
        plt.plot(data[:, 2], data[:, 3], '.')

        if show_uncertainty and self._has_std:
            ax = plt.gca()
            for i in range(len(data)):
                ax.add_patch(Ellipse((data[i, 2], data[i, 3]),
                                     width=data[i, 9]*2,
                                     height=data[i, 10]*2,
                                     linewidth=2, fill=False))

        plt.xlabel('Y', fontsize=40)
        plt.ylabel('Z', fontsize=40)
        plt.tight_layout()

        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        plt.plot(data[:, 0], data[:, 3])
        plt.plot(data[:, 0], data[:, 3], '.')
        plt.xlabel('Time (s)', fontsize=40)
        plt.ylabel('Z', fontsize=40)
        plt.tight_layout()

    def compare_to(self, other, crop_to_paired_times=True, max_states=None,
                   show_uncertainty=False):
        """Compare to another instance of PlatformPoseInterp.
        """
        data1 = self.pose_time_series
        data2 = other.pose_time_series

        tmin = max(min(data1[:, 0]), min(data2[:, 0]))
        tmax = min(max(data1[:, 0]), max(data2[:, 0]))

        ind = np.logical_and(data1[:, 0] >= tmin, data1[:, 0] <= tmax)
        data1 = data1[ind]

        ind = np.logical_and(data2[:, 0] >= tmin, data2[:, 0] <= tmax)
        data2 = data2[ind]

        if max_states is not None:
            ind = np.round(np.linspace(0, len(data1) - 1, max_states)).astype(int)
            data1 = data1[ind]
            ind = np.round(np.linspace(0, len(data2) - 1, max_states)).astype(int)
            data2 = data2[ind]

        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        plt.plot(data1[:, 1], data1[:, 2], 'r.', label='This')
        plt.plot(data2[:, 1], data2[:, 2], 'b.', label='Other')

        if show_uncertainty and self._has_std:
            ax = plt.gca()
            for i in range(len(data1)):
                ax.add_patch(Ellipse(data1[i, 1:3],
                                     width=data1[i, 8]*2,
                                     height=data1[i, 9]*2,
                                     linewidth=2, fill=False))

        plt.xlabel('X', fontsize=40)
        plt.ylabel('Y', fontsize=40)
        plt.axis('image')
        plt.legend(fontsize=24)
        plt.tight_layout()

        # x vs. z
        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        plt.plot(data1[:, 1], data1[:, 3], 'r.', label='This')
        plt.plot(data2[:, 1], data2[:, 3], 'b.', label='Other')

        if show_uncertainty and self._has_std:
            ax = plt.gca()
            for i in range(len(data1)):
                ax.add_patch(Ellipse(data1[i, 1:3],
                                     width=data1[i, 8]*2,
                                     height=data1[i, 9]*2,
                                     linewidth=2, fill=False))

        plt.xlabel('X', fontsize=40)
        plt.ylabel('Z', fontsize=40)
        plt.axis('image')
        plt.legend(fontsize=24)
        plt.tight_layout()

        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        txt = ['Pos-X', 'Pos-Y', 'Pos-Z']
        for i in range(1, 4):
            plt.subplot(3, 1, i)
            plt.plot(data1[:, 0], data1[:, i], 'r.', label='This')
            plt.plot(data2[:, 0], data2[:, i], 'b.', label='Other')

            if show_uncertainty and self._has_std:
                ax = plt.gca()
                for i in range(len(data1)):
                    ax.add_patch(Ellipse(data1[i, 1:3],
                                         width=data1[i, 8]*2,
                                         height=data1[i, 9]*2,
                                         linewidth=2, fill=False))

            plt.xlabel('Time (s)', fontsize=40)
            plt.ylabel(txt[i-1], fontsize=40)
            plt.legend(fontsize=14)
            plt.tight_layout()

        plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
        plt.rc('font', **{'size': 20})
        plt.rc('axes', linewidth=4)
        txt = ['Quat-X', 'Quat-Y', 'Quat-Z', 'Quat-W']
        for i in range(1, 5):
            plt.subplot(4, 1, i)
            plt.plot(data1[:, 0], data1[:, i+3], 'r.', label='This')
            plt.plot(data2[:, 0], data2[:, i+3], 'b.', label='Other')

            if show_uncertainty and self._has_std:
                ax = plt.gca()
                for i in range(len(data1)):
                    ax.add_patch(Ellipse(data1[i, 1:3],
                                         width=data1[i, 8]*2,
                                         height=data1[i, 9]*2,
                                         linewidth=2, fill=False))

            plt.xlabel('Time (s)', fontsize=40)
            plt.ylabel(txt[i-1], fontsize=40)
            plt.legend(fontsize=14)
            plt.tight_layout()

    def estimate_imu_output(self, rate=100, s=1/500, gravity=9.81,
                            plot_results=False):
        data = self.pose_time_series

        times = data[:, 0]
        N = np.ceil((times[-1] - times[0])*rate)
        times_out = np.arange(N + 1)/rate + times[0]

        if plot_results:
            plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
            plt.rc('font', **{'size': 20})
            plt.rc('axes', linewidth=4)
            labels = ['X', 'Y', 'Z']

        # Calculate the acceleration vector (m/s^2) within the east/north/up
        # coordinate system.
        accel = []
        for i in range(1, 4):
            sposx = UnivariateSpline(data[:, 0], data[:, i], k=5, s=len(data)*s)

            if plot_results:
                plt.subplot(3, 1, i)
                plt.plot(data[:, 0], data[:, i], 'bo')
                plt.plot(data[:, 0], sposx(data[:, 0]), 'r-')
                plt.ylabel('%s-Position (m)' % (labels[i-1]))
                plt.tight_layout()

            #plt.plot(data[:, 0], data[:, 1])
            #plt.plot(data[:, 0], sposx(data[:, 0]), 'r.')
            sposdxx = sposx.derivative(2)
            accel.append(sposdxx(times_out))

        accel = np.array(accel)
        accel[2] += gravity
        accel = accel.T

        quats = [self.quat(t) for t in times_out]

        if plot_results:
            plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
            plt.rc('font', **{'size': 20})
            plt.rc('axes', linewidth=4)
            labels = ['X', 'Y', 'Z', 'W']
            quats_ =  np.array([self.quat(t) for t in times]).T

            for i in range(4):
                plt.subplot(4, 1, i + 1)
                plt.plot(times, quats_[i], 'bo')
                plt.ylabel('%s-Quat' % (labels[i-1]))

            plt.tight_layout()

        rmats = [quaternion_matrix(quats[i])[:3, :3].T
                 for i in range(len(times_out))]

        # Project into the navigation coordinate system.
        accel_out = np.array([np.dot(rmats[i], accel[i])
                              for i in range(len(times_out))])

        omegas = []
        dt = 1/rate
        for i in range(len(quats) - 1):
            q1 = quats[i]
            q2 = quats[i + 1]
            dq = quaternion_multiply(quaternion_inverse(q1), q2)
            dq = dq/np.linalg.norm(dq)
            omegas.append(dq[:3]/np.linalg.norm(dq[:3])*2*np.arccos(dq[3])/dt)

        if False:
            # Test
            q1 = [1, 2, 3, 4]
            R1 = quaternion_matrix(q1)[:3, :3].T

            # Create random rotation vector in navigation coordinate system.
            omega = np.random.rand(3)*2-1
            omega_world = np.dot(R1.T, omega)
            norm = np.linalg.norm(omega)
            dquat = quat_wxyz_to_xyzw(transformations.quaternion_about_axis(norm, omega_world/norm))
            q2 = quaternion_multiply(dquat, q1)
            R2 = quaternion_matrix(q2)[:3, :3].T
            dq = quaternion_multiply(quaternion_inverse(q1), q2)
            omega2 = dq[:3]/np.linalg.norm(dq[:3])*2*np.arccos(dq[3])
            print(omega - omega2)

        accel_out = (accel_out[1:] + accel_out[:-1])/2
        times_out = (times_out[1:] + times_out[:-1])/2
        omegas = np.array(omegas)
        out = np.hstack([np.atleast_2d(times_out).T, accel_out, omegas])

        if plot_results:
            data = out.T
            plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
            plt.rc('font', **{'size': 20})
            plt.rc('axes', linewidth=4)
            plt.subplot(311)
            plt.plot(data[0], data[1], '.')
            plt.ylabel('X-Accel (m/s^2)')
            plt.subplot(312)
            plt.plot(data[0], data[2], '.')
            plt.ylabel('Y-Accel (m/s^2)')
            plt.subplot(313)
            plt.plot(data[0], data[3], '.')
            plt.ylabel('Z-Accel (m/s^2)')
            plt.tight_layout()

            plt.figure(num=None, figsize=(15.3, 10.7), dpi=80);
            plt.subplot(311)
            plt.plot(data[0], data[4], '.')
            plt.ylabel('X-Angular Vel (rad/s)')
            plt.subplot(312)
            plt.plot(data[0], data[5], '.')
            plt.ylabel('Y-Angular Vel (rad/s)')
            plt.subplot(313)
            plt.plot(data[0], data[6], '.')
            plt.ylabel('Z-Angular Vel (rad/s)')
            plt.tight_layout()

        return out


class PlatformPoseFixed(PlatformPoseProvider):
    def __init__(self, pos=np.array([0, 0, 0]),
                 quat=np.array([0, 0, 0, 1]), lat0=None,
                 lon0=None, h0=None):
        self._pos = pos
        self._quat = quat
        self.lat0 = lat0
        self.lon0 = lon0
        self.h0 = h0

    def pos(self, t):
        """See PlatformPose documentation.

        """
        return self._pos

    def quat(self, t):
        """See PlatformPose documentation.

        """
        return self._quat

    def pose(self, t):
        """See PlatformPose documentation.

        """
        return [self._pos, self._quat]

#!/usr/bin/env python
import numpy as np
import threading
import bisect
import json

# KAMERA imports
#from libnmea_navsat_driver.stream_archive import enumerate_packets
#from libnmea_navsat_driver.gsof import GsofEventDispatch, GsofInsDispatch, \
#    maybe_gsof, parse_gsof_stream, separate_nmea
from kamera.sensor_models.nav_conversions import (
        enu_to_llh,
        llh_to_enu,
        ned_quat_to_enu_quat
        )
from kamera.sensor_models import (
        euler_from_quaternion,
        quaternion_multiply,
        quaternion_from_matrix,
        quaternion_from_euler,
        quaternion_inverse,
        quaternion_matrix,
        quaternion_slerp
        )

lock = threading.Lock()


class NavStateProvider(object):
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
        :param t: Time at which to query the INS state (time in seconds since
            Unix epoch).
        :type t: float

        :return: Position of the navigation coordinate system relative to a
            local level east/north/up coordinate system.
        :rtype: 3-pos

        """
        raise NotImplementedError

    def quat(self, t):
        """
        :param t: Time at which to query the INS state (time in seconds since
            Unix epoch).
        :type t: float

        :return: Quaternion (qx,qy,qz,qw) specifying the orientatio of the
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


class NavStateINSBinary(NavStateProvider):
    """
    lat0 (degrees)
    lon0 (degrees)
    h0 (meters)

    """
    def __init__(self, nav_binary_fname, lat0=None, lon0=None, alt0=0):
        """
        :param nav_binary_fname: Path to binary file generated by the INS.
        :type nav_binary_fname: str

        """
        self._pose_time_series = np.zeros((0,8))
        #print('NavStateOdometry object subscribing to:', nav_odom_topic)

        with open(nav_binary_fname, 'rb') as fp:
            raw_stream = fp.read()

        # All navigation data at 100 Hz.
        all_nav_dict = {}

        events_dict = {}
        for i, rawdata in enumerate_packets(raw_stream):
            # todo: optionally archive stream

            nmea_list, gsof_data = separate_nmea(rawdata)
            if maybe_gsof(gsof_data):
                dispatches = parse_gsof_stream(gsof_data)
                # aprint(dispatch.msg)
                for d in dispatches:
                    if isinstance(d, GsofInsDispatch):
                        all_nav_dict[d.time] = d
                    elif isinstance(d, GsofEventDispatch):
                        events_dict[d.time] = d

        all_times = np.sort(all_nav_dict.keys())
        event_times = np.sort(events_dict.keys())

        nav_pose = []
        for curr_time in event_times:
            ind = bisect.bisect(all_times, curr_time)
            nav1 = all_nav_dict[all_times[ind-1]]
            nav2 = all_nav_dict[all_times[ind]]

            # How much weight should be given to the values at nav1. (1-weight)
            # will be given to the values at nav2.
            dt = (nav2.time - nav1.time)
            if dt == 0:
                weight = 0.5
                print('err: dt was zero, using 0.5 weight')
            else:
                weight = (curr_time - nav1.time)/dt

            quat1 = quaternion_from_euler(nav1.heading/180*np.pi,
                                          nav1.pitch/180*np.pi,
                                          nav1.roll/180*np.pi,
                                          axes='rzyx')
            quat1 = ned_quat_to_enu_quat(quat1)
            quat2 = quaternion_from_euler(nav2.heading/180*np.pi,
                                          nav2.pitch/180*np.pi,
                                          nav2.roll/180*np.pi,
                                          axes='rzyx')
            quat2 = ned_quat_to_enu_quat(quat2)

            # Interpolate quaternions.
            quat = quaternion_slerp(quat1, quat2, weight, spin=0,
                                    shortestpath=True)

            # Interpolate position.
            lat = nav1.latitude*weight + nav2.latitude*(1 - weight)
            lon = nav1.longitude*weight + nav2.longitude*(1 - weight)
            alt = nav1.altitude*weight + nav2.altitude*(1 - weight)

            nav_pose.append([curr_time,lat,lon,alt,quat[0],quat[1],quat[2],
                             quat[3]])

        nav_pose = np.array(nav_pose)

        if lat0 is not None:
            self.lat0 = lat0
        else:
            self.lat0 = np.median(nav_pose[:,1])

        if lon0 is not None:
            self.lon0 = lon0
        else:
            self.lon0 = np.median(nav_pose[:,2])

        self.h0 = alt0

        for i in range(len(nav_pose)):
            nav_pose[i,1:4] = llh_to_enu(nav_pose[i,1], nav_pose[i,2],
                                         nav_pose[i,3], self.lat0, self.lon0,
                                         self.h0, in_degrees=True)

        self._pose_time_series = nav_pose

    @property
    def pose_time_series(self):
        with lock:
            return self._pose_time_series

    def pose(self, t):
        """See NavState documentation.

        """
        ind = bisect.bisect(self._pose_time_series[:,0], t)

        if ind == len(self._pose_time_series):
            ind = ind - 1

        elif ind > 0 and abs(t - self._pose_time_series[ind - 1,0]) < \
           abs(t - self._pose_time_series[ind,0]):
            ind = ind - 1

        pose = self._pose_time_series[ind,1:]

        return [pose[:3],pose[3:]]

    def pos(self, t):
        """See NavState documentation.

        """
        return self.pose(t)[0]
        #return self.average_pos()

    def quat(self, t):
        """See NavState documentation.

        """
        return self.pose(t)[1]
        #return self.average_quat()

    def average_quat(self):
        if len(self.pose_time_series) > 0:
            return np.mean(self.pose_time_series[:,4:], 0)

    def average_pos(self):
        if len(self.pose_time_series) > 0:
            return np.mean(self.pose_time_series[:,1:4], 0)

    def llh(self, t):
        # East/north/up coordinates.
        pos = self.pos(t)

        lat, lon, alt = enu_to_llh(*pos, lat0=self.lat0, lon0=self.lon0,
                                   h0=self.h0)
        return lat, lon, alt


class NavStateINSJson(NavStateINSBinary):
    def __init__(self, json_paths, lat0=None, lon0=None, alt0=0):
        """
        :param json_path: GLOB path for metadata json file(s).
        :type json_path: str

        """
        #json_paths = glob.glob(json_path)
        frame_times = []
        nav_pose_dict = {}
        self._ins_nav_times = []
        self._ins_ypr = []
        self._ins_llh = []
        self.time_to_save_every_x_image = {}
        self.time_to_seq = {}
        json_counter = 0
        for fname in json_paths:
            with open(fname) as json_file:
                try:
                    d = json.load(json_file)
                except json.decoder.JSONDecodeError:
                    print("Failed to decode json file %s." % json_file)
                    continue

                if d['ins']['time'] in nav_pose_dict:
                    continue
                time = float(d['ins']['time'])
                self._ins_nav_times.append(time)
                self._ins_ypr.append([d['ins']['heading'],
                                      d['ins']['pitch'],
                                      d['ins']['roll']])

                self._ins_llh.append([d['ins']['latitude'],
                                      d['ins']['longitude'],
                                      d['ins']['altitude']])

                quat = quaternion_from_euler(d['ins']['heading']/180*np.pi,
                                             d['ins']['pitch']/180*np.pi,
                                             d['ins']['roll']/180*np.pi,
                                             axes='rzyx')
                quat = ned_quat_to_enu_quat(quat)

                nav_pose_dict[time] = [d['ins']['time'],
                                       d['ins']['latitude'],
                                       d['ins']['longitude'],
                                       d['ins']['altitude'],
                                       quat[0], quat[1], quat[2],
                                       quat[3]]

                try:
                    save_every_x_image = int(d['save_every_x_image'])
                except KeyError:
                    save_every_x_image = 1
                evt_time = d['evt']['time']
                self.time_to_save_every_x_image[evt_time] = save_every_x_image
                self.time_to_seq[evt_time] = int(d["evt"]["header"]["seq"])

                frame_times.append(evt_time)
                json_counter += 1
        print('Reconstructed navigation stream from %i meta.json files' %
              json_counter)

        self._ins_ypr = np.array(self._ins_ypr)
        self._ins_llh = np.array(self._ins_llh)
        self._ins_nav_times = np.array(self._ins_nav_times)
        ind = np.argsort(self._ins_nav_times)
        self._ins_nav_times = self._ins_nav_times[ind]
        self._ins_ypr = self._ins_ypr[ind, :]
        self._ins_llh = self._ins_llh[ind, :]

        frame_times = sorted(list(set(frame_times)))
        nav_times = sorted(list(nav_pose_dict.keys()))

        nav_pose = []
        for frame_time in frame_times:
            if len(nav_times) == 1:
                nav = nav_pose_dict[nav_times[0]]
                quat = nav[4:]
                lat = nav[1]
                lon = nav[2]
                alt = nav[3]
            else:
                ind = bisect.bisect(nav_times, frame_time)
                if ind == len(nav_times):
                    ind -= 1
                elif ind == 0:
                    ind += 1

                nav1 = nav_pose_dict[nav_times[ind-1]]
                nav2 = nav_pose_dict[nav_times[ind]]

                # How much weight should be given to the values at nav1.
                # (1-weight)  will be given to the values at nav2.
                dt = (nav2[0] - nav1[0])
                if dt == 0:
                    weight = 0.5
                    print('err: dt was zero, using 0.5 weight')
                else:
                    weight = (frame_time - nav1[0])/dt
                assert not np.isnan(weight) # This has to be real

                quat1 = nav1[4:]
                quat2 = nav2[4:]

                # Interpolate quaternions.
                quat = quaternion_slerp(quat1, quat2, weight, spin=0,
                                        shortestpath=True)

                # Interpolate position.
                lat = nav1[1]*(1 - weight) + nav2[1]*weight
                lon = nav1[2]*(1 - weight) + nav2[2]*weight
                alt = nav1[3]*(1 - weight) + nav2[3]*weight

            nav_pose.append([frame_time, lat, lon, alt, quat[0], quat[1],
                             quat[2], quat[3]])

        nav_pose = np.array(nav_pose)

        if lat0 is not None:
            self.lat0 = lat0
        else:
            self.lat0 = np.median(nav_pose[:, 1])

        if lon0 is not None:
            self.lon0 = lon0
        else:
            self.lon0 = np.median(nav_pose[:, 2])

        self.h0 = alt0

        for i in range(len(nav_pose)):
            nav_pose[i, 1:4] = llh_to_enu(nav_pose[i, 1], nav_pose[i, 2],
                                          nav_pose[i, 3], self.lat0, self.lon0,
                                          self.h0, in_degrees=True)

        self._pose_time_series = nav_pose

    def ins_heading_pitch_roll(self, t):
        """Yaw, pitch, and roll reported by the INS at time t.

        """
        ind = bisect.bisect(self._ins_nav_times, t)
        if ind == len(self._ins_nav_times):
            ind -= 1
        elif ind == 0:
            ind += 1

        t1 = self._ins_nav_times[ind-1]
        t2 = self._ins_nav_times[ind]
        ypr1 = self._ins_ypr[ind-1]
        ypr2 = self._ins_ypr[ind]

        if np.abs(t - t1) < 1e-4:
            return ypr1
        elif np.abs(t - t2) < 1e-4:
            return ypr2

        quat1 = quaternion_from_euler(*(ypr1/180*np.pi), axes='rzyx')
        quat2 = quaternion_from_euler(*(ypr2/180*np.pi), axes='rzyx')

        # How much weight should be given to the values at nav1.
        # (1-weight)  will be given to the values at nav2.
        dt = (t2 - t1)
        if dt == 0:
            weight = 0.5
            print('err: dt was zero, using 0.5 weight')
        else:
            weight = (t - t1) / dt
        quat = quaternion_slerp(quat1, quat2, weight, spin=0,
                                shortestpath=True)

        ypr = np.array(euler_from_quaternion(quat, axes='rzyx'))*180/np.pi

        if ypr[0] < 0:
            ypr[0] += 360

        return np.array(euler_from_quaternion(quat, axes='rzyx'))*180/np.pi

    def ins_llh(self, t):
        """Latitude, longitudem and height reported by INS.

        """
        ind = bisect.bisect(self._ins_nav_times, t)
        if ind == len(self._ins_nav_times):
            ind -= 1
        elif ind == 0:
            ind += 1

        t1 = self._ins_nav_times[ind-1]
        t2 = self._ins_nav_times[ind]
        nav1 = self._ins_llh[ind-1]
        nav2 = self._ins_llh[ind]

        if np.abs(t - t1) < 1e-4:
            return nav1
        elif np.abs(t - t2) < 1e-4:
            return nav2

        # How much weight should be given to the values at nav1.
        # (1-weight)  will be given to the values at nav2.
        dt = (t2 - t1)
        if dt == 0:
            weight = 0.5
            print('err: dt was zero, using 0.5 weight')
        else:
            weight = (t - t1)/dt

        # Interpolate position.
        lat, lon, h = nav1*(1 - weight) + nav2*weight

        return lat, lon, h


class NavStateFixed(NavStateProvider):
    def __init__(self, pos=np.array([0, 0, 0]),
                 quat=np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])):
        self._pos = pos
        self._quat = quat

    def pos(self, t):
        """See NavState documentation.

        """
        return self._pos

    def quat(self, t):
        """See NavState documentation.

        """
        return self._quat

    def pose(self, t):
        """See NavState documentation.

        """
        return [self._pos, self._quat]

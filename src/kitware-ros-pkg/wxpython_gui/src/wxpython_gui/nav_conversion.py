#!/usr/bin/env python
"""
The MIT License (MIT); this license applies to GeographicLib,
versions 1.12 and later.

Copyright (c) 2008-2017, Charles Karney

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

==============================================================================

Library handling conversions related to geo-spatial navigation. Algorithms are
ported from GeographicLib C++ code into Python.

Altitude refers to height about the WGS84 ellipsoid unless otherwise noted.

Some computations require GeographicLib
- https:#geographiclib.sourceforge.io/html/CartConvert.1.html
- sudo apt-get install geographiclib-tools

"""
from __future__ import division, print_function
import numpy as np
import subprocess
from math import cos, sin, sqrt

# ROS imports
from tf.transformations import quaternion_multiply, quaternion_matrix, \
    quaternion_from_euler, quaternion_inverse, euler_matrix

# WGS84 constants
_a = 6378137
_f = 1/(298257223563/1000000000)
_e2 = _f*(2-_f)
_e2m = np.square(1-_f)
_e2a = abs(_e2)
_e4a = np.square(_e2)
epsilon = np.finfo(float).eps
_maxrad = 2 * _a / epsilon


def llh_to_enu(lat, lon, h, lat0, lon0, h0, in_degrees=True, pure_python=True):
    """Convert east, north, and up to latitude, longitude, and altitude.

    East, north, and up are coordinates within a local level Cartesian
    coordinate system with origin at geodetic latitude lat0, longitude lon0,
    and height h0 above the WGS84 ellipsoid. Up is normal to the ellipsoid
    surface and north is in the direction of the true north.

    :param lat: Geodetic latitude of the point to convert.
    :type lat: float

    :param lon: Longitude of the point to convert.
    :type lon: float

    :param h: Height above WGS84 ellipsoid of the point to convert (meters).
    :type h: float

    :param lat0: Geodetic latitude of the local east/north/up coordinate
        system.
    :type lat: float

    :param lon0: Longitude of the local east/north/up coordinate system.
    :type lon: float

    :param h0: Height above the WGS84 ellipsoid of the local east/north/up
        coordinate system (meters).
    :type height: float

    :param in_degrees: Specify that all angles are in degrees.
    :type in_degrees: bool

    :param pure_python: Specify whether to use the pure-Python implementation
        (faster) or the reference implementation from GeographicLib's command
        line call to CartConvert.
    :type pure_python: bool

    :return: East, north, and up coordinates (meters) of the converted point.
    :rtype: list

    Example:
    lat = 35.906437
    lon = -79.056282
    h = 123
    lat0 = 35.905446
    lon0 = -79.060788
    h0 = 0

    """
    if not in_degrees:
        lat = lat*180/np.pi
        lon = lon*180/np.pi
        lat0 = lat0*180/np.pi
        lon0 = lon0*180/np.pi

    if pure_python:
        sphi, cphi = sincosd(lat0)
        slam, clam = sincosd(lon0)
        _r = geocentric_rotation(sphi, cphi, slam, clam)
        xc,yc,zc = llh_to_ecef(lat, lon, h, in_degrees=True)
        _x0,_y0,_z0 = llh_to_ecef(lat0, lon0, h0, in_degrees=True)
        xc -= _x0; yc -= _y0; zc -= _z0;
        x = _r[0] * xc + _r[3] * yc + _r[6] * zc;
        y = _r[1] * xc + _r[4] * yc + _r[7] * zc;
        z = _r[2] * xc + _r[5] * yc + _r[8] * zc;
        return [x,y,z]
    else:
        output = subprocess.check_output(['CartConvert','-l',
                                          str(lat0),str(lon0),
                                          str(h0),'--input-string',
                                          ' '.join([str(lat),str(lon),str(h)])])
        return [float(s) for s in output.split('\n')[0].split(' ')]


def enu_to_llh(east, north, up, lat0, lon0, h0, in_degrees=True,
               pure_python=True):
    """Convert latitude, longitude, and height to east, north, up.

    East, north, and up are coordinates within a local level Cartesian
    coordinate system with origin at geodetic latitude lat0, longitude lon0,
    and height h0 above the WGS84 ellipsoid. Up is normal to the ellipsoid
    surface and north is in the direction of the true north.

    :param east: Easting coordinate (meters) of the point to convert.
    :type east: float

    :param north: Northing coordinate (meters) of the point to convert.
    :type north: float

    :param up: Up coordinate (meters) of the point to convert.
    :type up: float

    :param lat0: Geodetic latitude of the local east/north/up coordinate
        system.
    :type lat: float

    :param lon0: Longitude of the local east/north/up coordinate system.
    :type lon: float

    :param h0: Height above the WGS84 ellipsoid of the local east/north/up
        coordinate system (meters).
    :type height: float

    :param in_degrees: Specify that all angles are in degrees.
    :type in_degrees: bool

    :param pure_python: Specify whether to use the pure-Python implementation
        (faster) or the reference implementation from GeographicLib's command
        line call to CartConvert.
    :type pure_python: bool

    :return: Geodetic latitude, longitude, and height (meters) above the WGS84
        ellipsoid of the converted point.
    :rtype: list

    Example:
    lat = 35.906437
    lon = -79.056282
    h = 123
    lat0 = 35.905446
    lon0 = -79.060788
    h0 = 0

    """
    if not in_degrees:
        lat0 = lat0*180/np.pi
        lon0 = lon0*180/np.pi

    if pure_python:
        x, y, z = east, north, up
        sphi, cphi = sincosd(lat0)
        slam, clam = sincosd(lon0)
        _r = geocentric_rotation(sphi, cphi, slam, clam)
        _x0,_y0,_z0 = llh_to_ecef(lat0, lon0, h0, in_degrees=True)

        xc = _x0 + _r[0] * x + _r[1] * y + _r[2] * z,
        yc = _y0 + _r[3] * x + _r[4] * y + _r[5] * z,
        zc = _z0 + _r[6] * x + _r[7] * y + _r[8] * z;
        lat, lon, h = ecef_to_llh(xc, yc, zc, in_degrees)
    else:
        output = subprocess.check_output(['CartConvert','-r','-l',str(lat0),
                                          str(lon0),str(h0),'--input-string',
                                          ' '.join([str(east),str(north),
                                                    str(up)])])

        lat, lon, h = [float(s) for s in output.split('\n')[0].split(' ')]

    if not in_degrees:
        lat = lat*180/np.pi
        lon = lon*180/np.pi

    return [lat,lon,h]


def ned_quat_to_enu_quat(quat):
    """
    :param quat: ROS-standard quaternion (x,y,z,w) that represents a rotation
        of the NED coordinate system into a moving coordinate system.
    :type quat: 4-array

    :return: ROS-standard quaternion (x,y,z,w) that represents a rotation
        of the NED coordinate system into a moving coordinate system.
    :rtype: 4-array

    """
    return quaternion_multiply([np.sqrt(2)/2,np.sqrt(2)/2,0,0], quat)


def enu_quat_to_ned_quat(quat):
    """
    :param quat: ROS-standard quaternion (x,y,z,w) that represents a rotation
        of the NED coordinate system into a moving coordinate system.
    :type quat: 4-array

    :return: ROS-standard quaternion (x,y,z,w) that represents a rotation
        of the NED coordinate system into a moving coordinate system.
    :rtype: 4-array

    """
    return quaternion_multiply([np.sqrt(2)/2,np.sqrt(2)/2,0,0], quat)


def ecef_to_llh(X, Y, Z, in_degrees=True):
    """
    Ported from GeographicLib.

    :param X: ECEF x coordinate (meters).
    :type X: float

    :param X: ECEF x coordinate (meters).
    :type X: float

    :param X: ECEF x coordinate (meters).
    :type X: float

    :param in_degrees: Specify that all angles are in degrees.
    :type in_degrees: bool

    :return: Geodetic latitude, longitude, and height (meters) above the WGS84
        ellipsoid of the converted point.
    :rtype: list

    Example:
    x = 5634247
    y = 2050698
    z = 2167698

    """
    R = np.hypot(X,Y)
    if R == 0:
        slam = 0
        clam = 1
    else:
        slam = Y / R
        clam = X / R

    h = np.hypot(R,Z)      # Distance to center of earth
    if (h > _maxrad):
        # We really far away (> 12 million light years) treat the earth as a
        # point and h, above, is an acceptable approximation to the height.
        # This avoids overflow, e.g., in the computation of disc below.  It's
        # possible that h has overflowed to inf but that's OK.
        #
        # Treat the case X, Y finite, but R overflows to +inf by scaling by 2.
        R = np.hypot(X/2, Y/2)

        if R == 0:
            slam = 0
            clam = 1
        else:
            slam = (Y/2) / R
            clam = (X/2) / R

        H = np.hypot(Z/2,R)
        sphi = (Z/2) / H
        cphi = R / H
    elif _e4a == 0:
        # Treat the spherical case.  Dealing with underflow in the general case
        # with _e2 = 0 is difficult.  Origin maps to N pole same as with
        # ellipsoid.
        if h == 0:
            H = np.hypot(1, R)
            sphi = 1 / H
        else:
            H = np.hypot(Z, R)
            sphi = Z / H

        cphi = R / H
        h -= _a
    else:
        # Treat prolate spheroids by swapping R and Z here and by switching
        # the arguments to phi = atan2(...) at the end.
        p = np.square(R / _a)
        q = _e2m * np.square(Z / _a)
        r = (p + q - _e4a) / 6
        if _f < 0:
            p,q = q,p

        if not (_e4a * q == 0 and r <= 0):
            # Avoid possible division by zero when r = 0 by multiplying
            # equations for s and t by r^3 and r, resp.
            S = _e4a * p * q / 4 # S = r^3 * s
            r2 = np.square(r)
            r3 = r * r2
            disc = S * (2 * r3 + S)
            u = r
            if (disc >= 0):
                T3 = S + r3
                # Pick the sign on the sqrt to maximize abs(T3).  This
                # minimizes loss of precision due to cancellation.  The result
                # is unchanged because of the way the T is used in definition
                # of u.
                if T3 < 0:  # T3 = (r * t)^3
                    T3 += -np.sqrt(disc)
                else:
                    T3 += np.sqrt(disc)

                # N.B. cbrt always returns the real root.  cbrt(-8) = -2.
                T = np.cbrt(T3) # T = r * t
                # T can be zero but then r2 / T -> 0.
                if T != 0:
                    u += T + (r2 / T)

            else:
                # T is complex, but the way u is defined the result is real.
                ang = np.arctan2(np.sqrt(-disc), -(S + r3))
                # There are three possible cube roots.  We choose the root
                # which avoids cancellation.  Note that disc < 0 implies that
                # r < 0.
                u += 2 * r * np.cos(ang / 3)

            v = np.sqrt(np.square(u) + _e4a * q) # guaranteed positive
            # Avoid loss of accuracy when u < 0.  Underflow doesn't occur in
            # e4 * q / (v - u) because u ~ e^4 when q is small and u < 0.
            if u < 0:  # u+v guaranteed positive
                uv = _e4a * q / (v - u)
            else:
                uv = u + v

            # Need to guard against w going negative due to roundoff in uv - q.
            w = np.maximum(0, _e2a * (uv - q) / (2 * v))
            # Rearrange expression for k to avoid loss of accuracy due to
            # subtraction.  Division by 0 not possible because uv > 0, w >= 0.
            k = uv / (np.sqrt(uv + np.square(w)) + w)
            if _f >= 0:
                k1 = k
                k2 = k + _e2
            else:
                k1 = k - _e2
                k2 = k

            d = k1 * R / k2
            H = np.hypot(Z/k1, R/k2)
            sphi = (Z/k1) / H
            cphi = (R/k2) / H
            h = (1 - _e2m/k1) * np.hypot(d, Z)

        else:    # e4 * q == 0 && r <= 0
            # This leads to k = 0 (oblate, equatorial plane) and k + e^2 = 0
            # (prolate, rotation axis) and the generation of 0/0 in the general
            # formulas for phi and h.  using the general formula and division by 0
            # in formula for h.  So handle this case by taking the limits:
            # f > 0: z -> 0, k      ->   e2 * sqrt(q)/sqrt(e4 - p)
            # f < 0: R -> 0, k + e2 -> - e2 * sqrt(q)/sqrt(e4 - p)
            if _f >= 0:
                zz = np.sqrt((_e4a - p) / _e2m)
            else:
                zz = np.sqrt(p / _e2m)

            if _f <  0:
                xx = np.sqrt(_e4a - p)
            else:
                xx = np.sqrt(p)

            H = np.hypot(zz, xx)
            sphi = zz / H
            cphi = xx / H
            if Z < 0:
                sphi = -sphi # for tiny negative Z (not for prolate)

            if _f >= 0:
                h = - _a * (_e2m) * H / _e2a
            else:
                h = - _a * (1) * H / _e2a

    lat = float(np.arctan2(sphi, cphi)*180/np.pi)
    lon = float(np.arctan2(slam, clam)*180/np.pi)
    return lat, lon, h


def llh_to_ecef(lat, lon, h, in_degrees=True):
    """
    Ported from GeographicLib.

    :param lat: Geodetic latitude of the point to convert.
    :type lat: float

    :param lon: Longitude of the point to convert.
    :type lon: float

    :param h: Height above WGS84 ellipsoid of the point to convert (meters).
    :type h: float

    :param in_degrees: Specify that all angles are in degrees.
    :type in_degrees: bool

    :return: ECEF X,Y,Z coordinates (meters).
    :rtype: list

    """
    if not in_degrees:
        lat = lat*180/np.pi
        lon = lon*180/np.pi

    sphi,cphi = sincosd(lat)
    slam,clam = sincosd(lon)

    n = _a/np.sqrt(1-_e2*np.square(sphi))
    Z = (_e2m * n + h) * sphi
    X = (n + h) * cphi
    Y = X * slam
    X *= clam
    return [float(X),float(Y),float(Z)]


def geocentric_rotation(sphi, cphi, slam, clam):
    """
    This rotation matrix is given by the following quaternion operations
    qrot(lam, [0,0,1]) * qrot(phi, [0,-1,0]) * [1,1,1,1]/2
    or
    qrot(pi/2 + lam, [0,0,1]) * qrot(-pi/2 + phi , [-1,0,0])
    where
    qrot(t,v) = [cos(t/2), sin(t/2)*v[1], sin(t/2)*v[2], sin(t/2)*v[3]]

    """
    M = np.zeros(9)
    # Local X axis (east) in geocentric coords
    M[0] = -slam;        M[3] =  clam;        M[6] = 0;
    # Local Y axis (north) in geocentric coords
    M[1] = -clam * sphi; M[4] = -slam * sphi; M[7] = cphi;
    # Local Z axis (up) in geocentric coords
    M[2] =  clam * cphi; M[5] =  slam * cphi; M[8] = sphi;
    return M


def sincosd(x):
    """
    * Evaluate the sine and cosine function with the argument in degrees
    *
    * @tparam T the type of the arguments.
    * @param[in] x in degrees.
    * @param[out] sinx sin(<i>x</i>).
    * @param[out] cosx cos(<i>x</i>).
    *
    * The results obey exactly the elementary properties of the trigonometric
    * functions, e.g., sin 9&deg; = cos 81&deg; = &minus; sin 123456789&deg;.
    * If x = &minus;0, then \e sinx = &minus;0; this is the only case where
    * &minus;0 is returned.

    """
    r = np.fmod(x, 360)
    q = int(np.floor(r / 90 + 0.5))
    r -= 90 * q
    # now abs(r) <= 45
    r *= np.pi / 180
    s = np.sin(r)
    c = np.cos(r)

    # python 2.7 on Windows 32-bit machines has problems dealingwith -0.0.
    if x == 0:
        s = x

    if np.uint8(q) & np.uint8(3) == np.uint(0):
        sinx =  s; cosx =  c
    elif np.uint8(q) & np.uint8(3) == np.uint(1):
        sinx =  c; cosx = -s
    elif np.uint8(q) & np.uint8(3) == np.uint(2):
        sinx = -s; cosx = -c
    else:
      sinx = -c; cosx =  s

    # Set sign of 0 results.  -0 only produced for sin(-0)
    if x:
        sinx += 0
        cosx += 0

    return sinx, cosx


def rmat_enu_ecef(lat, lon, in_degrees=True):
    """Rotation matrix that transforms an ENU vector in an ECEF vector.

        :param lat: Geodetic latitude of the east/north/up coordinate system.
    :type lat: float

    :param lon: Longitude of the east/north/up coordinate system.
    :type lon: float

    :param in_degrees: Specify that all angles are in degrees. Otherwise,
        radians are assumed.
    :type in_degrees: bool

    """
    if in_degrees:
        lat = lat/180*np.pi
        lon = lon/180*np.pi

    clat = cos(lat)
    slat = sin(lat)
    clon = cos(lon)
    slon = sin(lon)

    return np.array([[-slon, -slat*clon, clat*clon],
                     [clon, -slat*slon, clat*slon],
                     [0, clat, slat]])


def quat_enu_ecef(lat, lon, in_degrees=True):
    """Quaternion that transforms an ENU vector in an ECEF vector.

        :param lat: Geodetic latitude of the east/north/up coordinate system.
    :type lat: float

    :param lon: Longitude of the east/north/up coordinate system.
    :type lon: float

    :param in_degrees: Specify that all angles are in degrees. Otherwise,
        radians are assumed.
    :type in_degrees: bool

    """
    return quaternion_inverse(quat_ecef_enu(lat, lon, in_degrees=in_degrees))


def quat_ecef_enu(lat, lon, in_degrees=True):
    """Quaternion that transforms an ENU vector into an ECEF vector.

    :param lat: Geodetic latitude of the east/north/up coordinate system.
    :type lat: float

    :param lon: Longitude of the east/north/up coordinate system.
    :type lon: float

    :param in_degrees: Specify that all angles are in degrees. Otherwise,
        radians are assumed.
    :type in_degrees: bool

    :return: Quaternion (x,y,z,w).
    :rtype: Nympy 4-array of floats

    """
    if in_degrees:
        lat = lat/180*np.pi
        lon = lon/180*np.pi

    # This operator needs to rotate the axes of the ECEF coordinate system into
    # the ENU coordinate system.

    # First rotate around 90 degrees around ECEF Z.
    q1 = np.array([0, 0, 1/sqrt(2), 1/sqrt(2)])

    # Rotate around ECEF Y by latitude.
    q2 = np.array([0, sin((np.pi/2 - lat)/2), 0, cos((np.pi/2 - lat)/2)])

    # Rotate around ECEF Z by longitude.
    q3 = np.array([0, 0, sin(lon/2), cos(lon/2)])

    q = quaternion_multiply(q3, quaternion_multiply(q2, q1))

    return quaternion_inverse(q)


def rmat_ecef_enu(lat, lon, in_degrees=True):
    """Rotation matrix that transforms an ECEF vector into an ENU vector

    :param lat: Geodetic latitude of the east/north/up coordinate system.
    :type lat: float

    :param lon: Longitude of the east/north/up coordinate system.
    :type lon: float

    :param in_degrees: Specify that all angles are in degrees. Otherwise,
        radians are assumed.
    :type in_degrees: bool

    """
    if in_degrees:
        lat = lat/180*np.pi
        lon = lon/180*np.pi

    clat = cos(lat)
    slat = sin(lat)
    clon = cos(lon)
    slon = sin(lon)

    return np.array([[-slon, clon, 0],
                     [-clon*slat, -slon*slat, clat],
                     [clon*clat, slon*clat, slat]])


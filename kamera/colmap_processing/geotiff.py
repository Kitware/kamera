#!/usr/bin/env python
"""
ckwg +31
Copyright 2017-2018 by Kitware, Inc.
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

Note: the raster coordiante system has its origin at the center of the top left
pixel.

"""
from __future__ import division, print_function, absolute_import
import numpy as np
from osgeo import osr, gdal

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Kitware imports
from colmap_processing.geo_conversions import enu_to_llh, llh_to_enu, \
    FastENUConverter


wgs84_cs = osr.SpatialReference()
wgs84_cs.SetWellKnownGeogCS("WGS84")


gdal_interpolation = {}
gdal_interpolation['average'] = gdal.GRA_Average
gdal_interpolation['bilinear'] = gdal.GRA_Bilinear
gdal_interpolation['cubic'] = gdal.GRA_Cubic
gdal_interpolation['cubic_spline'] = gdal.GRA_CubicSpline
gdal_interpolation['lancosz'] = gdal.GRA_Lanczos
gdal_interpolation['nearest_neighbor'] = gdal.GRA_NearestNeighbour


def numpy_to_gdal_type(dtype):
    if dtype == np.uint8:
        return gdal.GDT_Byte
    if dtype == np.float32:
        return gdal.GDT_Float32
    if dtype == np.float64:
        return gdal.GDT_Float64
    if dtype == np.uint16:
        return gdal.GDT_Int16
    if dtype == np.uint32:
        return gdal.GDT_Int32

    raise ValueError('Unhandled type %s' % dtype)


class GeoTIFF(object):
    """Handling of loading, saving, and accessing a GeoTIFF.

    This object is primarily built around GDAL.

    """
    def __init__(self, gdal_ds):
        """

        """
        # The gdal_ds defines everything about the map coordinate system.
        self.gdal_ds = gdal_ds

        # Pre-compute some useful elements.
        self._raster_cs = osr.SpatialReference()
        self._raster_cs.ImportFromWkt(gdal_ds.GetProjectionRef())

        self._raster_to_lla = osr.CoordinateTransformation(self._raster_cs,
                                                           wgs84_cs)
        self._lla_to_raster = osr.CoordinateTransformation(wgs84_cs,
                                                          self._raster_cs)

        geo_tform_mat = np.identity(3)
        c = gdal_ds.GetGeoTransform()
        geo_tform_mat = np.array([[c[1], c[2], c[0]],
                                  [c[4], c[5], c[3]],
                                  [  0,    0,    1]])
        self._geo_inv_tform_mat = np.linalg.inv(geo_tform_mat)[:2]
        self._geo_tform_mat = geo_tform_mat[:2]

        # Calculated and set on first use or must be provided explicitly.
        self._origin_latitude = None
        self._origin_longitude = None
        self._origin_height = 0
        self.enu_converter = None

    @classmethod
    def load(cls, filename):
        ds = gdal.Open(filename, gdal.GA_ReadOnly)

        if ds is None:
            raise OSError()

        return cls(ds)

    @property
    def gdal_ds(self):
        return self._gdal_ds

    @gdal_ds.setter
    def gdal_ds(self, value):
        self._gsd_x = None
        self._gsd_y = None
        self._raster = None
        self._gdal_ds = value

    @property
    def res_x(self):
        return self.gdal_ds.RasterXSize

    @property
    def res_y(self):
        return self.gdal_ds.RasterYSize

    @property
    def origin_latitude(self):
        return self._origin_latitude

    @property
    def origin_longitude(self):
        return self._origin_longitude

    @property
    def origin_height(self):
        return self._origin_height

    @property
    def gsd(self):
        """Ground sampling distances (meters) along each axis of the raster.

        The GSD is calculated at the center of the raster. If the GSD varies
        noticeally across the raster, using just the center GSD may lead to
        innacurate results.

        """
        if self._gsd_x is None:
            ll1 = self.raster_to_lon_lat((self.res_x/2, self.res_y/2))
            ll2 = self.raster_to_lon_lat((self.res_x/2 + 1, self.res_y/2))
            ll3 = self.raster_to_lon_lat((self.res_x/2, self.res_y/2 + 1))
            dx = llh_to_enu(ll2[1], ll2[0], 0, ll1[1], ll1[0], 0)[:2]
            self._gsd_x = np.linalg.norm(dx)
            dy = llh_to_enu(ll3[1], ll3[0], 0, ll1[1], ll1[0], 0)[:2]
            self._gsd_y = np.linalg.norm(dy)

        return self._gsd_x, self._gsd_y

    @property
    def raster(self):
        """Return Numpy array.

        """
        if self._raster is None:
            raster = self.gdal_ds.GetRasterBand(1).ReadAsArray()
            if self.gdal_ds.RasterCount > 1:
                self._raster = np.zeros((self.res_y,self.res_x, self.gdal_ds.RasterCount),
                                       dtype=raster.dtype)
                self._raster[:, :, 0] = raster
                for i in range(1, self.gdal_ds.RasterCount):
                    self._raster[:,:,i] = self.gdal_ds.GetRasterBand(i+1).ReadAsArray()
            else:
                self._raster = raster

        return self._raster

    @raster.setter
    def raster(self, raster):
        gdal_drv = gdal.GetDriverByName('MEM')

        if raster.ndim == 2:
            channels = 1
        else:
            channels = raster.shape[2]

        dest_ds = gdal_drv.Create('', raster.shape[1], raster.shape[0],
                                  channels, numpy_to_gdal_type(raster.dtype))
        dest_ds.SetProjection(self.gdal_ds.GetProjection())
        dest_ds.SetGeoTransform(self.gdal_ds.GetGeoTransform())

        if channels == 1:
            dest_ds.GetRasterBand(1).WriteArray(raster[:,:], 0, 0)
        else:
            for i in range(channels):
                dest_ds.GetRasterBand(i+1).WriteArray(raster[:,:,i], 0, 0)

        self.gdal_ds = dest_ds

    @property
    def extent_meters(self):
        points = np.array([[0, 0, self.res_x, self.res_x],
                           [0, self.res_y, self.res_y, 0]])
        xy = self.raster_to_meters(points)
        return xy[0].min(), xy[0].max(), xy[1].min(), xy[1].max()

    def save(self, fname, compression=None):
        """Save GeoTIFF to file.

        :param fname:
        :type fname: str

        :param compression: JPEG compression quality
            setting (0-100) or None for no compression.
        :type compression: int | None

        """
        if compression is not None:
            # Compress the RGB and UV.
            gdal_settings = ['COMPRESS=JPEG',
                             'JPEG_QUALITY=%i' % compression]
        else:
            gdal_settings = []

        # Perform the projection/resampling
        gdal_drv = gdal.GetDriverByName('GTiff')
        dest_ds = gdal_drv.Create(fname, self.res_x, self.res_y,
                                  self.gdal_ds.RasterCount,
                                  numpy_to_gdal_type(self.raster.dtype),
                                  gdal_settings)
        dest_ds.SetProjection(self.gdal_ds.GetProjection())
        dest_ds.SetGeoTransform(self.gdal_ds.GetGeoTransform())

        if self.gdal_ds.RasterCount == 1:
            dest_ds.GetRasterBand(1).WriteArray(self.raster[:,:], 0, 0)
        else:
            for i in range(self.gdal_ds.RasterCount):
                dest_ds.GetRasterBand(i+1).WriteArray(self.raster[:,:,i], 0, 0)

        dest_ds.FlushCache()  # Write to disk.

    def set_origin(self, latitude, longitude, height=0,
                   fast_enu_converter=False):
        """Set latitude and longitude of local tangent plane coordinate system.

        Set the origin of the local tangent plane coordinate system from which
        world coordinates in meters can be measured.

        :param lat: Geodetic latitude (degrees) of the origin.
        :type lat: float

        :param lon: Longitude (degrees) of the origin.
        :type lon: float

        """
        self._origin_latitude = latitude
        self._origin_longitude = longitude
        self._origin_height = height

        points = np.array([[0, 0, self.res_x, self.res_x],
                           [0, self.res_y, self.res_y, 0]])
        lon, lat = self.raster_to_lon_lat(points)

        lat_range = [lat.min(), lat.max()]
        lon_range = [lon.min(), lon.max()]

        if fast_enu_converter:
            self.enu_converter = FastENUConverter(lat_range, lon_range,
                                                  [height-1, height + 1],
                                                  latitude, longitude, height,
                                                  accuracy=[1e-2, 1e-2, 1000])

    def raster_to_lon_lat(self, points):
        """Get latitude, longitude, height associated with raster coordinate(s).

        :param points: Coordinates of a point or points within the raster
            coordinate system.
        :type points: array with shape (2), (2,N)

        :param t: Time at which to query the camera's pose (time in seconds
            since Unix epoch).
        :type t: float | None

        :param cov: List of 2x2 covariance matrices indicating the
            positional uncertainty, in raster space, of the points.
        :type cov: list of 2x2 array | None

        :return: Latitude (degrees), longitude (degrees), and height above
            WGS84 ellipsoid (meters). If 'points_cov' is not None, the second
            element of the return is a list of 3x3 arrays, one for each output
            point, indicating the localization uncertainty in a local-tangent
            east, north, up coordinate system centered at the point.
        :rtype: numpy.ndarray of size (2,n) and (optionally) a list of 3x3
            arrays

        """
        points = np.array(points, dtype=np.float64)
        if points.ndim == 1:
            is_1d = True
            points = np.atleast_2d(points).T
        else:
            is_1d = False
            points = points

        num_points = points.shape[1]

        # Convert from raster coordinates to the coordinates of the projection
        # encoded in the geotiff.
        XY = np.ones((3, num_points))
        XY[:2] = points

        XY = [self._raster_to_lla.TransformPoint(*i) for i in points.T]
        XY = np.array(XY).T

        # Not sure what the third coordinate is supposed to be, but let's make
        # XY homogenous 2-D points.
        XY[2] = 1

        # Convert to latitude and longitude.
        lon_lat = np.dot(self._geo_tform_mat, XY)

        if is_1d:
            lon_lat = lon_lat.ravel()

        return lon_lat

    def lon_lat_to_raster(self, lon_lat):
        """Project latitude and longitude into raster coordinates.

        :param lon_lat: Latitude (degrees) longitude (degrees).
        :type points: array with shape (2) | (2, N)

        :return: Coordinates of a point or points within the raster
            coordinate system.
        :rtype: array with shape (2), (2, N)

        """
        lon_lat = np.array(lon_lat)
        if lon_lat.ndim == 1:
            is_1d = True
            lon_lat = np.atleast_2d(lon_lat)
        else:
            is_1d = False
            lon_lat = lon_lat.T

        XY = np.ones((len(lon_lat), 3))
        XY[:, :2] = [self._lla_to_raster.TransformPoint(_[0], _[1])[:2]
                     for _ in lon_lat]
        XY = XY.T

        # Convert from projection coordinates to raw raster coordinates.
        im_pt = np.dot(self._geo_inv_tform_mat, XY)

        if is_1d:
            im_pt = im_pt.ravel()

        return im_pt

    def raster_to_meters(self, points, height=None):
        """Convert from raster coordinates to easting/northing meters.

        Easting and northing meters are measured relative to the specific local
        tangent plane origin.

        :param height: Height (meters) above the WGS84 ellipsoid (if known).
        :type hieght: float

        """
        if self.origin_latitude is None:
            # self._origin_longitude gets set at same time as
            # self._origin_latitude, so we don't need to check both.
            raise Exception('Must call method \'set_origin\' first to set the '
                            'origin')

        lon_lat = self.raster_to_lon_lat(points)

        if height is None:
            if lon_lat.ndim == 2:
                height = np.zeros(lon_lat.shape[1])
            else:
                height = 0

        if self.enu_converter is not None:
            return np.array(self.enu_converter.llh_to_enu(lon_lat[1],
                                                          lon_lat[0], height))

        if lon_lat.ndim == 1:
            return np.array(llh_to_enu(lon_lat[1], lon_lat[0], height,
                                       self.origin_latitude,
                                       self.origin_longitude,
                                       self.origin_height))

        xy = [llh_to_enu(lon_lat_[1], lon_lat_[0], height[0],
                         self.origin_latitude, self.origin_longitude,
                         self.origin_height)
              for lon_lat_ in lon_lat.T]
        return np.array(xy).T

    def meters_to_raster(self, xy):
        """Convert easting/northing (optional up) meters to raster coordinates.

        Easting and northing meters are measured relative to the specific local
        tangent plane origin.

        The up coordinate can alter the associated latitude and longitude.

        :param xy: Easting, northing, and (optional) up coordinates in meters.
        :type xy: float array of size (2,) | (2,N) | (3,) | (3,N)

        """
        if self._origin_latitude is None:
            # self._origin_longitude gets set at same time as
            # self._origin_latitude, so we don't need to check both.
            raise Exception('Must call method \'set_origin\' first to set the '
                            'origin')

        xy = np.array(xy).T
        if xy.ndim == 1:
            is_1d = True
            xy = np.atleast_2d(xy)
        else:
            is_1d = False
            xy = xy

        if xy.shape[1] == 2:
            up = np.zeros(len(xy))
        else:
            up = xy[:, 2]

        if self.enu_converter is not None:
            lon_lat = self.enu_converter.enu_to_llh(xy[:, 0], xy[:, 1], up)
            lon_lat = np.array(lon_lat)
        else:
            lon_lat = [enu_to_llh(xy_[0], xy_[1], up, self.origin_latitude,
                                  self.origin_longitude, self.origin_height)
                       for xy_ in xy]
            lon_lat = np.array(lon_lat).T

        if is_1d:
            lon_lat = lon_lat.ravel()

        return self.lon_lat_to_raster(lon_lat[:2][::-1])

    def resample(self, ul=None, lr=None, scale=1, update_origin=False,
                 interpolation='bilinear'):
        """Resample to a new discretization.

        :param ul: The coordinates of the current raster that will be mapped to
            the upper-left corner, (0, 0), coordinates in the resampled raster.
        :type ul: array-like (2,)

        :param lr: The coordinates of the current raster that will be mapped to
            the lower-right corner of the resampled raster.
        :type lr: array-like (2,)

        :param scale: Scale factor applied during resampling. Setting scale to
            < 1 correspond to downsampling.
        :type downsample: float

        :param interpolation: Nearest neighbor (0), linear (1),
        :type interpolation: str {'nearest_neighbor', 'average', 'bilinear',
                                  'cubic', 'cubic_spline', 'lancosz'}

        """
        if ul is None:
            ul = (0,0)

        if lr is None:
            lr = (self.res_x, self.res_y)

        # Perform the projection/resampling
        gdal_drv = gdal.GetDriverByName('MEM')
        x_size = int(np.round((lr[0] - ul[0])*scale))
        y_size = int(np.round((lr[1] - ul[1])*scale))
        dest_ds = gdal_drv.Create('', x_size, y_size, self.gdal_ds.RasterCount,
                                  numpy_to_gdal_type(self.raster.dtype))
        dest_ds.SetProjection(self.gdal_ds.GetProjection())
        geotrans = list(self.gdal_ds.GetGeoTransform())
        geotrans[0] += geotrans[1]*ul[0] + geotrans[2]*ul[1]
        geotrans[3] += geotrans[4]*ul[0] + geotrans[5]*ul[1]

        geotrans[1] /= scale
        geotrans[2] /= scale
        geotrans[4] /= scale
        geotrans[5] /= scale
        dest_ds.SetGeoTransform(geotrans)
        gdal.ReprojectImage(self.gdal_ds, dest_ds,
                            self.gdal_ds.GetProjection(),
                            dest_ds.GetProjection(),
                            gdal_interpolation[interpolation])

        self.gdal_ds = dest_ds

    def show(self, coordinates='meters', colormap=None):
        font = {'size' : 26}
        plt.rc('font', **font)
        plt.rc('axes', linewidth=4)
        if coordinates == 'meters':
            left, top = self.raster_to_meters([0, 0])[:2]
            right, bottom = self.raster_to_meters([self.res_x, self.res_y])[:2]

            plt.imshow(self.raster, aspect='equal', interpolation='nearest',
                       extent=[left, right, bottom, top], cmap=colormap)
            plt.xlabel('Easting (meters)', fontsize=32)
            plt.ylabel('Northing (meters)', fontsize=32)
        else:
            raise NotImplementedError()

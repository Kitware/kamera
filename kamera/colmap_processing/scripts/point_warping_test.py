from __future__ import division, print_function
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from osgeo import osr, gdal
import PIL

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

camera_model1 = '/home/user/camera_model1.yaml'
camera_model2 = '/home/user/camera_model2.yaml'
base_layer_fname = '/home/user/3d_models/base_layer.tif'
ponts_fname = '/home/user/points.txt'

latitude0 = 0    # degrees
longitude0 = 0  # degrees
height0 = 0


def load_camera_model(camera_model):
    ret = load_static_camera_from_file(camera_model)
    K, d, R, depth_map, latitude, longitude, altitude = ret[2:]
    img_fname = '%s/ref_view.png' % os.path.split(camera_model)[0]
    image = cv2.imread(img_fname)
    return image, K, d, R, depth_map, latitude, longitude, altitude

image1, K1, d1, R1, depth_map1, latitude1, longitude1, altitude1 = load_camera_model(camera_model1)
image2, K2, d2, R2, depth_map2, latitude2, longitude2, altitude2 = load_camera_model(camera_model2)


def unproject_from_camera(im_pts, K, d, R, depth_map, latitude, longitude, altitude):
    # Unproject rays into the camera coordinate system.
    ray_dir = np.ones((3, len(im_pts)), dtype=np.float)
    ray_dir0 = cv2.undistortPoints(np.expand_dims(im_pts, 0), K, d, R=None)
    ray_dir[:2] = np.squeeze(ray_dir0, 0).T

    enu0 = llh_to_enu(latitude, longitude, altitude, latitude0, longitude0,
                      height0)
    enu0 = np.array(enu0)

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

    return enu


def project_to_camera(wrld_pts, K, d, R, depth_map, latitude, longitude, altitude):
    # Unproject rays into the camera coordinate system.
    cam_pos = llh_to_enu(latitude, longitude, altitude, latitude0, longitude0,
                         height0)
    tvec = -np.dot(R, cam_pos).ravel()
    rvec = cv2.Rodrigues(R)[0]
    im_pts = cv2.projectPoints(wrld_pts, rvec, tvec, K, d)[0]
    im_pts = np.squeeze(im_pts)

    return im_pts


class GeoImage(object):
    """Representation of a georeferenced image.

    """
    def __init__(self):
        self._dem = None

    @staticmethod
    def load_geotiff(fname):
        self = GeoImage()

        if fname is None:
            return

        if not os.path.isfile(fname):
            print('Could not open file \'%s\'' % fname)
            return

        ds = gdal.Open(fname, gdal.GA_ReadOnly)
        if ds.RasterCount > 1:
            raw_image = np.zeros((ds.RasterYSize,ds.RasterXSize,3), dtype=np.uint8)
            for i in range(3):
                raw_image[:,:,i] = ds.GetRasterBand(i+1).ReadAsArray()
        else:
            band = ds.GetRasterBand(1)
            raw_image = band.ReadAsArray()

        self._array = raw_image

        # Create lat/lon coordinate system.
        wgs84_wkt = """
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]"""
        wgs84_cs = osr.SpatialReference()
        wgs84_cs.ImportFromWkt(wgs84_wkt)

        image_cs= osr.SpatialReference()
        image_cs.ImportFromWkt(ds.GetProjectionRef())

        # Create a transform object to convert between projection of the image
        # and WGS84 coordinates.
        self._to_lla_tform = osr.CoordinateTransformation(image_cs, wgs84_cs)
        self._from_lla_tform = osr.CoordinateTransformation(wgs84_cs, image_cs)

        # See documentation for gdal GetGeoTransform().
        geo_tform_mat = np.identity(3)
        c = ds.GetGeoTransform()
        geo_tform_mat[0,0] = c[1]
        geo_tform_mat[0,1] = c[2]
        geo_tform_mat[0,2] = c[0]
        geo_tform_mat[1,0] = c[4]
        geo_tform_mat[1,1] = c[5]
        geo_tform_mat[1,2] = c[3]
        self.geo_tform_mat = geo_tform_mat[:2]
        self.geo_inv_tform_mat = np.linalg.inv(geo_tform_mat)[:2]

        # Try to load DEM if available.
        try:
            fname, ext = os.path.splitext(fname)
            depth_map = np.asarray(PIL.Image.open('%s.dem.tif' % fname))
            self._dem = depth_map
            print('Loaded')
        except (OSError, IOError):
            self._dem = None

        return self

    @property
    def array(self):
        return self._array

    def get_lon_lat_from_im_pt(self, im_pt):
        """Return latitude and longitude for image point.

        :param pos: Raw image coordinates of the geotiff that were clicked.
        :type pos: 2-array

        :return: Longitude (degrees) and latitude (degrees) associated with
            the clicked point.
        :rtype: 3-array

        """
        # Convert from image coordinates to the coordinates of the projection
        # encoded in the geotiff.
        Xp,Yp = np.dot(self.geo_tform_mat, [im_pt[0],im_pt[1],1])
        lon,lat,_ = self._to_lla_tform.TransformPoint(Xp, Yp)

        if self._dem is not None:
            height, width = self._dem.shape[:2]
            x, y = im_pt
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

            h = self._dem[iy, ix]

            return lon, lat, h

        return lon, lat

    def get_im_pt_from_lon_lat(self, lon, lat):
        """Return image coordinates for latitude and longitude.

        :param lon: Longitude (degrees).
        :type lon: float

        :param lat: Latitude (degrees).
        :type lat: float

        :return: Raw image coordinates of the geotiff.
        :rtype: 2-array

        """
        Xp,Yp,_ = self._from_lla_tform.TransformPoint(lon, lat)

        # Convert from image coordinates to the coordinates of the projection
        # encoded in the geotiff.
        im_pt = np.dot(self.geo_inv_tform_mat, [Xp,Yp,1])
        return im_pt


base_layer = GeoImage.load_geotiff(base_layer_fname)

# ----------------------------------------------------------------------------
# Clicked points in both camera1 and base layer.
res = np.loadtxt('/home/user/geo_points_test.txt')
im_pts0 = res[:, :2]
llh0 = res[:, 2:]

wrld_pts = np.array([llh_to_enu(_[0], _[1], _[2], latitude0, longitude0, height0)
                     for _ in llh0])
im_pts = project_to_camera(wrld_pts, K1, d1, R1, depth_map1, latitude1, longitude1, altitude1)
plt.imshow(image1)
#plt.plot(im_pts0[:, 0], im_pts0[:, 1], 'go')
plt.plot(im_pts[:, 0], im_pts[:, 1], 'ro')


# Project from camera 1 into the world.
wrld_pts1 = unproject_from_camera(im_pts0, K1, d1, R1, depth_map1, latitude1, longitude1, altitude1)
im_pts = project_to_camera(wrld_pts, K1, d1, R1, depth_map1, latitude1, longitude1, altitude1)
plt.plot(im_pts[:, 0], im_pts[:, 1], 'bo')

# Compare 3-D points selected from base-layer image versus unprojected from
# camera.
plt.figure()
plt.plot(wrld_pts[:, 0], wrld_pts[:, 1], 'go')
plt.plot(wrld_pts1[:, 0], wrld_pts1[:, 1], 'bo')








# Project from camera 2 into the world.
wrld_pts = unproject_from_camera(im_pts0, K1, d1, R1, depth_map1, latitude1, longitude1, altitude1)
llh = [enu_to_llh(_[0], _[1], _[2], latitude0, longitude0, height0)
        for _ in wrld_pts]
base_pts1 = [base_layer.get_im_pt_from_lon_lat(llh[1], llh[0]) for llh in llh]
base_pts1 = np.array(base_pts1)
base_pts0 = [base_layer.get_im_pt_from_lon_lat(llh[1], llh[0]) for llh in llh0]
base_pts0 = np.array(base_pts0)

plt.figure()
plt.imshow(base_layer.array)
plt.plot(base_pts1[:, 0], base_pts1[:, 1], 'go')
plt.plot(base_pts0[:, 0], base_pts0[:, 1], 'ro')


# ----------------------------------------------------------------------------

points = np.loadtxt(ponts_fname)
pts1 = points[:, :2]
pts2 = points[:, 2:]

# Project from camera 1 into the world.
wrld_pts1 = unproject_from_camera(pts1, K1, d1, R1, depth_map1, latitude1, longitude1, altitude1)

# Project from camera 2 into the world.
wrld_pts2 = unproject_from_camera(pts2, K2, d2, R2, depth_map2, latitude2, longitude2, altitude2)

plt.close('all')
plt.figure()
plt.imshow(base_layer.array)

llh1 = [enu_to_llh(_[0], _[1], _[2], latitude0, longitude0, height0)
        for _ in wrld_pts1]
base_pts1 = [base_layer.get_im_pt_from_lon_lat(llh[1], llh[0]) for llh in llh1]
base_pts1 = np.array(base_pts1)
plt.plot(base_pts1[:, 0], base_pts1[:, 1], 'go')

# Plot points from image 2 on base layer.
llh2 = [enu_to_llh(_[0], _[1], _[2], latitude0, longitude0, height0)
        for _ in wrld_pts2]
base_pts2 = [base_layer.get_im_pt_from_lon_lat(llh[1], llh[0]) for llh in llh2]
base_pts2 = np.array(base_pts2)
plt.plot(base_pts2[:, 0], base_pts2[:, 1], 'ro')



plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.plot(pts1[:, 0], pts1[:, 1], 'go')


#wrld_pts = unproject_from_camera(pts1, K1, d1, R1, depth_map1, latitude1, longitude1, altitude1)
im_pts = project_to_camera(wrld_pts, K1, d1, R1, depth_map1, latitude1, longitude1, altitude1)
plt.plot(im_pts[:, 0], im_pts[:, 1], 'ro')

plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.plot(pts2[:, 0], pts2[:, 1], 'go')

# Project from camera 1 onto camera 2.
im_pts = project_to_camera(wrld_pts, K2, d2, R2, depth_map2, latitude2, longitude2, altitude2)
plt.plot(im_pts[:, 0], im_pts[:, 1], 'ro')
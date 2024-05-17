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
import trimesh
import math
import PIL
from osgeo import osr, gdal
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

import colmap_processing.vtk_util as vtk_util
from colmap_processing.geo_conversions import enu_to_llh


# ----------------------------------------------------------------------------
if True:
    mesh_fname = '/enu.ply'
    save_dir = '/out'

    # Latitude and longitude associated with (0, 0, 0) in the model.
    latitude0 = 0      # degrees
    longitude0 = 0     # degrees
    height0 = 0            # meters above WGS84 ellipsoid

    # Set GSD of orthographic base layer.
    gsd = 0.05  # meters

    determine_dem = False


# VTK renderings are limited to monitor resolution (width x height).
monitor_resolution = (800, 800)
# ----------------------------------------------------------------------------


# Determine the bounds of the model.
mesh = trimesh.load(mesh_fname)
pts = mesh.vertices.T
model_bounds = np.array([[min(pts[i]), max(pts[i])] for i in range(3)])

if False:
    w = np.diff(model_bounds[0])
    h = np.diff(model_bounds[1])
    w = 400
    h = 450
    c = np.sum(np.array(model_bounds[:2]), 1)/2
    model_bounds = np.array([[c[0] - w/2, c[0] + w/2],
                             [c[1] - h/2, c[1] + h/2],
                             model_bounds[2]])
    model_bounds = np.array([[-220, 180], [-240, 210], model_bounds[2]])

delta_xyz = np.diff(model_bounds)

# Read model into VTK.
model_reader = vtk_util.load_world_model(mesh_fname)


def get_ortho_image(xbnds, ybnds, res_x, res_y, from_bottom=False):
    # Model orthographic as camera that is 'alt' meters high.
    alt = 1e5 + float(delta_xyz[2]) + model_bounds[2, 1]
    #alt = 1e3

    # Position of the camera.
    pos = [np.mean(xbnds), np.mean(ybnds), model_bounds[2, 1] + alt]

    assert res_x < monitor_resolution[0]
    assert res_y < monitor_resolution[1]

    dy = float(np.abs(np.diff(ybnds)))
    vfov = 2*np.arctan(dy/2/alt)*180/np.pi

    if from_bottom:
        pos[2] *= -1
        pan = 180
        tilt = 90
    else:
        pan = 0
        tilt = -90

    ortho_camera = vtk_util.CameraPanTilt(res_x, res_y, vfov, pos, pan, tilt)

    clipping_range = [alt, alt + float(delta_xyz[2]) + 1]
    #clipping_range = [1e3, 2e4]
    img = ortho_camera.render_image(model_reader,
                                    clipping_range=clipping_range,
                                    diffuse=0.6, ambient=0.6, specular=0.1,
                                    light_color=[1.0, 1.0, 1.0],
                                    light_pos=[0,0,1000])

    ret = ortho_camera.unproject_view(model_reader,
                                      clipping_range=clipping_range)
    z = ret[2] + height0

    if from_bottom:
        img = np.fliplr(img)
        z = np.fliplr(z)

    return img, z


# Build up geotiff by tiling the rendering.
full_res_x = int(math.ceil(delta_xyz[0]/gsd))
full_res_y = int(math.ceil(delta_xyz[1]/gsd))
ortho_image = np.zeros((full_res_y, full_res_x, 3), dtype=np.uint8)
dem = np.zeros((full_res_y, full_res_x), dtype=np.float)

if determine_dem:
    underside_dem = np.zeros((full_res_y, full_res_x), dtype=np.float)

num_cols = int(math.ceil(full_res_x/monitor_resolution[0]))
num_rows = int(math.ceil(full_res_y/monitor_resolution[1]))
indr = np.round(np.linspace(0, full_res_y, num_rows + 1)).astype(np.int)
indc = np.round(np.linspace(0, full_res_x, num_cols + 1)).astype(np.int)
xbnds_array = (indc/full_res_x)*delta_xyz[0] + model_bounds[0, 0]
ybnds_array = model_bounds[1, 1] - (indr/full_res_y)*delta_xyz[1]

for i in range(num_rows):
    for j in range(num_cols):
        ret = get_ortho_image(xbnds_array[j:j+2],
                              ybnds_array[i:i+2],
                              int(indc[j+1] - indc[j]),
                              int(indr[i+1] - indr[i]))
        ortho_image[indr[i]:indr[i+1], indc[j]:indc[j+1], :] = ret[0]
        dem[indr[i]:indr[i+1], indc[j]:indc[j+1]] = ret[1]

        if determine_dem:
            ret = get_ortho_image(xbnds_array[j:j+2],
                                  ybnds_array[i:i+2],
                                  int(indc[j+1] - indc[j]),
                                  int(indr[i+1] - indr[i]), from_bottom=True)
            underside_dem[indr[i]:indr[i+1], indc[j]:indc[j+1]] = ret[1]


print('DEM elevation ranges from %0.4f to %0.4f' %
      (dem.ravel().min(), dem.ravel().max()))


# ----------------------------------------------------------------------------
if False:
    # Determine the DEM for the ground level.
    # Radius of curvature
    scale = 10 # m

    xmin = float(pts[0].min()); xmax = float(pts[0].max())
    ymin = float(pts[1].min()); ymax = float(pts[1].max())
    xc = (xmin + xmax)/2
    yc = (ymin + ymax)/2
    zmax = 1000000000
    pts1 = np.array(pts.copy())
    tmp = np.array([[xmin, xmin, xmax, xmax],
                    [ymin, ymax, ymax, ymin],
                    [zmax + 1, zmax + 1, zmax + 1, zmax + 1]])
    pts1 = np.hstack([pts1, tmp])
    pts_ = pts1.copy()

    r = np.sqrt((xmax - xc)**2 + (ymax - yc)**2) + 1e-6
    pts_[2] = pts_[2] - scale*np.sqrt(r**2 - (pts1[0] - xc)**2 - (pts1[1] - yc)**2)

    hull = ConvexHull(pts_.T)

    #plt.plot(pts[0, hull.vertices], pts[1, hull.vertices], 'r.')

    x = np.linspace(xmin, xmax, full_res_x + 1)
    x = (x[1:] + x[:-1])/2
    y = np.linspace(ymax, ymin, full_res_y + 1)
    y = (y[1:] + y[:-1])/2
    X, Y = np.meshgrid(x, y)

    vertices = hull.vertices
    vertices = vertices[vertices < pts.shape[1]]
    dem1 = griddata(pts1[:2, vertices].T, pts1[2, vertices],
                    (X, Y), method='linear', fill_value=0) + height0

    fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
    plt.imshow(dem1); plt.colorbar()
    fig.tight_layout()
    plt.savefig('%s/ground_dem.jpg' % save_dir)
    fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
    plt.imshow(dem); plt.colorbar()
    fig.tight_layout()
    plt.savefig('%s/dem.jpg' % save_dir)

    mask = dem == dem.min()
    diff = dem - dem1
    diff[mask] = 0
    fig = plt.figure(num=None, figsize=(15.3, 10.7), dpi=80)
    plt.imshow(diff); plt.colorbar()
    fig.tight_layout()
    plt.savefig('%s/just_buildings.jpg' % save_dir)

    im = PIL.Image.fromarray(dem1.astype(np.float32), mode='F') # float32
    depth_map_fname = '%s/base_layer.ground.dem.tif' % save_dir
    im.save(depth_map_fname)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Identify holes in the model and then inpaint them.
hole_mask = dem < dem.ravel().min() + 0.1

output = cv2.connectedComponentsWithStats(hole_mask.astype(np.uint8), 8, cv2.CV_32S)
num_labels = output[0]
labels = output[1]
stats = output[2]
centroids = output[3]

# Remove components that touch outer boundary.
edge_labels = set(labels[:, 0])
edge_labels = edge_labels.union(set(labels[0, :]))
edge_labels = edge_labels.union(set(labels[:, -1]))
edge_labels = edge_labels.union(set(labels[-1, :]))

for i in edge_labels:
    labels[labels == i] = 0

mask = (labels > 0).astype(np.uint8)
ortho_image = cv2.inpaint(ortho_image, mask, 3, cv2.INPAINT_NS)
dem = cv2.inpaint(dem.astype(np.float32), mask, 3, cv2.INPAINT_NS)

im = PIL.Image.fromarray(dem.astype(np.float32), mode='F') # float32
depth_map_fname = '%s/base_layer.dem.tif' % save_dir
im.save(depth_map_fname)
# ----------------------------------------------------------------------------


def clahe(img, clim=2):
#    img = img.astype(float)
#    img -= np.percentile(img.ravel(), 0.1)
#    img[img < 0] = 0
#    img /= np.percentile(img.ravel(), 99.9)/255
#    img[img > 255] = 255
#    img = np.round(img).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clim, tileGridSize=(128, 128))
    if img.ndim == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        return clahe.apply(img)


def equalize_hist(img):
    img = img.astype(float)
    img -= np.percentile(img.ravel(), 0.1)
    img[img < 0] = 0
    img /= np.percentile(img.ravel(), 99.9)/255
    img[img > 255] = 255
    img = np.round(img).astype(np.uint8)
    if img.ndim == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        return clahe.apply(img)


#ortho_image = ortho_image0.copy()
ortho_image = clahe(ortho_image)
#ortho_image = equalize_hist(ortho_image)


# ----------------------------------------------------------------------------
# Save GeoTIFF
geotiff_fname = '%s/base_layer.tif' % save_dir
gdal_drv = gdal.GetDriverByName('GTiff')
wgs84_cs = osr.SpatialReference()
wgs84_cs.SetWellKnownGeogCS("WGS84")
wgs84_wkt = wgs84_cs.ExportToPrettyWkt()
gdal_settings = ['COMPRESS=JPEG', 'JPEG_QUALITY=%i' % 90]
#gdal_settings = []
ds = gdal_drv.Create(geotiff_fname, ortho_image.shape[1], ortho_image.shape[0],
                     ortho_image.ndim, gdal.GDT_Byte, gdal_settings)
ds.SetProjection(wgs84_cs.ExportToWkt())

enus = [[model_bounds[0, 0], model_bounds[1, 1], 0],
        [model_bounds[0, 1], model_bounds[1, 0], 0]]
lat_lon = [enu_to_llh(enu[0], enu[1], enu[2], latitude0, longitude0, 0)
           for enu in enus]
lat_lon = np.array(lat_lon)[:, :2]

A = np.zeros((2, 3))
A[:, 2] = lat_lon[0][::-1]

dll = lat_lon[1] - lat_lon[0]
A[0, 0] = dll[1] / full_res_x
A[1, 1] = dll[0] / full_res_y

# Xp = padfTransform[0] + P*padfTransform[1] + L*padfTransform[2]
# Yp = padfTransform[3] + P*padfTransform[4] + L*padfTransform[5]
geotrans = [A[0, 2], A[0, 0], A[0, 1], A[1, 2], A[1, 0], A[1, 1]]
ds.SetGeoTransform(geotrans)

if ds.RasterCount == 1:
    ds.GetRasterBand(1).WriteArray(ortho_image[:, :], 0, 0)
else:
    for i in range(ds.RasterCount):
        ds.GetRasterBand(i+1).WriteArray(ortho_image[:, :, i], 0, 0)

ds.FlushCache()  # Write to disk.
ds = None
# ----------------------------------------------------------------------------
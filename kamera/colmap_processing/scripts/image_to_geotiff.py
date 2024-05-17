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


src_fname = ''
geotiff_fname = ''

# Each row is Latitude (deg) and Longitude (deg)
lat_lon =
lat_lon = np.array(lat_lon)

# Each row is the associated image coordinates.
im_pts =
im_pts = np.array(im_pts)

# ----------------------------------------------------------------------------
ortho_image = cv2.imread(src_fname)

if ortho_image.ndim == 3:
    ortho_image = ortho_image[:, :, ::-1]

# Save GeoTIFF
gdal_drv = gdal.GetDriverByName('GTiff')
wgs84_cs = osr.SpatialReference()
wgs84_cs.SetWellKnownGeogCS("WGS84")
wgs84_wkt = wgs84_cs.ExportToPrettyWkt()
gdal_settings = ['COMPRESS=JPEG', 'JPEG_QUALITY=%i' % 99]
#gdal_settings = []
ds = gdal_drv.Create(geotiff_fname, ortho_image.shape[1], ortho_image.shape[0],
                     ortho_image.ndim, gdal.GDT_Byte, gdal_settings)
ds.SetProjection(wgs84_cs.ExportToWkt())


A, mask = cv2.estimateAffine2D(im_pts.reshape(-1, 1, 2),
                               lat_lon[:, ::-1].reshape(-1, 1, 2),
                               method=cv2.RANSAC,
                               ransacReprojThreshold=10000)

H, mask = cv2.findHomography(im_pts.reshape(-1, 1, 2),
                             lat_lon[:, ::-1].reshape(-1, 1, 2),
                             method=cv2.RANSAC,
                             ransacReprojThreshold=1)

# An affine transformation might not be enough to correct. So, let's warp image
# so that it is sufficient.
A_ = np.vstack([A, [0, 0, 1]])
H_ = np.dot(np.linalg.inv(A_), H)
ortho_image_ = cv2.warpPerspective(ortho_image, H_,
                                   dsize=ortho_image.shape[:2][::-1],
                                   flags=cv2.INTER_CUBIC)

# Xp = padfTransform[0] + P*padfTransform[1] + L*padfTransform[2]
# Yp = padfTransform[3] + P*padfTransform[4] + L*padfTransform[5]
geotrans = [A[0, 2], A[0, 0], A[0, 1], A[1, 2], A[1, 0], A[1, 1]]
ds.SetGeoTransform(geotrans)

if ds.RasterCount == 1:
    ds.GetRasterBand(1).WriteArray(ortho_image_[:, :], 0, 0)
else:
    for i in range(ds.RasterCount):
        ds.GetRasterBand(i+1).WriteArray(ortho_image_[:, :, i], 0, 0)

ds.FlushCache()  # Write to disk.
ds = None
# ----------------------------------------------------------------------------
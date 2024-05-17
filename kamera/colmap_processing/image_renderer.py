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

Note: the image coordiante system has its origin at the center of the top left
pixel.

"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp2d, RectBivariateSpline
import PIL


def save_gif(images, fname, duration=500,
                          loop=0):
    """
    :param img1
    :type img1:

    :param img2:
    :type img2:

    :param fname:
    :type fname:

    :param duration: Time (milliseconds) to move between frames.
    :type duration: float

    :param loop: Number of times to loop the GIF (0 loops forever).
    :type loop: int
    """
    images = [PIL.Image.fromarray(img) for img in images]
    images[0].save(fname, save_all=True, append_images=images[1:],
          duration=duration, loop=loop)


def warp_perspective(image, h, dsize, interpolation=0, use_pyr=True,
                     precrop=True):
    """
    :param h: Homography that takes output image coordinates and returns
        source image coordinates.
    :type h: 3x3 numpy.ndarray

    :param dsize: Width and height of the warped image.
    :type dsize: 2-array tuple/list

    :param interpolation: Interpolation method.
    :type interpolation: integer in range 0-4

    :param use_pyr: Specify whether to use an appropriate level of the
        image pyramid decomposition of the source image in order to avoid
        aliasing.
    :type use_pyr: bool

    :param precrop: For very large source images, running the warp
        operation on the full image might be slowed due to memory access.
        Setting this to True will initially crop out only the region of the
        image that is gets used during warping to speed up the operation.
    :type precrop: bool

    """
    res_y, res_x = image.shape[:2]

    if interpolation == 0:
        interpolation = cv2.INTER_NEAREST
    elif interpolation == 1:
        interpolation = cv2.INTER_LINEAR
    elif interpolation == 2:
        interpolation = cv2.INTER_AREA
    elif interpolation == 3:
        interpolation = cv2.INTER_CUBIC
    elif interpolation == 4:
        interpolation = cv2.INTER_LANCZOS4

    if precrop:
        # Select points from corners of output image.
        im_pts = np.ones((3,4), np.float32)
        im_pts[0] = [0,1,1,0]
        im_pts[0] *= dsize[0]
        im_pts[1] = [0,0,1,1]
        im_pts[1] *= dsize[1]
        src_im_pts = np.dot(h, im_pts)
        src_im_pts[0] /= src_im_pts[2]
        src_im_pts[1] /= src_im_pts[2]

        # Collect the bounding box on src_img
        x_range = np.array([src_im_pts[0].min(),src_im_pts[0].max()])
        y_range = np.array([src_im_pts[1].min(),src_im_pts[1].max()])

        # Clamp to actual image dimensions if exceeded.
        # Pad by p for interpolation.
        p = 8
        x_range[0] = np.maximum(0, x_range[0]-p)
        x_range[1] = np.minimum(res_x, x_range[1]+p)
        y_range[0] = np.maximum(0, y_range[0]-p)
        y_range[1] = np.minimum(res_y, y_range[1]+p)

        x_range = np.round(x_range).astype(np.int)
        y_range = np.round(y_range).astype(np.int)

        # Want a homography that maps from the full version of image to the
        # cropped version defined by x_range and y_range.
        h_crop = np.identity(3)
        h_crop[:2,2] = -x_range[0], -y_range[0]
        h = np.dot(h_crop, h)

        image = image[y_range[0]:y_range[1],x_range[0]:x_range[1]]

        if 0 in image.shape:
            if image.ndim == 3:
                return np.zeros((dsize[1],dsize[0],3), image.dtype)
            else:
                return np.zeros((dsize[1],dsize[0]), image.dtype)

    flags = interpolation | cv2.WARP_INVERSE_MAP
    warped_image = cv2.warpPerspective(image, h, dsize=dsize, flags=flags)
    return warped_image


def render_view(src_cm, src_img, src_t, dst_cm, dst_t, interpolation=1,
                block_size=1, homog_approx=False, world_model=None):
    """Render view onto destination camera from a source camera image.

    :param src_cm: Camera model for the camera that acquired src_img.
    :type src_cm:

    :param src_img: Source image used to render the view on the destination
        camera.
    :type src_img:

    :param src_t: Time at which src_image was acquired (time in seconds since
        Unix epoch).
    :type src_t: float

    :param dst_cm: Camera model for the camera that the view is generated.
    :type dst_cm:

    :param dst_t: Time at which dst_image would be acquired (time in seconds
        since Unix epoch).
    :type dst_t: float

    :param interpolation: Interpolation method (0=nearest, 1=linear, 2=area,
        3=cubic, 4=Lanczos4)
    :type interpolation: integer in range 0-4

    :param block_size: Calculating the full rigorous projection mapping for
        each pixel is expensive and often unnecessary. We can instead sample
        the projection mapping every 'block_size' pixels and then interpolate
        the result in between. Note, this only applies if homog_approx is
        False.
    :type block_size: positive int

    :param homog_approx: Approximate the mapping from azimuth/elevation to
            the camera image with a homography.
    :type homog_approx: bool

    :param surface_distance: Distance to the surface of the world being imaged
        (meters).
    :type surface_distance: float

    :return: Rendered image that would have been seen by the destination
        camera and a mask of valid pixels.
    :rtype: [numpy.ndarray, bool numpy.ndarray]

    """
    if world_model is None:
        surface_distance = 1e5

    if src_img is not None:
        if src_cm.width != src_img.shape[1] or src_cm.height != src_img.shape[0]:
            print('Camera model for image topic %s with encoded image '
                  'size %i x %i is being used to render images with '
                  'size %i x %i' % (src_cm.image_topic,src_cm.width,
                                    src_cm.height,src_img.shape[1],
                                    src_img.shape[0]))

    def get_mask(X, Y, edge_buffer=4):
        mask = np.logical_and(X > edge_buffer, Y > edge_buffer)
        mask = np.logical_and(mask, X < src_cm.width - edge_buffer)
        mask = np.logical_and(mask, Y < src_cm.height - edge_buffer)
        return mask

    # ------------------------------------------------------------------------
    if homog_approx:
        if False:
            # Select points from a square centered on the focal plane.
            im_pts = np.zeros((2,4), np.float32)
            im_pts[0] = [0.25,0.75,0.75,0.25]
            im_pts[0] *= dst_cm.width
            im_pts[1] = [0.25,0.25,0.75,0.75]
            im_pts[1] *= dst_cm.height
        else:
            # Select points from focal plane corners.
            im_pts = np.zeros((2,4), np.float32)
            im_pts[0] = [0,1,1,0]
            im_pts[0] *= dst_cm.width
            im_pts[1] = [0,0,1,1]
            im_pts[1] *= dst_cm.height

        # Unproject rays into camera coordinate system.
        ray_pos, ray_dir = dst_cm.unproject(im_pts, dst_t)

        if world_model is not None:
            points = world_model.intersect_rays(ray_pos, ray_dir)
        else:
            # Project the ray out to "infinity" (well, surface_distance).
            ray_dir *= surface_distance
            points = ray_pos + ray_dir

        im_pts_src = src_cm.project(points, src_t).astype(np.float32)

        h = cv2.findHomography(im_pts.T, im_pts_src.T)[0]
        #np.dot(h, [res_x/2,res_y/2,1])
        dst_img = warp_perspective(src_img, h, (dst_cm.width, dst_cm.height),
                                   interpolation=interpolation)
        mask = np.ones((dst_img.shape[0],dst_img.shape[1]), dtype=np.bool)
        return dst_img, mask
    else:
        if interpolation == 0:
            interpolation = cv2.INTER_NEAREST
        elif interpolation == 1:
            interpolation = cv2.INTER_LINEAR
        elif interpolation == 2:
            interpolation = cv2.INTER_AREA
        elif interpolation == 3:
            interpolation = cv2.INTER_CUBIC
        elif interpolation == 4:
            interpolation = cv2.INTER_LANCZOS4

        if block_size == 1:
            # Densely sample all pixels.
            x = np.linspace(0, dst_cm.width-1, dst_cm.width)
            y = np.linspace(0, dst_cm.height-1, dst_cm.height)
            X,Y = np.meshgrid(x, y)
            im_pts = np.vstack([X.ravel(),Y.ravel()])

            # Unproject rays into camera coordinate system.
            ray_pos, ray_dir = dst_cm.unproject(im_pts, dst_t)

            if world_model is not None:
                points = world_model.intersect_rays(ray_pos, ray_dir)
            else:
                # Project the ray out to "infinity" (well, surface_distance).
                ray_dir *= surface_distance
                points = ray_pos + ray_dir

            im_pts_src = src_cm.project(points, src_t).astype(np.float32)

            X = np.reshape(im_pts_src[0], (dst_cm.height,dst_cm.width))
            Y = np.reshape(im_pts_src[1], (dst_cm.height,dst_cm.width))
        else:
            # Sample the full projection equation every 'blocksize' pixels and
            # interpolate in between.
            nx = dst_cm.width//block_size
            ny = dst_cm.height//block_size
            x = np.linspace(0, dst_cm.width-1, nx)
            y = np.linspace(0, dst_cm.height-1, ny)
            X,Y = np.meshgrid(x, y)
            im_pts = np.vstack([X.ravel(),Y.ravel()])

            # Unproject rays into camera coordinate system.
            ray_pos, ray_dir = dst_cm.unproject(im_pts, dst_t)

            if world_model is not None:
                points = world_model.intersect_rays(ray_pos, ray_dir)
            else:
                # Project the ray out to "infinity" (well, surface_distance).
                ray_dir *= surface_distance
                points = ray_pos + ray_dir

            im_pts_src = src_cm.project(points, src_t).astype(np.float32)

            X = np.reshape(im_pts_src[0], (ny,nx))
            Y = np.reshape(im_pts_src[1], (ny,nx))

            if True:
                fx = interp2d(x, y, X)
                fy = interp2d(x, y, Y)
            else:
                # RectBivariateSpline is much faster than interp2d on a
                # rectangular grid.
                # Order of the interpolation
                n = 1
                fx = RectBivariateSpline(x, y, X.T, kx=n, ky=n)
                fy = RectBivariateSpline(x, y, Y.T, kx=n, ky=n)

            # Evaluate on dense sampling.
            xd = np.linspace(0, dst_cm.width-1, dst_cm.width)
            yd = np.linspace(0, dst_cm.height-1, dst_cm.height)
            X = fx(xd, yd).astype(np.float32)
            Y = fy(xd, yd).astype(np.float32)

        dst_img = cv2.remap(src_img, X, Y, interpolation)

        # Set mask to False any place where the src image coordinates are
        # within edge_buffer from the edge.
        mask = get_mask(X, Y, edge_buffer=4)

        return dst_img, mask


def show_warped_points(src_cm, src_img, src_t, dst_cm, dst_t, block_size=1):
    """Plot out points to be samples in the source image.

    """
    if block_size == 1:
        # Densely sample all pixels.
        x = np.linspace(0, dst_cm.width-1, dst_cm.width)
        y = np.linspace(0, dst_cm.height-1, dst_cm.height)
        X,Y = np.meshgrid(x, y)
        im_pts = np.vstack([X.ravel(),Y.ravel()])

        # Unproject rays into camera coordinate system.
        ray_pos, ray_dir = dst_cm.unproject(im_pts, dst_t)
        ray_dir *= 1000

        im_pts_src = src_cm.project(ray_dir, src_t).astype(np.float32)
    else:
        # Sample the full projection equation every 'blocksize' pixels and
        # interpolate in between.

        nx = dst_cm.width//block_size
        ny = dst_cm.height//block_size
        x = np.linspace(0, dst_cm.width-1, nx),
        y = np.linspace(0, dst_cm.height-1, ny)
        X,Y = np.meshgrid(x, y)
        im_pts = np.vstack([X.ravel(),Y.ravel()])

        # Unproject rays into camera coordinate system.
        ray_pos, ray_dir = dst_cm.unproject(im_pts, dst_t)
        ray_dir *= 1000

        im_pts_src = src_cm.project(ray_dir, src_t).astype(np.float32)

    plt.plot(im_pts_src[0], im_pts_src[1], 'ro')


def stitch_images(src_list, dst_cm, dst_t, interpolation=1, block_size=1,
                  homog_approx=False, world_model=None):
    """
    :param src_list: List of frames and cameras of the form [src_cm, src_img,
        src_t].
    :type src_list: [[Camera, Numpy array, float], ...]

    :param dst_cm: Camera to render the view onto.
    :type dst_cm: Camera

    :param dst_t: Time to at which to assume the navigation state of the
        destination camera for rendering.
    :type dst_t: float

    :param interpolation: Interpolation method (0=nearest, 1=linear, 2=area,
        3=cubic, 4=Lanczos4)
    :type interpolation: integer in range 0-4

    :param block_size: Calculating the full rigorous projection mapping for
        each pixel is expensive and often unnecessary. We can instead sample
        the projection mapping every 'block_size' pixels and then interpolate
        the result in between. Note, this only applies if homog_approx is
        False.
    :type block_size: positive int

    :param homog_approx: Approximate the mapping from azimuth/elevation to
            the camera image with a homography.
    :type homog_approx: bool

    """
    out_dtype = src_list[0][1].dtype
    if out_dtype == np.uint8:
        out_channels = 3
        dst_img = np.zeros((dst_cm.height, dst_cm.width, out_channels),
                           out_dtype)
    else:
        out_channels = 1
        dst_img = np.zeros((dst_cm.height, dst_cm.width), out_dtype)

    for i in range(len(src_list)):
        src_cm, src_img, src_t = src_list[i]

        #show_warped_points(src_cm, src_img, src_t, dst_cm, dst_t, 10)

        img, mask = render_view(src_cm, src_img, src_t, dst_cm, dst_t,
                                interpolation=interpolation,
                                block_size=block_size,
                                homog_approx=homog_approx,
                                world_model=world_model)

        if out_channels == 3:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Couldn't find a more elegant/efficient way to hangle a 2-D mask with
            # an RGB image.
            mask2 = np.zeros_like(dst_img, dtype=np.bool)
            mask2[:,:,0] = mask2[:,:,1] = mask2[:,:,2] = mask

            dst_img[mask2] = img[mask2]
        else:
            dst_img[mask] = img[mask]

    return dst_img

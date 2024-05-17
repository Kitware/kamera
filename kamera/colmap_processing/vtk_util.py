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
import copy, os, cv2, vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection
# from mpl_toolkits.mplot3d import Axes3D

from colmap_processing.camera_models import StandardCamera, \
    quaternion_from_matrix
from colmap_processing.platform_pose import PlatformPoseFixed


class Camera(object):
    def __init__(self, res_x, res_y, vfov, pos, rmat):
        """Create camera.

        When pan and tilt are zero, the camera points north.

        :param res_x: Horizontal resolution of the image.
        :type res_x: int

        :param res_y: Vertical resolution of the image.
        :type res_y: int

        :param vfov: Vertical field of view (degrees).
        :type vfov: float

        :param pos: Position of the camera within the world.
        :type pos: 3-array of float

        :param pan: Pan of the camera (degrees). When pan and tilt are zero,
            the camera points along world y axis (i.e., north).
        :type pan: float

        :param tilt: Tilt of the camera (degrees). When pan and tilt are zero,
            the camera points along world y axis (i.e., north).
        :type tilt: float

        """
        self._res_x = res_x
        self._res_y = res_y
        self._vfov = vfov
        self._pos = pos
        self._rmat = rmat
        self._model_reader = None

        self._update_vtk_camera()

    @property
    def focal_length(self):
        """Return the unitless focal length.

        """
        return 1/(2*np.tan(self.vfov/180*np.pi/2))*self.res_y

    @property
    def res_x(self):
        return self._res_x

    @res_x.setter
    def res_x(self, val):
        self._res_x = val
        self._update_vtk_camera()

    @property
    def res_y(self):
        return self._res_y

    @res_y.setter
    def res_y(self, val):
        self._res_y = val

    @property
    def vfov(self):
        return self._vfov

    @vfov.setter
    def vfov(self, val):
        self._vfov = val

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, val):
        self._pos = val

    @property
    def rmat(self):
        """Orientation rotation matrix.

        """
        return self._rmat

    @property
    def kmat(self):
        """Camera calibration matrix.

        """
        return np.array([[self.focal_length, 0, self.res_x/2],
                         [0, self.focal_length, self.res_y/2], [0, 0, 1]])

    def ifov_image(self, downsample=1):
        """Return angle subtended by each pixel.

        The angle if the RMS value over all directions of 1-pixel steps on the
        focal plane.

        """
        inv_kmat = np.linalg.inv(self.kmat)

        res_x = self.res_x//downsample
        res_y = self.res_y//downsample
        X,Y = np.meshgrid(np.arange(res_x), np.arange(res_y))
        xy1 = np.ones((3,res_x*res_y))
        xy1[0] = X.ravel()
        xy1[1] = Y.ravel()

        xy2 = copy.copy(xy1)
        xy2[0] += 1

        xy3 = copy.copy(xy1)
        xy3[1] += 1

        # Convert from homogenous pixel coordinates to 3-D ray coordinates.
        xy1 = np.dot(inv_kmat, xy1)
        xy2 = np.dot(inv_kmat, xy2)
        xy3 = np.dot(inv_kmat, xy3)

        #Normalize
        xy1 /= np.sqrt(np.sum(xy1**2, 0))
        xy2 /= np.sqrt(np.sum(xy2**2, 0))
        xy3 /= np.sqrt(np.sum(xy3**2, 0))

        ifov_x = np.arccos(np.maximum(np.minimum(np.sum(xy1*xy2, 0), 1), -1))
        ifov_y = np.arccos(np.maximum(np.minimum(np.sum(xy1*xy3, 0), 1), -1))
        ifov = np.sqrt(ifov_x**2 + ifov_y**2)
        ifov.shape = (self.res_y,self.res_x)

        return ifov

    def _update_vtk_camera(self):
        camera = vtk.vtkCamera()

        # Set vertical field of view in degrees.
        camera.SetViewAngle(self.vfov)

        # Define a level camera looking along the world y-axis (i.e., north).
        R = self.rmat
        focal_point = copy.deepcopy(self.pos)
        focal_point += R[2]
        camera.SetPosition(self.pos)
        camera.SetFocalPoint(focal_point)
        camera.SetViewUp(-R[1])

        #camera.ParallelProjectionOn()
        #camera.SetParallelScale(10)
        camera.SetClippingRange(10,200)

        #camera.Pitch(-5)
        #camera.OrthogonalizeViewUp()

        self.vtk_camera = camera

    def render_image(self, model_reader, clipping_range=[2,100], diffuse=0.6,
                     ambient=0.6, specular=0.1, light_color=[1.0, 1.0, 1.0],
                     light_pos=[0,0,1000]):
        """Render a view of the loaded model.

        :param model_reader: World mesh model reader.
        :type model_reader: vtkAlgorithmOutput

        A window will pop up with the rendered image but then be subsequently
        deleted. This is needed until I figure out how to do offscreen
        rendering.

        :param diffuse: Diffuse light coefficient (range 0-1).
        :type diffuse:

        :param ambient Ambient lighting coefficient (range 0-1).
        :param ambient:

        :param specular: Lighting specularity coefficient (range 0-1).
        :type specular: float

        """
        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(model_reader.GetOutputPort())

        # Actor
        actor =vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set lighting properties
        actor.GetProperty().SetDiffuse(diffuse)
        actor.GetProperty().SetAmbient(ambient)
        actor.GetProperty().SetSpecular(specular)

        # Renderer
        renderer = vtk.vtkRenderer()
        self._update_vtk_camera()
        self.vtk_camera.SetClippingRange(clipping_range[0],clipping_range[1])
        renderer.SetActiveCamera(self.vtk_camera)
        renderer.AddActor(actor)

        light = vtk.vtkLight()
        light.SetColor(*light_color)
        light.SetPosition(light_pos)
        renderer.AddLight(light)

        # RenderWindow
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(self.res_x, self.res_y)

        # Render image
        rgbfilter = vtk.vtkWindowToImageFilter()
        rgbfilter.SetInput(renderWindow)
        rgbfilter.SetInputBufferTypeToRGB()
        rgbfilter.Update()
        dims = rgbfilter.GetOutput().GetDimensions()
        npdims = [dims[1],dims[0],3]
        image = numpy_support.vtk_to_numpy(rgbfilter.GetOutput().GetPointData().GetScalars()).reshape(npdims)
        image = np.flipud(image)

        del renderer, renderWindow, rgbfilter

        return image

    def project(self, wrld_pt):
        if wrld_pt.ndim == 2:
            if wrld_pt.shape[0] == 3:
                wrld_pt = np.vstack([wrld_pt,np.ones(wrld_pt.shape[1])])
        else:
            raise Exception('Not implemented')

        im_pt = np.dot(self.get_camera_matrix(), wrld_pt)
        return im_pt[:2]/im_pt[2]

    def unproject(self, im_pt):
        pass

    def get_camera_matrix(self):
        """Return camera projection matrix that maps world to image.

        """
        T = -np.dot(self.rmat, self.pos)
        P = np.dot(self.kmat, np.hstack([self.rmat,np.atleast_2d(T).T]))

        return P

    def unproject_view(self, model_reader, clipping_range=[2,100],
                       return_image=False):
        """Return the coordinates of intersection of each pixel with the world.

        :param model_reader: World mesh model output.
        :type model_reader: vtkAlgorithmOutput

        """
        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(model_reader.GetOutputPort())

        # Actor
        actor =vtk.vtkActor()
        actor.SetMapper(mapper)

        # Renderer
        renderer = vtk.vtkRenderer()
        self._update_vtk_camera()
        self.vtk_camera.SetClippingRange(clipping_range[0],clipping_range[1])
        renderer.SetActiveCamera(self.vtk_camera)
        renderer.AddActor(actor)

        # RenderWindow
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(self.res_x, self.res_y)

        # Read Z Buffer
        zfilter = vtk.vtkWindowToImageFilter()
        zfilter.SetInput(renderWindow)
        zfilter.SetInputBufferTypeToZBuffer()
        zfilter.Update()

        # transform zbuffer to numpy array
        dims = zfilter.GetOutput().GetDimensions()
        npdims = [dims[1],dims[0]]
        array = numpy_support.vtk_to_numpy(zfilter.GetOutput().GetPointData().GetScalars()).reshape(npdims)
        array = np.flipud(array)

        if return_image:
            # Render image
            rgbfilter = vtk.vtkWindowToImageFilter()
            rgbfilter.SetInput(renderWindow)
            rgbfilter.SetInputBufferTypeToRGB()
            rgbfilter.Update()

        del renderer, renderWindow, zfilter

        # Convert ZBuffer to range.
        near, far = self.vtk_camera.GetClippingRange()
        b = near*far/(near - far)
        a = -b/near
        depth = b/(array - a)

        X, Y = np.meshgrid(np.arange(depth.shape[1]),
                           np.arange(depth.shape[0]))
        xy = np.vstack([X.ravel(), Y.ravel(),
                        np.ones(depth.shape[0]*depth.shape[1])])
        inv = np.linalg.inv(np.dot(self.kmat, self.rmat))
        ray_dir = np.dot(inv, xy)

        # Z-buffer is the distance the ray travels projected onto the optical
        # axis. The rays already have unit length when projected along the
        # optical axis.
        xyz = ray_dir*depth.ravel() + np.atleast_2d(self.pos).T

        X = np.reshape(xyz[0], (self.res_y, self.res_x))
        Y = np.reshape(xyz[1], (self.res_y, self.res_x))
        Z = np.reshape(xyz[2], (self.res_y, self.res_x))

        if return_image:
            # Render image
            dims = rgbfilter.GetOutput().GetDimensions()
            npdims = [dims[1], dims[0], 3]
            image = numpy_support.vtk_to_numpy(rgbfilter.GetOutput().GetPointData().GetScalars()).reshape(npdims)
            del rgbfilter
            image = np.flipud(image)
            return X, Y, Z, depth, image
        else:
            return X, Y, Z, depth

    def to_standard_camera(self):
        quat = quaternion_from_matrix(self.rmat.T)
        platform_pose_provider = PlatformPoseFixed(self.pos, quat)
        cm = StandardCamera(self.res_x, self._res_y, self.kmat, np.zeros(5),
                            [0, 0, 0], [0, 0, 0, 1], platform_pose_provider)
        return cm


class CameraPanTilt(Camera):
    def __init__(self, res_x, res_y, vfov, pos, pan, tilt):
        """Create camera.

        When pan and tilt are zero, the camera points north.

        :param res_x: Horizontal resolution of the image.
        :type res_x: int

        :param res_y: Vertical resolution of the image.
        :type res_y: int

        :param vfov: Vertical field of view (degrees).
        :type vfov: float

        :param pos: Position of the camera within the world.
        :type pos: 3-array of float

        :param pan: Pan of the camera (degrees). When pan and tilt are zero,
            the camera points along world y axis (i.e., north).
        :type pan: float

        :param tilt: Tilt of the camera (degrees). When pan and tilt are zero,
            the camera points along world y axis (i.e., north).
        :type tilt: float

        """
        self._pan = pan
        self._tilt = tilt
        self.set_rmat_from_pan_tilt()
        super(CameraPanTilt, self).__init__(res_x, res_y, vfov, pos,
                                            self._rmat)

    @property
    def pan(self):
        """Pan in degrees.

        """
        return self._pan

    @pan.setter
    def pan(self, val):
        self._pan = val
        self.set_rmat_from_pan_tilt()

    @property
    def tilt(self):
        """Tilt in degrees.

        """
        return self._tilt

    @tilt.setter
    def tilt(self, val):
        self._tilt = val
        self.set_rmat_from_pan_tilt()

    def set_rmat_from_pan_tilt(self):
        """Orientation rotation matrix.

        """
        tilt = self.tilt
        pan = self.pan

        theta = tilt/180*np.pi
        c = np.cos(theta)
        s = np.sin(theta)
        rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])

        theta = pan/180*np.pi
        c = np.cos(-theta)
        s = np.sin(-theta)
        rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])

        r1 = np.dot(rx, np.array([[1,0,0],[0,0,-1],[0,1,0]]).T).T

        r = np.dot(rz, r1.T).T

        self._rmat = r


def render_distored_image(width, height, K, dist, cam_pos, R, model_reader,
                          return_depth=True, monitor_resolution=(1000, 1000),
                          clipping_range=[1, 2000], fill_holes=True):
    """Render a view from a camera with distortion.

    """
    render_resolution = list(monitor_resolution)

    if monitor_resolution[0] != monitor_resolution[1]:
        raise Exception('There is a bug when the monitor resolution isn\'t '
                        'square. Your actual monitor doesn\'t need to be '
                        'square, just pick the largest square resolution that '
                        'fits inside your monitor.')

    # Generate points along the border of the distorted camera.
    num_points = 1000
    perimeter = 2*(height + width)
    ds = num_points/float(perimeter)
    xn = np.max([2, int(ds*width)])
    yn = np.max([2, int(ds*height)])
    x = np.linspace(0, width, xn)
    y = np.linspace(0, height, yn)[1:-1]
    pts = np.vstack([np.hstack([x, np.full(len(y), width, dtype=np.float64),
                                x[::-1], np.zeros(len(y))]),
                     np.hstack([np.zeros(xn), y,
                                np.full(xn, height, dtype=np.float64),
                                y[::-1]])]).T

    # Unproject these rays.
    ray_dir = np.ones((len(pts), 3), dtype=np.float)
    ray_dir0 = cv2.undistortPoints(np.expand_dims(pts, 0), K, dist, R=None)
    ray_dir[:, :2] = np.squeeze(ray_dir0)

    K_ = np.identity(3)
    points2 = cv2.projectPoints(ray_dir, np.zeros(3, dtype=np.float32),
                                np.zeros(3, dtype=np.float32), K_, None)[0]
    points2 = np.squeeze(points2, 1).T

    # points2 are now in the distortion-free camera.

    if False:
        plt.plot(points2[0], points2[1])
        plt.plot([0, render_resolution[0], render_resolution[0], 0, 0],
                 [0, 0, render_resolution[1], render_resolution[1], 0])

    r1 = np.abs(points2[0]).max()
    r2 = np.abs(points2[1]).max()
    s1 = render_resolution[0]/2/r1
    s2 = render_resolution[1]/2/r2
    K_[0, 0] = K_[1, 1] = min([s1, s2])*0.98

    if s1 > s2:
        render_resolution[0] = int(np.ceil(render_resolution[1]*r1/r2))
    else:
        render_resolution[1] = int(np.ceil(render_resolution[0]*r2/r1))

    K_[0, 2] = render_resolution[0]/2
    K_[1, 2] = render_resolution[1]/2

    vfov = np.arctan(render_resolution[1]/2/K_[1, 1])*2*180/np.pi
    vtk_camera = Camera(render_resolution[0], render_resolution[1], vfov,
                        cam_pos, R)

    img = vtk_camera.render_image(model_reader, clipping_range=clipping_range,
                                  diffuse=0.6, ambient=0.6, specular=0.1,
                                  light_color=[1.0, 1.0, 1.0],
                                  light_pos=[0, 0, 1000])

    #img = cv2.resize(img, tuple(render_resolution))

    if return_depth or fill_holes:
        ret = vtk_camera.unproject_view(model_reader,
                                        clipping_range=clipping_range)
        E, N, U, depth = ret
        #depth = cv2.resize(depth, tuple(render_resolution))

    #plt.figure(); plt.imshow(img)

    #plt.figure(); plt.imshow(real_image)

    # Warp the rendered view back to the original, possibly distorted, camera
    # view.

    # These are the pixel coordinates for the centers of all the pixels in the
    # image of size img.shape, which we will scale up to the pixel coordinates
    # for those same locations in the image of size (height, width).
    x = (np.arange(img.shape[1]) + 0.5)*width/img.shape[1]
    y = (np.arange(img.shape[0]) + 0.5)*height/img.shape[0]

    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()])
    ray_dir = cv2.undistortPoints(np.expand_dims(points.T, 0), K, dist, None)
    ray_dir = np.squeeze(ray_dir).astype(np.float32).T
    ray_dir = np.vstack([ray_dir, np.ones(ray_dir.shape[1])])
    points2 = cv2.projectPoints(ray_dir.T, np.zeros(3, dtype=np.float32),
                                np.zeros(3, dtype=np.float32), K_, None)[0]
    points2 = np.squeeze(points2, 1).T
    X2 = np.reshape(points2[0], X.shape).astype(np.float32)
    Y2 = np.reshape(points2[1], Y.shape).astype(np.float32)
    X2 = cv2.resize(X2, (width, height), cv2.INTER_LINEAR)
    Y2 = cv2.resize(Y2, (width, height), cv2.INTER_LINEAR)
    img = cv2.remap(img, X2, Y2, cv2.INTER_CUBIC)

    if return_depth or fill_holes:
        X = cv2.remap(E, X2, Y2, cv2.INTER_LINEAR)
        Y = cv2.remap(N, X2, Y2, cv2.INTER_LINEAR)
        Z = cv2.remap(U, X2, Y2, cv2.INTER_LINEAR)
        depth = cv2.remap(depth, X2, Y2, cv2.INTER_LINEAR)

    # ------------------------------------------------------------------------
    # Identify holes in the model and then inpaint them.
    if fill_holes:
        hole_mask = depth > clipping_range[-1] - 0.1

        # Holes that extend to the edge of the image won't be filled via
        # inpainting.


        output = cv2.connectedComponentsWithStats(hole_mask.astype(np.uint8),
                                                  8, cv2.CV_32S)
        labels = output[1]

        # Remove components that touch outer boundary.
        edge_labels = set(labels[:, 0])
        edge_labels = edge_labels.union(set(labels[0, :]))
        edge_labels = edge_labels.union(set(labels[:, -1]))
        edge_labels = edge_labels.union(set(labels[-1, :]))

        for i in edge_labels:
            labels[labels == i] = 0

        mask = (labels > 0).astype(np.uint8)
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

        if return_depth:
            X = cv2.inpaint(X.astype(np.float32), mask, 3, cv2.INPAINT_NS)
            Y = cv2.inpaint(Y.astype(np.float32), mask, 3, cv2.INPAINT_NS)
            Z = cv2.inpaint(Z.astype(np.float32), mask, 3, cv2.INPAINT_NS)
            depth = cv2.inpaint(depth.astype(np.float32), mask, 3,
                                cv2.INPAINT_NS)

    if return_depth:
        return img, depth, X, Y, Z
    else:
        return img
    # ------------------------------------------------------------------------


def load_world_model(fname):
    """
    :param fname: Path to .stl or .ply file.
    :type fname: str

    """
    ext = os.path.splitext(fname)[-1]

    if ext == '.stl':
        model_reader = vtk.vtkSTLReader()
    elif ext == '.ply':
        model_reader = vtk.vtkPLYReader()
    elif ext == '.obj':
        model_reader = vtk.vtkOBJReader()
    else:
        raise Exception('Unhandled model extension: \'%s\'' % ext)

    model_reader.SetFileName(fname)
    model_reader.Update()
    return model_reader


def get_azel_from_pts(pts, cam_pos):
    ray_dir = (pts.T - cam_pos).T
    return get_azel_from_ray_dir(ray_dir, cam_pos)


def get_azel_from_ray_dir(ray_dir, cam_pos):
    ray_dir /= np.atleast_2d(np.sqrt(np.sum(ray_dir**2, 1))).T
    azel = np.zeros((2, ray_dir.shape[1]))
    azel[0] = np.arctan2(ray_dir[0], ray_dir[1])
    azel[1] = np.arctan(ray_dir[2]/np.sqrt(ray_dir[0]**2 + ray_dir[1]**2))
    return azel


def get_ray_dir_from_azel(azel):
    azel = np.atleast_2d(azel)
    azel.shape = (2,-1)
    ray_dir = np.zeros((3, azel.shape[1]))
    ray_dir[0] = np.sin(azel[0])*np.cos(azel[1])
    ray_dir[1] = np.cos(azel[0])*np.cos(azel[1])
    ray_dir[2] = np.sin(azel[1])
    return ray_dir

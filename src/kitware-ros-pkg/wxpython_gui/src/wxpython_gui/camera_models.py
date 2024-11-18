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
from __future__ import division, print_function, absolute_import
import numpy as np
import cv2
import time
import yaml
from scipy.interpolate import RectBivariateSpline
import threading
import copy

# ROS imports
import genpy
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_multiply, quaternion_matrix, \
    quaternion_from_euler, quaternion_inverse, euler_matrix


# Instantiate CvBridge
bridge = CvBridge()

# TODO: there should be a separate lock for each camera.
lock = threading.RLock()


def to_str(v):
    """Convert numerical values (scalar or float) to string for saving to yaml

    """
    if hasattr(v,  "__len__"):
        if len(v) > 1:
            return repr(list(v))
        else:
            v = v[0]

    return repr(v)


def load_from_file(filename, nav_state_provider=None):
    """Load from configuration yaml for any of the Camera subclasses.

    """
    with open(filename, 'r') as f:
        calib = yaml.load(f)

    if calib['model_type'] == 'standard':
        return StandardCamera.load_from_file(filename, nav_state_provider)

    if calib['model_type'] == 'ptz':
        return PTZCamera.load_from_file(filename, nav_state_provider)

    if calib['model_type'] == 'azel':
        return AzelCamera.load_from_file(filename, nav_state_provider)


class Camera(object):
    """Base class for all imaging sensors.

    The imaging sensor is attached to a navigation coordinate system (i.e., the
    frame of the INS), which may move relative to the East/North/Up world
    coordinate system. The pose (position and orientation) of this navigation
    coordinate system within the ENU coordinate system is provided by the
    nav_state_provider attribute, which is an instance of a subclass of
    nav_state.NavStateProvider.

    The Camera object captures all of the imaging properties of the sensor.
    Intrinsic and derived parameters can be queried, and projection operations
    (pixels to world coordinates and vice versa) are provided.

    Most operations require specification of time in order determine values of
    any time-varying parameters (e.g., navigation coordinate system state).

    """
    def __init__(self, width, height, image_topic, frame_id=None,
                 nav_state_provider=None):
        """
        :param width: Width of the image provided by the imaging sensor,
        :type width: int

        :param height: Height of the image provided by the imaging sensor,
        :type height: int

        :param image_topic: The topic that the image is published on.
        :type image_topic: str | None

        :param frame_id: Frame ID. If set to None, the fully resolved topic
            name wil be used.
        :type frame_id: str | None

        :param nav_state_provider: Object that returns the state of the
            navigation coordinate system as a function of time. If None is
            passed, the navigation coordinate system will always have
            its x-axis aligned with world y, its y-axis aligned with world x,
            and its z-axis pointing down (negative world z).
        :type nav_state_provider: subclass of NavStateProvider

        """
        self._width = width
        self._height = height
        self._image_topic = image_topic

        # Initialize
        self._image_subscriber = None
        self._image_publisher = None
        self._image_patch_server = None
        self._images = []
        self._image_times = np.zeros(0)

        if frame_id is None:
            self._frame_id = image_topic
        else:
            self._frame_id = frame_id

        if nav_state_provider is None:
            self._nav_state_provider = NavStateFixed()
        else:
            self._nav_state_provider = nav_state_provider

        self.publishing_images = False

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
    def buffer_size(self):
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        self._buffer_size = int(value)

    @property
    def image_topic(self):
        return self._image_topic

    @image_topic.setter
    def image_topic(self, value):
        self._image_topic = value

    @property
    def image_subscriber(self):
        return self._image_subscriber

    @property
    def frame_id(self):
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value):
        self._frame_id = value

    @property
    def nav_state_provider(self):
        """Instance of a subclass of NavStateProvider

        """
        return self._nav_state_provider

    @property
    def images(self):
        """Current queue of images captured from ROS messages

        """
        return self._images

    @property
    def image_times(self):
        """Times associated self.images

        """
        return self._image_times

    @property
    def image_patch_server(self):
        """RequestImagePatches service.

        """
        return self._image_patch_server

    def __str__(self):
        string = [''.join(['image_width: ',repr(self._width),'\n'])]
        string.append(''.join(['image_height: ',repr(self._height),'\n']))
        string.append(''.join(['image_topic: ',repr(self._image_topic),'\n']))
        string.append(''.join(['frame_id: ',repr(self._frame_id),'\n']))
        string.append(''.join(['nav_state_provider: ',
                               repr(self._nav_state_provider)]))

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
    def load_from_file(cls, filename, nav_state_provider=None):
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

    def unproject(self, points, t=None):
        """Unproject image points into the world at a particular time.

        :param points: Coordinates of a point or points within the image
            coordinate system. The coordinate may be Cartesian or homogenous.
        :type points: array with shape (2), (2,N), (3) or (3,N)

        :param t: Time at which to unproject the point(s) (time in seconds
            since Unix epoch).
        :type t: float

        :return: Ray position and direction corresponding to provided image
            points. The direction points from the center of projection to the
            points. The direction vectors are not necassarily normalized.
        :rtype: [ray_pos, ray_dir] where both of type numpy.ndarry with shape
            (3,n).

        """
        raise NotImplementedError

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

    def start_collecting_images(self, buffer_size=None, burn_in_metadata=False,
                                copy=False):
        """Start collecting images published on self.image_topic

        :param buffer_size: Number of images to keep in image list. When a new
            image would cause the image list to exceed this size, the first
            image added is removed to make room. If buffer_size has been
            previously set, passing value None will use that previous value.

        :param burn_in_metdata: Burn metadata, such as the timestamp into raw
            frames.
        :type burn_in_metdata: bool

        :param copy: Copy images on receipt. This is useful to avoid modifying
            the source data that the shared pointer references.

        :type copy: bool

        """
        if hasattr(self, '_buffer_size'):
            if buffer_size is None:
                buffer_size = self._buffer_size
            elif self._buffer_size != buffer_size:
                raise Exception("""Camera with frame_id %s previously had
                                buffer_size set to %i (e.g., from a call to
                                start_publishing_images) and would be
                                overwritten with the different value %i.""" % \
                                (self._frame_id, self._buffer_size,
                                 buffer_size))
        elif buffer_size is None:
            raise Exception('Must provide valid buffer_size, one has not been '
                            'previously set.')
        else:
            self._buffer_size = buffer_size

        self.burn_in_metadata = burn_in_metadata
        if burn_in_metadata:
            self._copy_images_on_receive = True
        else:
            self._copy_images_on_receive = copy

        if self.image_topic is not None:
            self._image_subscriber = rospy.Subscriber(self.image_topic, Image,
                                                      self.image_callback_ros,
                                                      queue_size=buffer_size)

    def stop_collecting_images(self):
        """Stop collecting images published on self.image_topic

        """
        if self._image_subscriber is not None:
            self._image_subscriber.unregister()

    def start_publishing_images(self, topic=None, compressed=False,
                                buffer_size=None):
        """Activates the publisher waiting for calls from
        publish_image_from_list.

        :param topic: Topic to publish images on. Defaults to self.image_topic.
        :type topic: str

        :param compressed: Publish compressed image.
        :type compressed: bool

        :param buffer_size: Number of images to keep in image list. When a new
            image would cause the image list to exceed this size, the first
            image added is removed to make room. If buffer_size has been
            previously set, passing value None will use that previous value.

        """
        if hasattr(self, '_buffer_size'):
            if buffer_size is None:
                buffer_size = self._buffer_size
            elif self._buffer_size != buffer_size:
                raise Exception("""Camera with frame_id %s previously had
                                buffer_size set to %i (e.g., from a call to
                                start_collecting_images) and would be
                                overwritten with the different value %i.""" % \
                                (self._frame_id, self._buffer_size,
                                 buffer_size))
        elif buffer_size is None:
            raise Exception('Must provide valid buffer_size, one has not been '
                            'previously set.')
        else:
            self._buffer_size = buffer_size

        if topic is None:
            topic = self.image_topic

        self.publishing_images = True
        if self.image_topic is not None:
            if compressed:
                self._image_publisher = rospy.Publisher(topic, CompressedImage,
                                                        queue_size=100)
            else:
                self._image_publisher = rospy.Publisher(topic, Image,
                                                        queue_size=100)

        self._seq_ind = 0

    def stop_publishing_images(self):
        self.publishing_images = False
        if self._image_publisher is not None:
            self._image_publisher.unregister()

    def image_callback_ros(self, image_msg):
        """Method that receives messages published on self.image_topic

        :param image_msg: ROS image message.
        :type image_msg: Image

        """
        try:
            # Convert ROS Image message to OpenCV2.
            if False:
                print('----- image_callback_ros debug -----')
                print('Height:', image_msg.height)
                print('width:', image_msg.width)
                print('encoding:', image_msg.encoding)
                print('is_bigendian:', image_msg.is_bigendian)
                print('step:', image_msg.step)
                print('')

            if image_msg.encoding == 'mono8':
                raw_image = bridge.imgmsg_to_cv2(image_msg, 'mono8')
            elif image_msg.encoding in ['rgb8','bgr8']:
                raw_image = bridge.imgmsg_to_cv2(image_msg, 'rgb8')
            elif image_msg.encoding == '32FC1':
                raw_image = bridge.imgmsg_to_cv2(image_msg, '32FC1')
                raw_image = raw_image.astype(np.float32)
            else:
                raise Exception('Unhandled image encoding: %s' %
                                image_msg.encoding)
        except CvBridgeError as e:
            print(e)
            return None

        stamp = image_msg.header.stamp

        if self._copy_images_on_receive:
            raw_image = copy.copy(raw_image)

        if self.burn_in_metadata:
            x = 20; y = 30
            font = cv2. FONT_HERSHEY_PLAIN
            cv2.putText(raw_image, 'Timestamp: ' + str(stamp), org=(x,y),
                        fontFace=font, fontScale=1, color=(0,0,0),
                        thickness=2, bottomLeftOrigin=False)

        self.add_image_to_list(raw_image, stamp.to_sec())

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

    def remove_image_from_list(self, t):
        """Add image to image buffer.

        """
        with lock:
            ind = np.argmin(np.abs(self.image_times - t))
            self.images.pop(ind)
            self._image_times = np.delete(self.image_times, ind)

    def get_image_from_list(self, t=None):
        """Return image from self.image_list with time closest to t.

        :param t: Nearest time to draw image from (time in seconds since Unix
            epoch). If time is None, the current time will be used.
        :type t: float

        :return: Image and actual time of the image
        :rtype: [numpy.ndarray, float]

        """
        if t is None:
            t = time.time()

        if len(self._image_times) == 0:
            return None

        with lock:
            ind = np.argmin(np.abs(self.image_times - t))
            #ind = len(self.images)-1

            return self.images[ind], self.image_times[ind]

    def clear_image_list(self):
        with lock:
            self._images = []
            self._image_times = np.zeros(0)

    def publish_image_from_list(self, t):
        """Publish image from self.image_list with time closest to t.

        The image is published on topic self.image_topic.

        :param t: Nearest time to draw image from (time in seconds since Unix
            epoch). If time is None, the current time will be used.
        :type t: float

        """
        if not self.publishing_images:
            raise Exception('Must run method start_publishing_images first.')

        if t is None:
            t = time.time()

        img,t = self.get_image_from_list(t)

        if img is None:
            return

        if self._image_publisher.data_class == Image:
            if img.ndim == 3:
                image_message = bridge.cv2_to_imgmsg(img, encoding="rgb8")
            else:
                image_message = bridge.cv2_to_imgmsg(img, encoding="mono8")
        else:
            # Compressed image
            image_message = CompressedImage()
            image_message.format = "jpeg"
            image_message.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()

        image_message.header.frame_id = self._frame_id
        image_message.header.stamp = genpy.Time.from_sec(t)
        image_message.header.seq = self._seq_ind
        self._seq_ind += 1
        self._image_publisher.publish(image_message)

    def add_image_patch_server(self, image_patch_server_topic,
                               frame_time_window=2):
        """Subscribe to image patch server service.

        In cases where this camera's imagery is available on a remote node but
        network bandwidth is insufficient to transfer full-resolution images,
        an image patch server can be run on the remote node providing the
        RequestImagePatches service in order to access windowed or reduced-
        resolution versions of the imagery, as is typically needed for
        synthetic camera view rendering.

        :param image_patch_server_topic: Topic providing the
            RequestImagePatches service.
        :type image_patch_server_topic: str

        :parma frame_time_window: If an image with the desired frame time
            cannot be found, this defines the maximum acceptable difference in
            seconds between the closest available frame time and the desired
            frame time. If no images are found within this specified frame time
            radius, then a response will be returned with `success` set to
            false.
        :type frame_time_window: float

        """
        self._image_patch_server_lock =threading.RLock()
        with self._image_patch_server_lock:
            rospy.loginfo('Subscribing to image patch server \'%s\'' %
                          image_patch_server_topic)
            self._image_server_dt = frame_time_window
            self._image_patch_server = rospy.ServiceProxy(
                                                    image_patch_server_topic,
                                                    RequestImagePatches)

    def get_patches_from_image_server(self, frame_time, homography_list,
                                      patch_heights, patch_widths,
                                      interpolation, antialias):
        """Return warped patches from image patch server.

        This method provides homography-warped image patches from source
        imagery stored on a remote image server. The primary use case is
        providing access to windowed or reduced-resolution versions of a high-
        resolution image over a limited-bandwidth connection when sending the
        entire image would be prohibitively expensive.

        The image patch server may service multiple camera image streams. To
        select this camera's images, this method passes the image topic where
        the full-resolution image are published during the image patch request
        service call.

        :param frame_time: Desired frame time of the image from which to draw
            the patches.
        :type frame_time: float

        :param homography_list: Encodes the set of 3x3 homographies to be used
            to warp the image patches. An image patch will be returned for each
            homography in the list.
        :type homography_list: list of 3x3 arrays

        :param patch_heights: List of patch heights, one for each patch to be
            returned.
        :type patch_heights: list of int

        :param patch_widths: List of patch widths, one for each patch to be
            returned.
        :type patch_widths: list of int

        :param interpolation: Set the interpolation algorithm: 0 - nearest
            neighbor, 1 - linear, 3 - cubic, 4 - Lanczos4.
        :type interpolation: int

        :param antialias: Indicates whether anti-aliasing should be done in
            cases where image downsampling occurs. The additional processing on
            the image server required for the anti-aliasing may substantially
            increase response latency in some cases.
        :type antialias: bool

        :return: A Boolean indication of whether the request was successfully
            serviced. If this is true, the second element is a list of images.
        :rtype: [bool,list of Numpy images]

        """
        assert self.image_patch_server is not None, 'No image patch server ' \
            'specified'

        with self._image_patch_server_lock:
            try:
                homographies = list(np.array(homography_list).ravel())
                resp = self.image_patch_server(genpy.Time.from_sec(frame_time),
                                               self._image_server_dt,
                                               homographies, patch_heights,
                                               patch_widths, interpolation,
                                               antialias)
                ros_image_patches = resp.image_patches
                image_patches = []
                for ros_image_patch in ros_image_patches:
                    if ros_image_patch.encoding == 'mono8':
                        image_patches.append(bridge.imgmsg_to_cv2(ros_image_patch,
                                                                  'mono8'))
                    elif ros_image_patch.encoding in ['rgb8','bgr8']:
                        image_patches.append(bridge.imgmsg_to_cv2(ros_image_patch,
                                                                  'rgb8'))
                    else:
                        raise Exception('Unhandled image encoding: %s' %
                                        ros_image_patch.encoding)

                success = resp.success
            except rospy.service.ServiceException as e:
                rospy.logwarn(e)
                success = False
                image_patches = None

            return [success,image_patches]


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

    :param cam_quat: Quaternion specifying the orientation of the camera
        relative to the navigation coordinate system. The quaternion represents
        a coordinate system rotation that takes the navigation coordinate
        system and rotates it into the camera coordinate system.
    :type cam_quat: numpy.ndarray | None

    :param dist: Input vector of distortion coefficients (k1, k2, p1, p2, k3,
        k4, k5, k6) of 4, 5, or 8 elements.
    :type dist: numpy.ndarray

    """
    def __init__(self, width, height, K, dist, cam_pos, cam_quat, image_topic,
                 frame_id=None, nav_state_provider=None):
        """
        See additional documentation from base class above.



        """
        super(StandardCamera, self).__init__(width, height, image_topic,
                                             frame_id, nav_state_provider)

        self._K = np.array(K, dtype=np.float64)
        self._dist = np.atleast_1d(dist).astype(np.float32)
        self._cam_pos = np.array(cam_pos)
        self._cam_quat = np.array(cam_quat, np.float)
        self._cam_quat /= np.linalg.norm(self._cam_quat)

    def __str__(self):
        string = ['model_type: standard\n']
        string.append(super(self.__class__, self).__str__())
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
    def load_from_file(cls, filename, nav_state_provider=None):
        """See base class Camera documentation.

        """
        with open(filename, 'r') as f:
            calib = yaml.load(f)

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
        image_topic = calib['image_topic']
        frame_id = calib['frame_id']

        return cls(width, height, K, dist, cam_pos, cam_quat, image_topic,
                   frame_id, nav_state_provider)

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

            f.write(''.join(['# Quaternion specifying the orientation of the ',
                             'camera relative to the\n# navigation coordinate',
                             ' system. The quaternion represents a coordinate',
                             ' system\n# rotation that takes the navigation ',
                             'coordinate system and rotates it into the\n# ',
                             'camera coordinate system.\n',
                             'camera_quaternion: ',
                             to_str(self._cam_quat),'\n\n']))

            f.write(''.join(['# Position of the camera\'s center of ',
                             'projection within the navigation\n# coordinate ',
                             'system.\n',
                             'camera_position: ',to_str(self._cam_pos),
                             '\n\n']))

            f.write('# Topic on which this camera\'s image is published\n')
            f.write(''.join(['image_topic: ',self._image_topic,'\n\n']))

            f.write('# The frame_id embedded in the published image header.\n')
            f.write(''.join(['frame_id: ',self._frame_id]))

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

    @property
    def focal_length(self):
        return self._K[0,0]

    @focal_length.setter
    def focal_length(self, value):
        self._K[0,0] = value
        self._K[1,1] = value

    @property
    def fx(self):
        return self._K[0,0]

    @property
    def fy(self):
        return self._K[1,1]

    @fx.setter
    def fx(self, value):
        self._K[0,0] = value

    @fy.setter
    def fy(self, value):
        self._K[1,1] = value

    @property
    def cx(self):
        return self._K[0,2]

    @property
    def cy(self):
        return self._K[1,2]

    @cx.setter
    def cx(self, value):
        self._K[0,2] = value

    @cy.setter
    def cy(self, value):
        self._K[1,2] = value

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

    def update_intrinsics(self, K=None, cam_quat=None, dist=None):
        """
        """
        if K is not None:
            self._K = K.astype(np.float64)
        if cam_quat is not None:
            self._cam_quat = cam_quat
        if dist is not None:
            self._dist = dist

    def get_camera_pose(self, t):
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

        ins_pos, ins_quat = self.nav_state_provider.pose(t)

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
        return np.squeeze(im_pts, 1).T

    def unproject(self, points, t=None):
        """See Camera.unproject documentation.

        """
        points = np.array(points, dtype=np.float64)
        if points.ndim == 1:
            points = np.atleast_2d(points).T
            points.shape = (2,-1)

        if t is None:
            t = time.time()

        ins_pos, ins_quat = self.nav_state_provider.pose(t)
        #print('ins_pos', ins_pos)
        #print('ins_quat', ins_quat)

        # Unproject rays into the camera coordinate system.
        ray_dir = np.ones((3,points.shape[1]), dtype=points.dtype)
        ray_dir0 = cv2.undistortPoints(np.expand_dims(points.T, 0),
                                       self._K, self._dist, R=None)
        ray_dir[:2] = np.squeeze(ray_dir0, 0).T

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

        # Normalize
        ray_dir /= np.sqrt(np.sum(ray_dir**2, 0))

        return ray_pos, ray_dir

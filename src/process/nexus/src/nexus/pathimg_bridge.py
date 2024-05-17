import os
import errno
import sys
import warnings
from typing import Union

import numpy as np
import cv2
import cv_bridge
from cv_bridge import CvBridgeError
import sensor_msgs.msg
from nexus.borg import Borg


try:
    from custom_msgs.msg import PathImage
except ImportError:
    warnings.warn("Could not import custom messages. This is just a shim for type inspection.")
    from kaminterfaces.custom_msgs.msg import PathImage


ImageMsgType = Union[PathImage, sensor_msgs.msg.Image]


class ExtendedBridge(cv_bridge.CvBridge):
    """Able to read PathImage, but cannot write it"""
    msg = sensor_msgs.msg.Image

    def __init__(self, name='?'):
        super(ExtendedBridge, self).__init__()
        self.verbose = True
        self.name = name


class SharedMemBridge(ExtendedBridge):
    msg = PathImage

    def __init__(self, name='?'):
        super(SharedMemBridge, self).__init__(name=name)
        self._idx = 0
        self._cycle_size = 16
        self._dirname = None
        self.bridge = cv_bridge.CvBridge()  # convenience object for original CV bridge

    @property
    def cycle_size(self):
        return self._cycle_size

    @cycle_size.setter
    def cycle_size(self, size):
        self._cycle_size = size

    @property
    def dirname(self):
        return self._dirname

    @dirname.setter
    def dirname(self, path):
        """This has to be set to enable the automatic filename generation"""
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self._dirname = path

    def next_filename(self):
        """Automatically index to the next mmap file in the buffer.
        todo: this logic should be moved to some external lock, this is mostly for testing
        """
        self._idx += 1
        self._idx %= self._cycle_size
        name = "img{:02}.bin".format(self._idx)

        return os.path.join(self._dirname, name)

    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough", mode="r"):
        raise NotImplementedError()

    def cv2_to_imgmsg(self, cvim, encoding="passthrough", filename=None, force_flush=False):
        raise NotImplementedError()

    def imgmsg_to_pathimg(self, img_msg):
        raise NotImplementedError()

    def pathimg_to_imgmsg(self, img_msg):
        raise NotImplementedError()


class PathImgBridge(SharedMemBridge):
    msg = PathImage

    def __init__(self, name='?'):
        super(PathImgBridge, self).__init__(name=name)


    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough", mode="r"):
        # type: (PathImage, str, str) -> np.core.memmap
        """
        Convert a custom_msgs::PathImage message to an OpenCV :cpp:type:`cv::Mat`.
        :param img_msg:   A :cpp:type:`custom_msgs::PathImage` message
        :param desired_encoding:  The encoding of the image data, one of the following strings:
           * ``"passthrough"``
           * one of the standard strings in sensor_msgs/image_encodings.h
        :rtype: :cpp:type:`cv::Mat`
        :raises CvBridgeError: when conversion is not possible.
        If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
        Otherwise desired_encoding must be one of the standard image encodings
        This function returns an OpenCV :cpp:type:`cv::Mat` message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
        If the image only has one channel, the shape has size 2 (width and height)
        """
        dtype, n_channels = self.encoding_to_dtype_with_channels(img_msg.encoding)
        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder(">" if img_msg.is_bigendian else "<")

        if n_channels == 1:
            shape = (img_msg.height, img_msg.width)
        else:
            shape = (img_msg.height, img_msg.width, n_channels)

        filename = img_msg.filename
        # print("PBB {}: imgmsg_to_cv2: {}".format(self.name, filename))

        if not os.path.isfile(filename):
            raise IOError("memmap file not found: {}".format(filename))
        im = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
        if self.verbose:
            print("Accessing {} bytes from mmap: {}".format(im.size, filename))

        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == "little"):
            # this shouldn't happen in our system since it's the same machine
            warnings.warn(
                "Byte order mismatch, system is {} endian, message is_bigendian={}".format(
                    sys.byteorder, img_msg.is_bigendian
                )
            )
            im = im.byteswap().newbyteorder()

        if desired_encoding == "passthrough":
            return im

        from cv_bridge.boost.cv_bridge_boost import cvtColor2

        try:
            res = cvtColor2(im, img_msg.encoding, desired_encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

        return res

    def cv2_to_imgmsg(self, cvim, encoding="passthrough", filename=None, force_flush=False):
        # type: (Union[np.core.memmap, np.ndarray], str, str, bool) -> PathImage
        """
        Convert an OpenCV :cpp:type:`cv::Mat` type to a ROS sensor_msgs::Image message.
        :param cvim:      An OpenCV :cpp:type:`cv::Mat` that may be an mmap
        :param encoding:  The encoding of the image data, one of the following strings:
           * ``"passthrough"``
           * one of the standard strings in sensor_msgs/image_encodings.h
        :rtype:           A sensor_msgs.msg.Image message
        :raises CvBridgeError: when the ``cvim`` has a type that is incompatible with ``encoding``
        If encoding is ``"passthrough"``, then the message has the same encoding as the image's OpenCV type.
        Otherwise desired_encoding must be one of the standard image encodings
        This function returns a sensor_msgs::Image message on success, or raises :exc:`cv_bridge.CvBridgeError` on failure.
        """

        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError("Your input type is not a numpy array")
        img_msg = PathImage()
        img_msg.height = cvim.shape[0]
        img_msg.width = cvim.shape[1]
        img_msg.step = cvim.size // img_msg.height

        if len(cvim.shape) < 3:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
        else:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
        if encoding == "passthrough":
            img_msg.encoding = cv_type
        else:
            img_msg.encoding = encoding

            # Verify that the supplied encoding is compatible with the type of the OpenCV image
            if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:
                raise CvBridgeError(
                    "encoding specified as %s, but image has incompatible type %s" % (encoding, cv_type)
                )
        if cvim.dtype.byteorder == ">":
            img_msg.is_bigendian = True

        if isinstance(cvim, np.core.memmap):
            mmim = cvim
        else:
            if filename is None:
                if self.dirname is None:
                    raise ValueError("Must provide a filename or dirname when converting regular numpy array to mmap")
                filename = self.next_filename()
            mmim = np.memmap(filename, dtype=cvim.dtype, mode="w+", shape=cvim.shape)
            mmim[:] = cvim[:]
            force_flush = True

        # print("PBB {}: cv2_to_imgmsg: {}".format(self.name, filename))

        img_msg.filename = mmim.filename
        if force_flush:
            mmim.flush()
        if self.verbose:
            print("Wrote {} bytes to mmap: {}".format(mmim.size, filename))

        return img_msg

    def imgmsg_to_pathimg(self, img_msg):
        # type: (sensor_msgs.msg.Image) -> (PathImage)
        """Convert a conventional ROS Image Message to a PathImage"""
        img = self.bridge.imgmsg_to_cv2(img_msg)
        msg = self.cv2_to_imgmsg(img)
        return msg

    def pathimg_to_imgmsg(self, img_msg):
        # type: (PathImage) -> sensor_msgs.msg.Image
        """Convert a PathImage to a conventional ROS Image message"""
        img = self.imgmsg_to_cv2(img_msg)
        msg = self.bridge.cv2_to_imgmsg(img)
        return msg


class InMemBridge(SharedMemBridge):
    msg = PathImage

    def __init__(self, name='?'):
        super(InMemBridge, self).__init__(name=name)
        self.data = Borg()
        # print("established in-mem bridge: {}".format(name))

    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough", mode="r"):
        # type: (PathImage, str, str) -> np.core.memmap
        """
        Extract from in-process in-memory store
        """
        filename = img_msg.filename
        # print("IMB {}: imgmsg_to_cv2: {}".format(self.name, filename))
        try:
            return self.data[filename]
        except KeyError:
            print('Key error in imgmsg_to_cv2: {} \nkeys:'.format(filename, self.data.keys()))


    def cv2_to_imgmsg(self, cvim, encoding="passthrough", filename=None, force_flush=False):
        # type: (Union[np.core.memmap, np.ndarray], str, str, bool) -> PathImage
        """
        Write to in-process in-memory store
        """
        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError("Your input type is not a numpy array")
        img_msg = PathImage()
        img_msg.height = cvim.shape[0]
        img_msg.width = cvim.shape[1]
        img_msg.step = cvim.size // img_msg.height

        if len(cvim.shape) < 3:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
        else:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
        if encoding == "passthrough":
            img_msg.encoding = cv_type
        else:
            img_msg.encoding = encoding

        filename = self.next_filename()
        # print("IMB {}: cv2_to_imgmsg: {}".format(self.name, filename))
        self.data[filename] = cvim
        img_msg.filename = filename

        return img_msg

    def imgmsg_to_pathimg(self, img_msg):
        # type: (sensor_msgs.msg.Image) -> PathImage
        """Convert a conventional ROS Image Message to a PathImage"""
        img = self.bridge.imgmsg_to_cv2(img_msg)
        msg = self.cv2_to_imgmsg(img)
        return msg

    def pathimg_to_imgmsg(self, img_msg):
        # type: (PathImage) -> sensor_msgs.msg.Image
        """Convert a PathImage to a conventional ROS Image message"""
        img = self.imgmsg_to_cv2(img_msg)
        msg = self.bridge.cv2_to_imgmsg(img)
        return msg


def coerce_message(img_msg, bridge, to_type=None):
    # type: (ImageMsgType, Union[ExtendedBridge, SharedMemBridge], type) -> ImageMsgType
    if to_type is None:
        to_type = getattr(bridge, "msg", sensor_msgs.msg.Image)

    if type(img_msg) == to_type:  # yes I want exact type
        return img_msg

    err_tmp = "Trying to coerce {} to {}. {}."
    if not isinstance(bridge, ExtendedBridge):
        raise TypeError(err_tmp.format(type(img_msg), to_type, "This requires ExtendedBridge/SharedMemBridge class"))
    if not isinstance(bridge, SharedMemBridge) and (to_type is PathImage or isinstance(img_msg, PathImage)):
        raise TypeError(err_tmp.format(type(img_msg), to_type, "This requires the SharedMemBridge class"))

    if isinstance(img_msg, PathImage):  # coerce to PathImage
        if to_type is PathImage:
            return img_msg
        elif to_type is sensor_msgs.msg.Image:
            return bridge.pathimg_to_imgmsg(img_msg)
    elif isinstance(img_msg, sensor_msgs.msg.Image):
        if to_type is sensor_msgs.msg.Image:
            return img_msg
        elif to_type is PathImage:
            return bridge.imgmsg_to_pathimg(img_msg)

    raise TypeError("Trying to coerce {} to {}. {}".format(type(img_msg), to_type, "This transform is not implemented"))

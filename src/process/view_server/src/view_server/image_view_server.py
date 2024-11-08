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
import os
import socket
import traceback
from contextlib import contextmanager
import numpy as np
import threading
from collections import deque
from six.moves.queue import Queue, deque
from six import BytesIO
from hashlib import md5

import cv2

# ROS imports
import rospy
import rospkg
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
from roskv.util import hash_genpy_msg
from roskv.rendezvous import ConditionalRendezvous

# Kamera Imports
from custom_msgs.msg import SynchronizedImages, SyncedPathImages
from custom_msgs.srv import RequestImageMetadata, RequestCompressedImageView, \
    RequestImageView
from nexus.pathimg_bridge import PathImgBridge, ExtendedBridge, coerce_message, InMemBridge

from view_server.img_nexus import Nexus


MEM_TRANSPORT_DIR = os.environ.get('MEM_TRANSPORT_DIR', False)
MEM_TRANSPORT_NS = os.environ.get('MEM_TRANSPORT_NS', False)
if MEM_TRANSPORT_DIR:
    print('MEM_TRANSPORT_DIR', MEM_TRANSPORT_DIR)
    membridge = PathImgBridge(name='VS')
    membridge.dirname = MEM_TRANSPORT_DIR
    SyncedImageMsg = SyncedPathImages
elif MEM_TRANSPORT_NS:
    print('MEM_TRANSPORT_NS', MEM_TRANSPORT_NS)
    membridge = InMemBridge(name='VS')
    membridge.dirname = MEM_TRANSPORT_NS
    SyncedImageMsg = SyncedPathImages
else:
    membridge = ExtendedBridge(name='VS')
    SyncedImageMsg = SynchronizedImages

bridge = CvBridge()


rospack = rospkg.RosPack()

import time
import threading
from contextlib import contextmanager

class NopContext(object):

    @property
    def nop(self):
        return self.nopContext()

    @contextmanager
    def nopContext(self):
        yield True


class TimeoutLock(object):
    def __init__(self, default_timeout=None):
        self._lock = threading.RLock()
        self._default_timeout = default_timeout

    def acquire(self, blocking=True, timeout=-1):
        return self._lock.acquire(blocking, timeout)

    @contextmanager
    def acquire_timeout(self, timeout=None):
        timeout = self._default_timeout if timeout is None else timeout
        result = self._lock.acquire(timeout=timeout)
        yield result
        if result:
            self._lock.release()

    def release(self):
        self._lock.release()



def get_interpolation(interpolation):
    if interpolation == 4:
        flags = cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP
    elif interpolation == 3:
        flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
    elif interpolation == 2:
        flags = cv2.INTER_AREA | cv2.WARP_INVERSE_MAP
    elif interpolation == 1:
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    else:
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    return flags


class ImageViewServer(object):
    """Provides windowed or reduced-resolution image access over network.

    When a request for imagery is made, it is not returned until a new image is
    received by this node.

    """
    def __init__(self, sync_image_topic, rgb_service_topic=None,
                 rgb_metadata_service_topic=None, ir_service_topic=None,
                 ir_metadata_service_topic=None, uv_service_topic=None,
                 uv_metadata_service_topic=None, rgb_queue=None):
        """
        :param sync_image_topic: Topic providing SynchronizedImages messages.
        :type sync_image_topic: str

        :param _service_topic: The service topic providing access to windowed or
            reduced-resolution imagery.
        :type service_topic: str

        :param _metadata_service_topic: The service topic providing metadata for
            the raw imagery stored on this server.
        :type service_topic: str

        """
        self.image_lock = threading.RLock()
        self.rgb_msg = None
        self.ir_msg = None
        self.uv_msg = None

        self.frame2newimg = {"rgb_msg":dict(), "ir_msg":dict(), "uv_msg":dict()}
        self.frame2hash = {"rgb_msg":dict(), "ir_msg":dict(), "uv_msg":dict()}

        if rgb_queue is None:
            raise ValueError("You must provide a rgb_queue parameter")
        self.rgb_queue = rgb_queue  # type: queue.dequeue

        self.queue = {"rgb_msg": Queue(1), "ir_msg": Queue(1), "uv_msg": Queue(1), }
        self.img_stamp_blocks = {"rgb_msg": ConditionalRendezvous(1), "ir_msg": ConditionalRendezvous(1), "uv_msg": ConditionalRendezvous(1), }
        self.req_hash_blocks = {"rgb_msg": ConditionalRendezvous(1), "ir_msg": ConditionalRendezvous(1), "uv_msg": ConditionalRendezvous(1), }


        if isinstance(SyncedImageMsg(), SyncedPathImages):
            sync_image_topic += '_shm'

        hostname_ns = '/' + socket.gethostname()

        # rospy.loginfo('Subscribing to SynchronizedImages topic \'{}\''.format(sync_image_topic))
        # rospy.Subscriber(sync_image_topic, SyncedImageMsg, self.sync_images_callback, queue_size=1)

        self.enabled = {'rgb': True, 'uv': True, 'ir': True}
        # todo: deal with channel enable config

        def subscribe_to_single_image(modality='rgb'):
            if self.enabled[modality]:
                topic = rospy.get_param('_topic'.format(modality),
                                        hostname_ns + '/{}/image_raw'.format(modality))

                rospy.loginfo('Subscribing to Images topic \'{}\''.format(topic))
                rospy.Subscriber(topic, Image, self.any_queue_callback,
                                 callback_args=modality, queue_size=1)

        def subscribe_to_image_service(service_topic, metadata_service_topic,
                                       key):
            if service_topic is not None:
                rospy.loginfo('Creating RequestImageView service to provide '
                              '\'%s\' image views on topic \'%s\'' %
                              (key,service_topic))
                rospy.Service(service_topic, RequestImageView,
                              lambda req: self.image_patch_service_request(req,
                                                                           key,
                                                                           False))

            if service_topic is not None:
                compressed_service_topic = '%s/compressed' % service_topic
                rospy.loginfo('Creating RequestCompressedImageView service to '
                              'provide \'%s\' image views on topic \'%s\'' %
                              (key,compressed_service_topic))
                rospy.Service(compressed_service_topic,
                              RequestCompressedImageView,
                              lambda req: self.image_patch_service_request(req,
                                                                           key,
                                                                           True))

            if metadata_service_topic is not None:
                rospy.loginfo('Creating RequestImageMetadata service to '
                              'provide \'%s\' image metadata via '
                              'RequestImageMetadata on topic \'%s\'' %
                              (key,metadata_service_topic))
            rospy.Service(metadata_service_topic, RequestImageMetadata,
                          lambda req: self.metadata_service_topic_request(req,
                                                                          key))

        for modality in self.enabled:
            subscribe_to_single_image(modality)
        subscribe_to_image_service(rgb_service_topic, rgb_metadata_service_topic, 'rgb_msg')
        subscribe_to_image_service(ir_service_topic, ir_metadata_service_topic, 'ir_msg')
        subscribe_to_image_service(uv_service_topic, uv_metadata_service_topic, 'uv_msg')

    @property
    def nop_lock(self):
        return self.image_lock
        # return self.nopContext()

    @contextmanager
    def nopContext(self):
        yield True

    def any_queue_callback(self, msg, modality='rgb'):
        modality = modality.lower() + '_msg'
        rospy.loginfo("image callback {}".format(modality))
        for frame in self.frame2newimg[modality]:
            try:
                self.frame2newimg[modality][frame][0] = True
            except:
                pass
        with self.nop_lock:
            setattr(self, modality, msg)

    def sync_images_callback(self, msg):
        rospy.loginfo("sync images callback")
        with self.nop_lock:
            try:
                rgb_str = ('RGB=%ix%i %s' % (msg.image_rgb.width,
                                             msg.image_rgb.height,
                                             msg.image_rgb.encoding))
                modality = "rgb_msg"
                img_rendezvous = self.img_stamp_blocks[modality]
                img_rendezvous.put(msg.header.stamp)
                # queue = self.queue[modality]
                # if queue.empty():
                #     rospy.loginfo("Enqueued: {}".format(modality))
                    # queue.put(True)
                # else:
                #     rospy.loginfo("Full or something: {}".format(modality))
            except Exception as exc:
                rospy.logerr('RGB fail: {}: {}'.format(exc.__class__.__name__, exc))
                rgb_str = 'No RGB'

            try:
                ir_str = ('IR=%ix%i %s' % (msg.image_ir.width,
                                           msg.image_ir.height,
                                           msg.image_ir.encoding))
                modality = "ir_msg"

                img_rendezvous = self.img_stamp_blocks[modality]
                img_rendezvous.put(msg.header.stamp)
                # queue = self.queue[modality]
                # if queue.empty():
                    # rospy.loginfo("Enqueued: {}".format(modality))
                    # queue.put(True)
                # else:
                #     rospy.loginfo("Full or something: {}".format(modality))
            except Exception as exc:
                rospy.logerr('IR  fail {}: {}'.format(exc.__class__.__name__, exc))
                ir_str = 'No IR'

            try:
                uv_str = ('UV=%ix%i %s' % (msg.image_uv.width,
                                           msg.image_uv.height,
                                           msg.image_uv.encoding))
                modality = "uv_msg"

                img_rendezvous = self.img_stamp_blocks[modality]
                img_rendezvous.put(msg.header.stamp)
                # queue = self.queue[modality]
                # if queue.empty():
                #     rospy.loginfo("Enqueued: {}".format(modality))
                #     queue.put(True)
                # else:
                #     rospy.loginfo("Full or something: {}".format(modality))
            except Exception as exc:
                rospy.logerr('UV  fail {}: {}'.format(exc.__class__.__name__, exc))
                uv_str = 'No UV'

            rospy.loginfo('Received SynchronizedImages message with [%s] [%s] '
                          '[%s]' % (rgb_str,ir_str,uv_str))
            self.rgb_msg = msg.image_rgb
            self.ir_msg = msg.image_ir
            self.uv_msg = msg.image_uv

    def image_patch_service_request(self, req, modality, compress):
        """
        see custom_msgs/srv/RequestImagePatches.srv for more details.

        :param modality: Which image stream from which to return an image view.
        :type modality: str {'RGB','IR','UV'}

        """
        tic = time.time()
        req_hash = hash_genpy_msg(req)
        #rospy.loginfo('Requesting {} \'{}\' image view of size {} x {}: {}'.format(
        #    'compressed' if compress else '', modality,
        #    req.output_width, req.output_height, req_hash))

        with self.nop_lock:
            # rospy.logwarn('pre lock')
            if modality == 'rgb_msg':
                try:
                    img_msg = self.rgb_queue[0]
                except IndexError as exc:
                    rospy.logwarn("{}: {}".format(exc.__class__.__name__, exc))
                    img_msg = None
            else:
                img_msg = getattr(self, modality)
                # Remove cache
                # setattr(self, modality, None)
            # rospy.logwarn('post lock')

        if img_msg is None:
            #rospy.logerr('exit early due to lack of encoding')
            return False, Image()

        try:
            stale_hash = req_hash == self.frame2hash[modality][req.frame][0]
        except:
            stale_hash = False
        try:
            newimg = self.frame2newimg[modality][req.frame][0]
        except:
            newimg = True

        if newimg or not stale_hash:
            try:
                image = membridge.imgmsg_to_cv2(img_msg, 'passthrough')
            except Exception as exc:
                rospy.logerr('{}: {}'.format(exc.__class__.__name__, exc))
                return False, Image()
        else:
            return True, Image()
        try:
            self.frame2newimg[modality][req.frame][0] = False
        except:
            self.frame2newimg[modality][req.frame] = deque([False], maxlen=1)
        try:
            self.frame2hash[modality][req.frame][0] = req_hash
        except:
            self.frame2hash[modality][req.frame] = deque([req_hash], maxlen=1)

        # rospy.logwarn(inspect.currentframe().f_lineno)
        flags = get_interpolation(req.interpolation)
        dsize = (req.output_width, req.output_height)

        homography = np.reshape(req.homography, (3,3)).astype(np.float32)
        raw_image = cv2.warpPerspective(image, homography, dsize=dsize,
                                     flags=flags)

        if modality == "ir_msg":
            if req.apply_clahe:
                clahe = cv2.createCLAHE(clipLimit=req.contrast_strength,
                                        tileGridSize=(8,8))
                raw_image = clahe.apply(raw_image)
            # Don't need mono16 for display
            raw_image = np.round(raw_image/256).astype('uint8')
            image2 = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
            if req.show_saturated_pixels:
                maxval = 255
                saturation_mask = np.all(image2 == maxval, -1)
                image2[:, :, 1][saturation_mask] = 0
                image2[:, :, 2][saturation_mask] = 0
        elif modality == "uv_msg":
            image2 = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
            if req.show_saturated_pixels:
                maxval = 255
                saturation_mask = np.all(image2 == maxval, -1)
                image2[:, :, 1][saturation_mask] = 0
                image2[:, :, 2][saturation_mask] = 0
        elif modality == "rgb_msg":
            image2 = raw_image
            if req.show_saturated_pixels:
                maxval = 255
                saturation_mask = np.all(image2 == maxval, -1)
                image2[:, :, 1][saturation_mask] = 0
                image2[:, :, 2][saturation_mask] = 0
        else:
            image2 = raw_image

        # rospy.logwarn('post warp')
        if compress:
            # raise NotImplementedError('disabled for now')
            out_msg = CompressedImage()
            out_msg.format = "jpeg"
            out_msg.data = np.array(cv2.imencode('.jpg', image2)[1]).tostring()
            out_msg.header = img_msg.header
            # rospy.logwarn("Compressed")
        else:
            out_msg = bridge.cv2_to_imgmsg(image2, encoding="rgb8")
            out_msg.header = img_msg.header
            # rospy.logwarn("UnCompressed")

        # Cache the request hash so we can block the next time around
        toc = time.time()
        rospy.loginfo("{:.2f} Releasing {: >3}".format(img_msg.header.stamp.to_sec(), modality[:3]))
        print("Time to process request was %0.3fs" % (toc - tic))
        return True, out_msg

    def metadata_service_topic_request(self, req, modality):
        """
        see custom_msgs/srv/RequestImagePatches.srv for more details.

        :param modality: Which image stream from which to return an image view.
        :type modality: str {'RGB','IR','UV'}

        """
        # We want to return the next image received.
        rospy.logerr_throttle(1.0, "request: {} modality: {}".format(req, modality))
        with self.nop_lock:
            img_msg0 = getattr(self, modality)

        if img_msg0 is None:
            msg = (False,0,0,'')
        else:
            msg = (True,img_msg0.height,img_msg0.width,img_msg0.encoding)
        # print('metadata: {}'.format(msg))

        # invalidate cache lanes
        img_rendezvous = self.img_stamp_blocks[modality]
        if req.release:
            img_rendezvous.release()

        return msg


def set_up_nexus(rgb_queue):
    import socket
    # node = 'img_nexus'
    # rospy.init_node(node)
    print('Parent', rospy.get_namespace())
    node_name = rospy.get_name()
    # for param in rospy.get_param_names():
    #     print(param)
    # /nuvo1/img_nexus/ir_topic
    """
    root@nuvo0:~/kamera_ws# rosparam get /nuvo0/img_nexus
{ir_topic: ir/image_raw, max_wait: 0.9, out_topic: synched, rgb_topic: rgb/image_raw,
  uv_topic: uv/image_raw, verbosity: 9}"""
    hostname_ns =  '/' + socket.gethostname()
    sys_prefix = hostname_ns + '/img_nexus'
    verbosity = rospy.get_param('verbosity', 9)
    rgb_topic = rospy.get_param('rgb_topic', hostname_ns + '/rgb/image_raw')
    ir_topic  = rospy.get_param('ir_topic', hostname_ns + '/ir/image_raw')
    uv_topic  = rospy.get_param('uv_topic', hostname_ns + '/uv/image_raw')
    # out_topic = rospy.get_param('out_topic', sys_prefix + 'synced')
    out_topic = hostname_ns + '/synched'
    max_wait  = rospy.get_param('/max_frame_period', 444) / 1000.0
    send_image_data  = rospy.get_param('~send_image_data')
    compress_imagery  = rospy.get_param('~compress_imagery')

    nexus = Nexus(rgb_topic, ir_topic, uv_topic, out_topic, compress_imagery,
                  send_image_data, max_wait, rgb_queue=rgb_queue, verbosity=verbosity)
    return nexus


def rospy_spin(delay=1.0):
    """
    Blocks until ROS node is shutdown. Yields activity to other threads.
    @raise ROSInitException: if node is not in a properly initialized state
    """

    if not rospy.core.is_initialized():
        raise rospy.exceptions.ROSInitException("client code must call rospy.init_node() first")
    rospy.logdebug("node[%s, %s] entering spin(), pid[%s]", rospy.core.get_caller_id(), rospy.core.get_node_uri(),
             os.getpid())
    try:
        while not rospy.core.is_shutdown():
            rospy.rostime.wallsleep(delay)
            rospy.loginfo('spin')
            # print('.', end='')
    except KeyboardInterrupt:
        rospy.logdebug("keyboard interrupt, shutting down")
        rospy.core.signal_shutdown('keyboard interrupt')


def main():
    # Launch the node.
    node = 'image_view_server'
    rospy.init_node(node, anonymous=False)
    node_name = rospy.get_name()

    # -------------------------- Read Parameters -----------------------------
    sync_image_topic = rospy.get_param('%s/sync_image_topic' % node_name)
    rgb_service_topic = rospy.get_param('%s/rgb_service_topic' % node_name)
    ir_service_topic = rospy.get_param('%s/ir_service_topic' % node_name)
    uv_service_topic = rospy.get_param('%s/uv_service_topic' % node_name)

    rgb_metadata_service_topic = rospy.get_param('%s/rgb_metadata_service_topic' % node_name)
    ir_metadata_service_topic = rospy.get_param('%s/ir_metadata_service_topic' % node_name)
    uv_metadata_service_topic = rospy.get_param('%s/uv_metadata_service_topic' % node_name)
    # ------------------------------------------------------------------------

    # Share debayered RGB images between nexus and image view
    rgb_queue = deque(maxlen=1)
    nexus = set_up_nexus(rgb_queue)

    ImageViewServer(sync_image_topic, rgb_service_topic,
                    rgb_metadata_service_topic, ir_service_topic,
                    ir_metadata_service_topic, uv_service_topic,
                    uv_metadata_service_topic, rgb_queue=rgb_queue)

    rospy_spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print('Interrupt')
        pass

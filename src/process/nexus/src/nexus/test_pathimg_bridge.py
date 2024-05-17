import os
import numpy as np
from backports import tempfile
from sensor_msgs.msg import Image as MsgImage
from custom_msgs.msg import PathImage

from nexus import pathimg_bridge


def test_mmap():
    with tempfile.TemporaryDirectory() as tempdir:
        b = pathimg_bridge.PathImgBridge()
        b.dirname = tempdir
        zimg = np.random.randint(0, 255, (7, 5, 3), dtype='uint8')
        msg1 = b.cv2_to_imgmsg(zimg)
        assert os.path.dirname(msg1.filename) == tempdir
        zimg2 = b.imgmsg_to_cv2(msg1)
        assert np.all(zimg.shape == zimg2.shape)
        assert np.all(zimg == zimg2)

        ros_msg = pathimg_bridge.coerce_message(msg1, b, MsgImage)
        msg2 = pathimg_bridge.coerce_message(ros_msg, b, PathImage)
        ros_msg2 = pathimg_bridge.coerce_message(msg2, b, MsgImage)
        assert ros_msg == ros_msg2


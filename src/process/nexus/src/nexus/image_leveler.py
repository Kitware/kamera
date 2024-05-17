#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sensor_msgs.msg import Image as MsgImage

from nexus.ros_numpy_lite import image_msg_to_numpy, numpy_to_image, name_to_dtypes


def eliminate_inband(img):
    # type: (np.ndarray) -> np.ndarray
    """Eliminate weird top row noise / in-band signalling
     in a6750 IR camera output images"""
    # img[0] = img[1]
    return img[1:]


def apply_color_map(ary, colormap=cv2.COLORMAP_PARULA):
    # type: (np.ndarray, int) -> np.ndarray
    """
    Applies a color map to a grayscale image
    Args:
        ary: grayscale image
        colormap: cv2.COLORMAP enum

    Returns:
        Color-mapped RGB UINT8
    """
    out = cv2.applyColorMap(ary, colormap)
    out[:,:,[0,2]] = out[:,:,[2,0]] # fix cv2 color order from BGR to RGB
    return out


def perc_remap(ary, percs=(1,99), output_min=0, output_max=255, out_type='uint8'):
    # type: (np.ndarray, tuple, int, int, str) -> np.ndarray
    """
    Percentile remap. Rescale values based on percentile as lower/upper bound
    Args:
        ary: image to remap
        output_min: minimum output value
        output_max: maximum output value
        percs: Tuple of (lower_percent, upper_percent), where 50 = 50th percentile
        out_type: dtype of output array

    Returns:
        rescaled array
    """
    ary_cast = ary.astype(float)
    lower, upper = np.percentile(ary_cast, percs)
    ary_cast -= lower
    ary_cast /= (upper-lower)
    ary_cast  = np.clip(ary_cast, 0, 1)
    ary_cast *= (output_max - output_min)
    ary_cast += output_min
    return ary_cast.astype(out_type)


def zee_remap(ary, new_mu=None, new_sigma=None, out_type=float, amin=0,
              amax=255, verbose=False):
    """
    Perform a "z-correction". Perform a z-norm on the input data, and rescale
    it according to the params.
    :param ary: Input data to be transformed
    :param new_mu: Intended mean of distribution after transformation
    :param new_sigma: Intended std_dev of distribution after transformation
    :param out_type: Output data type
    :param amin: Clip the output data, min
    :param amax: Clip the output data, max
    :return:
    """
    ary_cast = ary.astype(float)
    mu = np.mean(ary_cast)
    sigma = np.std(ary_cast)
    new_mu = np.mean((amin, amax)) if new_mu is None else new_mu
    new_sigma = sigma if new_sigma is None else new_sigma
    if verbose:
        print('input mu/sigma: {:.3f} {:.3f}'.format(mu, sigma))
    ary_cast -= mu
    ary_cast /= sigma
    ary_cast *= new_sigma
    ary_cast += new_mu
    ary_cast = np.clip(ary_cast, amin, amax).astype(out_type)
    return ary_cast


data_ranges = {'mono8': (np.uint8, 0, 255), 'mono16': (np.uint16, 0, 65535)}


# @timecall
def img_msg_zee_remap(msg, new_mu=None, new_sigma=None, encoding='mono8'):
    """
    Like zee_remap, but unpacks/repacks images messages.
    Not sure if works on multi-channel.
    :param msg:
    :param new_mu:
    :param new_sigma:
    :param out_type:
    :param amin:
    :param amax:
    :return:
    """
    data = image_msg_to_numpy(msg)
    # rqt_image_view does NOT like mono16, may take a few tries to start
    if encoding not in ['mono8', 'mono16']:
        raise NotImplementedError("Cannot handle encoding [{}]".format(encoding))
    dtype, depth = name_to_dtypes[msg.encoding]
    # just going to output only 8 bit for now. Not enough time for extensible.
    if depth != 1:
        raise NotImplementedError("Can only handle 1 channel")
    out_type, amin, amax = data_ranges[encoding]
    data = zee_remap(data, new_mu=new_mu, new_sigma=new_sigma,
                     out_type=out_type, amin=amin, amax=amax)
    # print(data.dtype, "\n========")
    return numpy_to_image(data, encoding=encoding)


def ir_perc_remap(msg, percs=(1, 99), encoding='mono8'):
    # type: (MsgImage, tuple, str) -> MsgImage
    """

    Args:
        msg: IR image message

    Returns:

    """
    data = eliminate_inband(image_msg_to_numpy(msg))

    # rqt_image_view does NOT like mono16, may take a few tries to start
    if encoding not in ['mono8', 'mono16']:
        raise NotImplementedError(
            "Cannot handle encoding [{}]".format(encoding))
    dtype, depth = name_to_dtypes[msg.encoding]
    # just going to output only 8 bit for now. Not enough time for extensible.
    if depth != 1:
        raise NotImplementedError("Can only handle 1 channel")
    out_type, amin, amax = data_ranges[encoding]
    data = perc_remap(data, percs=percs, output_min=amin, output_max=amax,
                      out_type=out_type, )
    return numpy_to_image(data, encoding=encoding)

def ir_trim_top(msg):
    # type: (MsgImage) -> MsgImage
    """
    The IR frames raw off the device have some data buffer in the top row. Remove it
    :param msg:
    :return:
    """
    data = eliminate_inband(image_msg_to_numpy(msg))

    return numpy_to_image(data, encoding=msg.encoding, header=msg.header)



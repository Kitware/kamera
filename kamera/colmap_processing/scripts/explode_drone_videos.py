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
import natsort


def process(video_path, image_prefix, output_dir, opt_flow_thresh=25,
            downsample_rate=2):
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    filepattern = '%s/%s_%%05d.png' % (output_dir, image_prefix)

    # Call ffmpeg to explode the video file to an image directory.
    subprocess.call(' '.join(['ffmpeg', '-i', video_path, filepattern]),
                    shell=True)

    # Delete redundant frames.
    img_fnames = glob.glob('%s/%s_*.png' % (output_dir, image_prefix))
    img_fnames = natsort.natsorted((img_fnames))
    
    img1 = cv2.imread(img_fnames[0])
    
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    for i in range(1, len(img_fnames)):
        try:
            img2 = cv2.imread(img_fnames[i])

            if img2.ndim == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        except:
            # Some png are corrupt for some reason.
            print('Deleting', img_fnames[i])
            os.remove(img_fnames[i])
            continue

        w = 256
        XY = np.meshgrid(np.arange(w//2, img1.shape[1] - w//2, w),
                         np.arange(w//2, img1.shape[0] - w//2, w))
        pts = np.vstack([XY[0].ravel(), XY[1].ravel()]).T.astype(np.float32)

        # calculate optical flow
        lk_params = dict(winSize  = (w,w),
                         maxLevel = 2,
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        print('Calculating optical flow:', img_fnames[i])
        pts2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, pts, None,
                                                 **lk_params)

        delta = np.sqrt(np.sum((pts - pts2)**2, 1))
        
        # Only consider valid points.
        delta = delta[st.ravel().astype(np.bool)]
        
        new_key_frame = np.mean(st) < 0.5
        
        if ~new_key_frame:
            print('Optical flow is', np.percentile(delta, 90), 'pixels')
            new_key_frame = np.percentile(delta, 90) > opt_flow_thresh        
        
        if new_key_frame:
            img1 = img2
        else:
            print('Deleting', img_fnames[i])
            os.remove(img_fnames[i])

    # Delete redundant frames.
    img_fnames = glob.glob('%s/%s_*.png' % (output_dir, image_prefix))
    img_fnames = natsort.natsorted((img_fnames))
    k = 0
    for img_fname in img_fnames:
        if k > 0:
            os.remove(img_fname)

        k += 1

        if k == downsample_rate:
            k = 0

    os.remove(video_path)


# Path to the video.
video_path = '0006.mp4'

# Path to the list of videos that are publicly released.
image_prefix = 'test'

# Directory to save results
output_dir = 'out'


for i in range(10,22):
    video_path = '%s.mp4'

    try:
        base_fname = os.path.splitext(video_path)[0]
        os.rename('%s.MP4' % base_fname, '%s.mp4' % base_fname)
    except:
        pass

    if not os.path.isfile(video_path):
        continue

    # Directory to save results
    head, tail = os.path.split(video_path)
    output_dir = '%s/images/%s' % (head, os.path.splitext(tail)[0])

    image_prefix = '%s_%s' % (head.split('/')[-1], str(i).zfill(4))

    process(video_path, image_prefix, output_dir, opt_flow_thresh=23,
            downsample_rate=5)

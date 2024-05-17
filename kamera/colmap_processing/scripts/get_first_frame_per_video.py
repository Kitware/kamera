from __future__ import division, print_function
import numpy as np
import os
import cv2
import subprocess
import matplotlib.pyplot as plt
import glob
import natsort

base_dir = '/media/in'
out_dir = '/media/out'


list_of_files = []
for (dirpath, dirnames, filenames) in os.walk(base_dir):
    for filename in filenames:
        if (filename.endswith('.mp4') or filename.endswith('.MP4')
             or filename.endswith('.avi')):
            list_of_files.append(os.sep.join([dirpath, filename]))

for fname in list_of_files:
    cap = cv2.VideoCapture(fname)
    ret, frame = cap.read()
    if frame is not None:
        print(frame.shape)

    base = os.path.splitext(os.path.split(fname)[1])[0]
    fname = '%s/%s.png' % (out_dir, base)
    cv2.imwrite(fname, frame)

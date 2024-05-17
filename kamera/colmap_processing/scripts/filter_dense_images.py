from __future__ import division, print_function
import numpy as np
import os
import cv2
import subprocess
import matplotlib.pyplot as plt
import glob
import natsort

fname = '/mnt/data/colmap/stereo/patch-match.cfg'
with open(fname, 'r') as fp:
    patch_match_lines = fp.readlines()

fname = '/mnt/data/colmap/stereo/fusion.cfg'
with open(fname, 'r') as fp:
    fusion_lines = fp.readlines()

k = 0
fnames = []
while True:
    try:
        fnames.append(patch_match_lines[k])
        k += 1
        l2 = patch_match_lines[k]
        k += 1
    except IndexError:
        break

fnames = natsort.natsorted(fnames)

ind = np.arange(0, len(fnames), 3)
fnames = [fnames[_] for _ in ind]

nl2 = '__auto__, 20\n'

fname = '/mnt/data/colmap/stereo/patch-match2.cfg'
with open(fname, 'w') as fp:
    for fn in fnames:
        fp.write(fn)
        fp.write(nl2)

fname = '/mnt/data/colmap/stereo/fusion2.cfg'
with open(fname, 'w') as fp:
    for fn in fnames:
        fp.write(fn)
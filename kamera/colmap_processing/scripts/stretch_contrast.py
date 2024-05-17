import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

fnames = glob.glob('/mnt/data/*.JPG')

stretch_percentiles = [0.1, 99.9]
monochrome = False
clip_limit = 3.5
median_blur_diam = 0


for fname in fnames:
    print('Processing image \'%s\'' % fname)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and monochrome:
        img = img[:, :, 0]

    img = img.astype(np.float)
    img -= np.percentile(img.ravel(), stretch_percentiles[0])
    img[img < 0] = 0
    img /= np.percentile(img.ravel(), stretch_percentiles[1])/255
    img[img > 255] = 255
    img = np.round(img).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))

    if img.ndim == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        img = clahe.apply(img)

    if median_blur_diam > 0:
        img = cv2.medianBlur(img, median_blur_diam)

    cv2.imwrite(fname, img)

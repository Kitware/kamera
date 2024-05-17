import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

img_fnames = glob.glob('images/*.png')
img_fnames.sort()

def apply_clahe(img, clip_limit):
    img = img.astype(np.float)
    img -= np.percentile(img.ravel(), 0.01)
    img /= np.percentile(img.ravel(), 99.99)
    img = np.clip(img, 0, 1)
    img = np.round(img*255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))

    return clahe.apply(img)


for img_fname in img_fnames:
    print(img_fname)
    img = cv2.imread(img_fname, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = apply_clahe(img, clip_limit=3)
    cv2.imwrite(img_fname, img)


k = 0

img1 = cv2.cvtColor(cv2.imread(img_fnames[0]), cv2.COLOR_BGR2GRAY)

# Frame rate
frame_rate = 60  # Hz

# Measure optical flow.
optical_flow_10 = []
optical_flow_50 = []
optical_flow_90 = []
sharpness = []
while True:
    try:
        img2 = cv2.imread(img_fnames[k], cv2.IMREAD_UNCHANGED)
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except IndexError:
        break

    # Calculate sharpness.
    sobelx = cv2.Sobel(img2, cv2.CV_64F, 1, 0).ravel()
    sobely = cv2.Sobel(img2, cv2.CV_64F, 0, 1).ravel()
    sharpness.append(np.mean(np.sqrt(sobelx**2 + sobely**2)))

    if False:
        # Just calculate sharpness
        print(k)
        k += 1
        continue

    w = 256
    XY = np.meshgrid(np.arange(w//2, img1.shape[1] - w//2, w),
                     np.arange(w//2, img1.shape[0] - w//2, w))
    pts = np.vstack([XY[0].ravel(),XY[1].ravel()]).T.astype(np.float32)

    # calculate optical flow
    lk_params = dict(winSize=(w,w),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    print('Calculating optical flow:', img_fnames[k])
    pts2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, pts, None, **lk_params)

    delta = np.sqrt(np.sum((pts - pts2)**2, 1))

    optical_flow_10.append(np.percentile(delta, 10))
    optical_flow_50.append(np.percentile(delta, 50))
    optical_flow_90.append(np.percentile(delta, 90))
    print('Optical flow is', optical_flow_50[-1], 'pixels')

    if False:
        if k > 0 and optical_flow_50[-1] < 10:
            print('Removing', img_fnames[k])
            os.remove(img_fnames[k])
        else:
            img1 = img2
    else:
        img1 = img2

    k += 1


optical_flow_10 = np.array(optical_flow_10)
optical_flow_50 = np.array(optical_flow_50)
optical_flow_90 = np.array(optical_flow_90)
sharpness = np.array(sharpness)
times = np.arange(0, len(optical_flow_10))/float(frame_rate)
plt.plot(times, optical_flow_10)
plt.plot(times, optical_flow_50)
plt.plot(times, optical_flow_90)
plt.plot(times, sharpness/sharpness.max())

if False:
    ind = np.argsort(optical_flow_50)[::-1][3]
    plt.imshow(cv2.imread(img_fnames[ind], cv2.IMREAD_UNCHANGED), 'gray')

ind = np.nonzero(sharpness < 0.28*sharpness.max())[0]

#ind = optical_flow > 10

# Pick the frame with the local-minimum optical flow over half a second.
window = int(0.1*frame_rate)
L = len(sharpness)
local_max = []
for i in range(L):
    ind_left = i
    ind_right = i + window
    if ind_right > L:
        di = ind_right - L
        ind_right -= di
        ind_left -= di

    local_max.append(max(sharpness[ind_left:ind_right]))

local_max = np.array(local_max)

ind = np.nonzero(local_max > sharpness)[0]

for i in ind:
    try:
        os.remove(img_fnames[i])
        print('Removing', img_fnames[i])
    except OSError:
        pass


# Downsample by a fixed amount taking the sharpest image in each window.
final_size = 1500
sharpness = np.array(sharpness)
inds = np.round(np.linspace(0, len(sharpness), final_size)).astype(np.int)
delete_fnames = []
for i in range(len(inds) - 1):
    indi = np.arange(inds[i], inds[i+1])
    indi = indi[np.argsort(sharpness[indi])[:-1]]
    delete_fnames = delete_fnames + [img_fnames[_] for _ in indi]

for fname in delete_fnames:
    print('Removing', fname)
    try:
        os.remove(fname)
    except OSError:
        pass




# Delete 2/3 frames.
img_fnames = glob.glob('GoPro/*.png')
img_fnames.sort()
k = 0
k2 = 0
while True:
    if k2 == 1 or k2 == 2:
        #print(k)
        print('Removing', img_fnames[k])
        os.remove(img_fnames[k])

    k += 1

    if k2 == 2:
        k2 = 0
        continue

    k2 += 1


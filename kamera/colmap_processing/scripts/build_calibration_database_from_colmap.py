#!/usr/bin/env python
"""
ckwg +31
Copyright 2019 by Kitware, Inc.
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
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import time
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import threading
from numba import jit

# Colmap Processing imports.
from colmap_processing.geo_conversions import llh_to_enu
from colmap_processing.colmap_interface import read_images_binary, Image, \
    read_points3D_binary, read_cameras_binary, qvec2rotmat
from colmap_processing.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array



# ----------------------------------------------------------------------------
# Base path to the colmap directory.
image_dir = 'small_test'

# Path to the images.bin file.
images_bin_fname = 'images.bin'

# Path to the points3D.bin file.
points_3d_bin_fname = 'points3D.bin'

# Path to save images to in order to geo-register.
georegister_data_dir = 'small'

model_save_location = 'models'

# Define ENU coordinate system origin.
lat0 = 42.8646162
lon0 = -73.7710985


# Read in the details of all images.
images = read_images_binary(images_bin_fname)


# Remove image keypoints without associated reconstructed 3-D point.
for image_num in images:
    image = images[image_num]
    ind = [_ for _ in range(len(image.xys)) if image.point3D_ids[_] != -1]
    xys = image.xys[ind]
    point3D_ids = image.point3D_ids[ind]

    images[image_num] = Image(id=image.id, qvec=image.qvec, tvec=image.tvec,
                              camera_id=image.camera_id, name=image.name,
                              xys=xys, point3D_ids=point3D_ids)


if False:
    # Save images with keypoints superimposed. This allows selection of
    # pixels near keypoints to be geolocated.

    try:
        os.makedirs(georegister_data_dir)
    except OSError:
        pass

    for image_num in images:
        image = images[image_num]
        img_fname = '%s/%s' % (image_dir, image.name)
        img = cv2.imread(img_fname)

        for i, xy in enumerate(image.xys):
            if image.point3D_ids[i] == -1:
                continue

            xy = tuple(np.round(xy).astype(np.int))
            cv2.circle(img, xy, 5, color=(0, 0, 255), thickness=1)

        img_fname = '%s/images/%s.jpg' % (georegister_data_dir, image.id)
        img = cv2.imwrite(img_fname, img)


pts_3d = read_points3D_binary(points_3d_bin_fname)


# Load in georegistration points.
xyz_to_enu = []
for fname in glob.glob('%s/*_points.txt' % georegister_data_dir):
    image_id = int(os.path.split(fname)[1].split('_points.txt')[0])
    image = images[image_id]

    pts = np.loadtxt(fname)
    pts = np.atleast_2d(pts)

    for pt in pts:
        ind0 = [_ for _ in range(len(image.xys)) if image.point3D_ids[_] != -1]
        xys = image.xys[ind0]
        d = np.sqrt(np.sum((xys - np.atleast_2d(pt[:2]))**2, 1))
        ind1 = np.argmin(d)

        print('Distance to select point:', d[ind1], 'pixels')
        if d[ind1] > 20:
            continue

        xyz = pts_3d[image.point3D_ids[ind0[ind1]]].xyz

        lat, lon = pt[2:]
        enu = llh_to_enu(lat, lon, 0, lat0, lon0, 0)
        xyz_to_enu.append(np.hstack([xyz, enu]))


xyz_to_enu = np.array(xyz_to_enu)
east_north = xyz_to_enu[:, 3:].T
xyz0 = xyz_to_enu[:, :3].T


def tform_err(x):
    R = cv2.Rodrigues(x[:3])[0]
    xyz = np.dot(R, xyz0)

    if x[3] < 0:
        return 1e10

    S = np.diag(np.ones(3)*x[3])

    xyz = np.dot(S, xyz)

    xyz[0] += x[4]
    xyz[1] += x[5]
    xyz[2] += x[6]

    err = np.sqrt(np.mean((east_north - xyz)**2))
    print(err)
    return err


# Solve for optimal transform
x = np.hstack([(np.random.rand(3)*2 - 1)*np.pi, [1, 0, 0, 0]])
x = minimize(tform_err, x).x

T = np.identity(4)
T[:3, :3] = np.dot(np.diag(np.ones(3)*x[3]), cv2.Rodrigues(x[:3])[0])
T[:3, 3] = x[4:]

pts_enu = {}
c = []
for key in pts_3d:
    xyz = pts_3d[key].xyz
    c.append(pts_3d[key].rgb)

    if True:
        xyz = np.dot(T, np.hstack([xyz, 1]))
        xyz = xyz[:3]/xyz[3]

    pts_enu[key] = xyz


if True:
    # Verify that the model is right-side up.
    pts2 = np.array([_ for _ in pts
                     if np.all(np.abs(_) < 100) and abs(_[2]) < 40])
    pts2 = pts2.T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(pts2[0], pts2[1], pts2[2], '.')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')


# Compute ORB feature for each 3-D point in each image.
orb = cv2.ORB_create(nfeatures=int(1e4), nlevels=10, scaleFactor=1.2)

# feature is a dictionary that takes the image index and returns a list of
# [3d_pt_index, descriptor]
features = {}
for image_num in images:
    print('Processing image', image_num, 'of', len(images))
    image = images[image_num]
    img_fname = '%s/%s' % (image_dir, image.name)
    img = cv2.imread(img_fname)

    if False:
        col, row = np.round(image.xys).astype(np.int).T
        data = np.ones(len(row), dtype=np.uint8)
        mask = csr_matrix((data, (row, col)), shape=img.shape[:2]).todense()
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    else:
        mask = None

    kp, des = orb.detectAndCompute(img, mask)
    kp_xy = np.array([_.pt for _ in kp])

    features[image_num] = []

    max_d = 5   # pixels
    tree = spatial.KDTree(list(zip(kp_xy[:,0], kp_xy[:,1])))
    d, ind = tree.query(image.xys, k=1, distance_upper_bound=max_d)

    features[image_num] = [[pts_enu[image.point3D_ids[i]], des[ind[i]]]
                           for i in range(len(d)) if d[i] < max_d]

    # [image.xys[_] - np.array(kp[ind[_]].pt) for _ in range(len(d)) if d[_] < max_d]


try:
    os.makedirs(model_save_location)
except OSError:
    pass

fname = '%s/features_per_image.p' % model_save_location
with open(fname, 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

keys = images.keys()
keys.sort()
fname = '%s/source_image_list.txt' % model_save_location
with open(fname, 'w') as f:
    for key in keys:
        f.write('%s %s\n' % (key, images[key].name))


# Cluster descriptors to do a Bag-of-Words lookup of the nearest image.
# sklearn doesn't support Hamming distance, so convert to binary vector.
descriptors = []
for image_num in features:
    for el in features[image_num]:
        descriptors.append(np.unpackbits(el[1]))

descriptors = np.array(descriptors, np.float32)
num_clusters = len(descriptors)/100
num_clusters = 100
clf = KMeans(n_clusters=num_clusters, n_init=1, random_state=0,
             precompute_distances=True, verbose=5, n_jobs=10)
kmeans = clf.fit(descriptors).cluster_centers_
bow = kmeans.cluster_centers_
#bow = np.array([np.packbits(_) for _ in np.round(bow).astype(np.bool)])

tree = spatial.KDTree(bow)


def get_hist(desc):
    hist = np.zeros(num_clusters, np.int)
    inds = tree.query(np.array(desc), k=1)[1]
    hist = np.zeros(num_clusters, np.int)
    for ind in inds:
        hist[ind] += 1

    hist = hist.astype(np.float32)
    hist /= np.linalg.norm(hist)
    return hist


bow_hist = []
for image_num in features:
    print('Processing image number:', image_num)
    desc = []
    for el in features[image_num]:
        desc.append(np.unpackbits(el[1]))

    if len(desc) > 0:
        hist = get_hist(desc)
    else:
        hist = np.zeros(num_clusters, np.int)

    bow_hist.append(hist)

bow_hist = np.array(bow_hist)


fname = '%s/bow.p' % model_save_location
with open(fname, 'wb') as handle:
    pickle.dump([bow, bow_hist], handle, protocol=pickle.HIGHEST_PROTOCOL)



# Read in test image.
img_fname = 'ref_image.png'

orb = cv2.ORB_create(nfeatures=int(1e4), nlevels=20, scaleFactor=1.2)
img = cv2.imread(img_fname)
kp, des = orb.detectAndCompute(img, None)
des = np.array([np.unpackbits(_) for _ in des])
hist = get_hist(des)

d = np.dot(bow_hist, hist)
ind = np.argsort(d)[::-1]

keys = images.keys();   keys.sort()
img_fname = '%s/%s' % (image_dir, images[keys[ind[1]]].name)
img_ref = cv2.imread(img_fname)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(img_ref)








img_fname = 'ref_image.png'
orb = cv2.ORB_create(nfeatures=int(1e4), nlevels=30, scaleFactor=1.2)
img = cv2.imread(img_fname)
kp, des = orb.detectAndCompute(img, None)

ind = 341
img_fname = '%s/%s' % (image_dir, images[ind].name)
img_ref = cv2.imread(img_fname)
img_ref = cv2.fastNlMeansDenoisingColored(img_ref, None, 10, 10, 7, 21)
kp_ref, des_ref = orb.detectAndCompute(img_ref, None)

if False:
    pts, des0 = zip(*features[ind])
    pts = np.array(pts)
    des0 = np.array(des0)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.match(des, des_ref)

im_pts = []
im_pts_ref = []

for i, m in enumerate(matches):
    im_pts.append(kp[m.queryIdx].pt)
    im_pts_ref.append(kp_ref[m.trainIdx].pt)

im_pts = np.array(im_pts, dtype=np.float32)
im_pts_ref = np.array(im_pts_ref, dtype=np.float32)

if False:
    F, mask = cv2.findFundamentalMat(im_pts, im_pts_ref, cv2.FM_RANSAC)
else:
    H, mask = cv2.findHomography(im_pts, im_pts_ref, method=cv2.RANSAC,
                                 ransacReprojThreshold=1)

mask = mask.ravel() == 1
good_matches = [matches[_] for _ in range(len(matches)) if mask[_]]

# Draw first 10 matches.
plt.imshow(cv2.drawMatches(img, kp, img_ref, kp_ref, good_matches, None))

# We select only inlier points
im_pts = im_pts[mask.ravel() == 1]
im_pts_ref = im_pts_ref[mask.ravel() == 1]


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]

    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1 = img1.copy()

    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2 = img2.copy()

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines = cv2.computeCorrespondEpilines(im_pts.reshape(-1,1,2), 2, F)
lines = lines.reshape(-1,3)
img5,img6 = drawlines(img, img_ref, lines, im_pts, im_pts_ref)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()








    inds = [np.argmin(np.sum((image.xys - np.array(_.pt))**2, 1)) for _ in kp]
    ds = [np.sqrt(np.sum((image.xys[inds[i]] - np.array(kp[i].pt))**2))
          for i in range(len(inds))]


    image.xys

    img_fname = '%s/%s' % (image_dir, image.name)
    img = cv2.imread(img_fname)

    for i, xy in enumerate(image.xys):
        if image.point3D_ids[i] == -1:
            continue

        xy = tuple(np.round(xy).astype(np.int))
        cv2.circle(img, xy, 5, color=(0, 0, 255), thickness=1)

    features




if False:
    # Draw manual key points.
    save_dir = ''
    for key in manual_matches:
        image_id1,image_id2 = key
        img1 = cv2.imread(image_fnames[image_id1 - 1])[:,:,::-1].copy()
        img2 = cv2.imread(image_fnames[image_id2 - 1])[:,:,::-1].copy()

        # These are the manually selected coordinates.
        kps = np.round(manual_matches[key]).astype(np.int)
        kp1s = kps[:,:2]
        kp2s = kps[:,2:]

        for i in range(len(kp1s)):
            cv2.circle(img1, (kp1s[i][0],kp1s[i][1]), 5, (255,0,255), 2)
            cv2.circle(img2, (kp2s[i][0],kp2s[i][1]), 5, (255,0,255), 2)

        fname1 = os.path.split(image_fnames[image_id1 - 1])[-1]
        fname2 = os.path.split(image_fnames[image_id2 - 1])[-1]
        cv2.imwrite(save_dir + fname1, img1[:,:,::-1])
        cv2.imwrite(save_dir + fname2, img2[:,:,::-1])


db = COLMAPDatabase.connect(database_path)
cursor = db.cursor()


if False:
    keep_pairs = set()
    # Remove matches with insufficient inliers.
    min_num_matches = 20
    cursor.execute("SELECT pair_id, data FROM two_view_geometries")
    for row in cursor:
        pair_id = row[0]
        if row[1] is not None:
            inlier_matches = np.fromstring(row[1],
                                           dtype=np.uint32).reshape(-1, 2)
            if len(inlier_matches) > min_num_matches:
                keep_pairs.add(pair_id)

    all_pairs = [pair_id
                 for pair_id, _ in db.execute("SELECT pair_id, data FROM matches")]

    for pair_id in all_pairs:
        if pair_id not in keep_pairs:
            print('Deleting pair:', pair_id)
            db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))


# Add missing keypoints.
for key in manual_matches:
    keypoints = dict((image_id, blob_to_array(data, np.float32, (-1, 2)))
                     for image_id, data in db.execute(
                     "SELECT image_id, data FROM keypoints"))

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
        if data is not None)

    image_id1,image_id2 = key
    keypoints1 = keypoints[image_id1]
    keypoints2 = keypoints[image_id2]

    # These are the manually selected coordinates.
    kp1 = manual_matches[key][:,:2]
    kp2 = manual_matches[key][:,2:]

    for kp in kp1:
        d = np.sqrt(np.sum((kp - keypoints1)**2, 1))
        if d.min() > 2:
            keypoints1 = np.vstack([keypoints1,kp])

    for kp in kp2:
        d = np.sqrt(np.sum((kp - keypoints2)**2, 1))
        if d.min() > 2:
            keypoints2 = np.vstack([keypoints2,kp])

    # Remove old set of keypoints
    db.execute("DELETE FROM keypoints WHERE image_id=?", (image_id1,))
    db.execute("DELETE FROM keypoints WHERE image_id=?", (image_id2,))

    db.add_keypoints(image_id1, keypoints1.copy())
    db.add_keypoints(image_id2, keypoints2.copy())


# Rebuild keypoint dictionary.
keypoints = dict((image_id, blob_to_array(data, np.float32, (-1, 2)))
                 for image_id, data in db.execute(
                 "SELECT image_id, data FROM keypoints"))

# Assing manual matches to keypoints.
for key in manual_matches:
    image_id1,image_id2 = key

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
        if data is not None)

    try:
        matches = matches[(image_id1,image_id2)]
    except:
        matches = np.zeros((0,2), dtype=np.int)

    keypoints1 = keypoints[image_id1]
    keypoints2 = keypoints[image_id2]

    # These are the manually selected coordinates.
    kp1 = manual_matches[key][:,:2]
    kp2 = manual_matches[key][:,2:]

    if False:
        img1 = cv2.imread(image_fnames[image_id1 - 1])[:,:,::-1]
        img2 = cv2.imread(image_fnames[image_id2 - 1])[:,:,::-1]

        plt.figure()
        plt.imshow(img1)
        #plt.plot(keypoints1[:,0], keypoints1[:,1], 'ro')
        plt.plot(kp1[:,0], kp1[:,1], 'go')

        plt.figure()
        plt.imshow(img2)
        #plt.plot(keypoints2[:,0], keypoints2[:,1], 'ro')
        plt.plot(kp2[:,0], kp2[:,1], 'go')

        m = matches[(image_id1,image_id2)][:20]
        plt.figure()
        plt.imshow(img1)
        plt.plot(keypoints1[m[:,0],0], keypoints1[m[:,0],1], 'ro')
        plt.figure()
        plt.imshow(img2)
        plt.plot(keypoints2[m[:,0],0], keypoints2[m[:,0],1], 'ro')

    for i in range(len(kp1)):
        d = np.sqrt(np.sum((kp1[i] - keypoints1)**2, 1))
        ind1 = np.argmin(d)
        assert d[ind1] < 2

        d = np.sqrt(np.sum((kp2[i] - keypoints2)**2, 1))
        ind2 = np.argmin(d)
        assert d[ind2] < 2

        # Loop to artificially increase confidence.
        for _ in range(10):
            matches = np.vstack([matches,[ind1,ind2]])

    pair_id = image_ids_to_pair_id(image_id1, image_id2)
    db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
    db.add_matches(image_id1, image_id2, matches.copy())

    print('Adding manually registered matches to pair:', image_id1, image_id2)


# Commit the data to the file.
db.commit()

# Clean up.

db.close()
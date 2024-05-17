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
image_dir = 'test'

# Path to the images.bin file.
images_bin_fname = 'images.bin'
images_bin_fname = 'images.bin'

camera_bin_fname = 'cameras.bin'
camera_bin_fname = 'cameras.bin'

# Path to the points3D.bin file.
points_3d_bin_fname = 'points3D.bin'
points_3d_bin_fname = 'points3D.bin'

# Path to .db file.
database_fname = 'database.db'

# Reduced point cloud.
reduced_point_cloud = 'small.asc'

# Read in the details of all images.
images = read_images_binary(images_bin_fname)

cameras = read_cameras_binary(camera_bin_fname)

# Get position of camera corresponding to each image.
cam_pos = {}
for image_id in images:
    image = images[image_id]
    R = qvec2rotmat(image.qvec)
    cam_pos[image_id] = np.dot(-R.T, image.tvec)


# Remove image keypoints without associated reconstructed 3-D point.
image_ids = list(images.keys())
for image_num in images:
    image = images[image_num]
    ind = [_ for _ in range(len(image.xys)) if image.point3D_ids[_] != -1]
    xys = image.xys[ind]
    point3D_ids = image.point3D_ids[ind]

    images[image_num] = Image(id=image.id, qvec=image.qvec, tvec=image.tvec,
                              camera_id=image.camera_id, name=image.name,
                              xys=xys, point3D_ids=point3D_ids)

pts_3d = read_points3D_binary(points_3d_bin_fname)

if False:
    pts = np.array([pts_3d[_].xyz for _ in pts_3d])
    rgb = [pts_3d[_] for _ in pts_3d]
    pca = PCA(n_components=2)
    pts_2d = pca.fit_transform(pts).T
    plt.plot(pts_2d[0], pts_2d[1], '.', c=rgb)

if True:
    # Filter down 3d pts.
    pts_3d_keys = list(pts_3d.keys())
    pts = np.array([pts_3d[_].xyz for _ in pts_3d_keys])
    red_pts = np.loadtxt(reduced_point_cloud)[:, :3]

    kdtree = KDTree(pts)

    inds = set()
    for i, red_pt in enumerate(red_pts):
        print('Processing', i + 1, 'of', len(red_pts))
        inds.add(kdtree.query(red_pt, distance_upper_bound=0.1)[1])

    pts_3d = {pts_3d_keys[_]: pts_3d[pts_3d_keys[_]] for _ in inds}
    pts_3d_id = set(pts_3d.keys())

    images0 = images
    images = {}
    num_points_in_image = []
    for key in images0:
        image = images0[key]
        #xys = image.xys
        point3D_ids = image.point3D_ids
        ind = [i for i in range(len(point3D_ids))
               if point3D_ids[i] in pts_3d_id]
        #xys = [xys[_] for _ in ind]
        point3D_ids = [point3D_ids[_] for _ in ind]
        s = len(ind)
        if s > 20:
            images[image.id] = Image(id=image.id, qvec=image.qvec,
                                     tvec=image.tvec,
                                     camera_id=image.camera_id,
                                     name=image.name, xys=xys,
                                     point3D_ids=point3D_ids)
            num_points_in_image.append(s)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
if False:
    L = len(images)
    num_matches = np.zeros((L, L), np.int)
    pt3d_ids = [set(images[_].point3D_ids) for _ in images]
    for i in range(L):
        print('Processing %i/%i' % (i + 1, L))
        for j in range(i + 1, L):
            s = len(pt3d_ids[i].intersection(pt3d_ids[j]))
            num_matches[i, j] = s

    m = num_matches + num_matches.T


# Start by picking the view with the most 3-D points
ind = np.argsort([len(images[key].point3D_ids) for key in image_ids])
image_id0s = [image_ids[_] for _ in ind]


def get_inv_depth_accuracy_array(i, j, inv_depth_accuracy):
    ind1 = min([i, j])
    ind2 = max([i, j])
    return inv_depth_accuracy[:, ind1*num_images + ind2]


for image_id0 in image_id0s:
    s0 = set(images[image_id0].point3D_ids)
    num_matches = []
    for image_id in image_ids:
        if image_id0 == image_id:
            num_matches.append(0)
            continue

        num_matches.append(len(set(images[image_id].point3D_ids).intersection(s0)))

    num_matches = np.array(num_matches)
    inds = np.argsort(num_matches)[::-1]
    inds = inds[num_matches[inds] > 0.95 * num_matches[inds[0]]]

    if len(inds) == 1:
        continue

    raise Exception()
    image_ids_to_consider = [image_ids[_] for _ in inds]

    # We want to consider all other images in 'image_ids_to_consider' to see if
    # we can pick two element from this set and when considered with image_id0,
    # one is sufficiently redundant to the other two.

    das = []
    for i in image_ids_to_consider:
        das.append(get_inv_depth_accuracy_array(image_id0, i, depth_accuracy))

    sum((das[0] < das[1]).toarray())/sum((das[0] > 0).toarray())

    das = np.array(das)



    da1 = depth_accuracy[:, ind1*num_images + ind2].toarray()
    depth_accuracy


    num_matches

    num_matches = [sum([i in pts_3d[pt3d_id].image_ids for pt3d_id in pts_3d])
                   for i in image_ids]
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Consider which points are visible from which cameras.
image_ids = list(images.keys())

all_id = []
for i in images:
    all_id = all_id + list(images[i].point3D_ids)

all_id = list(set(all_id))
all_id.sort()

pt3d_id_map = {all_id[_]: _ for _ in range(len(all_id))}
visibility = lil_matrix((len(all_id), len(images)), dtype=np.bool)
L = len(images)
for i in range(L):
    print('Considering results for image %i/%i' % (i + 1, L))
    image = images[image_ids[i]]
    ids = np.array([pt3d_id_map[_] for _ in image.point3D_ids])
    visibility[ids, i] = 1

visibility = visibility.tocsc()
counts = np.sum(visibility, axis=0).tolist()[0]


removed_image = []
num_points_after_removal = []
while True:
    # visibility[i, j] indicates where the ith 3-D point is visible by images[j].
    c0 = sum(np.array(np.sum(visibility, axis=1).tolist()).ravel() > 3)
    without_i = []
    L = visibility.shape[1]
    for i in range(L):
        print('Considering results without image %i/%i' % (i + 1, L))
        mask = np.ones(L, np.bool)
        mask[i] = 0
        c = sum(np.array(np.sum(visibility[:, mask], axis=1).tolist()).ravel() > 3)
        without_i.append(c)

    ind = np.argmax(without_i)
    num_missed_pts = c0 - without_i[ind]

    print('Removing image %s drops from %i->%i 3-D points visible '
          '(difference=%i) ' % (str(images[image_ids[ind]].id), without_i[ind],
                                c0, num_missed_pts))

    if num_missed_pts > 5000:
        break

    # Remove image ind.
    removed_image.append(images[image_ids[ind]])
    num_points_after_removal.append(without_i[ind])

    del images[image_ids[ind]]
    image_ids = list(images.keys())
    mask = np.ones(L, np.bool)
    mask[ind] = 0
    visibility = visibility[:, mask]
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
def calculate_pair_point_inv_accuracy(images, min_d_acc=0.001, max_d_acc=0.1,
                                      show_timing=False):
    """Return 3-D point's inverse depth accuracy for each image pair.

    :param image_ids:

    :param inv_depth_accuracy: Sparse matrix of size num_3d_points x
        (num_images^2) Matrix. For each ith 3-D point considered (indices
        aren't related to pts_3d keys) and images with 'images' keys j and k,
        where j < k, inv_depth_accuracy[i, j*num_images + k] is the depth
        accuracy constrained by the pairing of images j and k.
    :type inv_depth_accuracy: scipy csc sparse matrix

    """

    num_images = len(images)
    image_ids = list(images.keys())
    image_ids_set = set(image_ids)
    image_id_map = {image_ids[_]: _ for _ in range(len(image_ids))}

    # For each 3-D point, we consider the depth accuracy from each two-view
    # geometry.
    L = len(pts_3d)
    inv_depth_accuracy = lil_matrix((L, num_images*num_images), dtype=np.float32)
    t0 = time.time()
    for ii, pt_3d_id in enumerate(pts_3d):
        pt_3d = pts_3d[pt_3d_id]
        image_ids_ = pt_3d.image_ids
        image_ids_ = image_ids_set.intersection(set(image_ids_))
        image_ids_ = list(image_ids_)
        image_ids_.sort()

        if len(image_ids_) == 0:
            continue

        # Unit vectors pointing from cameras to point.
        v = []

        for image_id in image_ids_:
            cam_pos1 = cam_pos[image_id]
            v.append(pt_3d.xyz - cam_pos1)

        # Normalize.
        v = np.array(v)
        d = np.sqrt(np.sum(v**2, axis=1))
        v = v / np.atleast_2d(d).T

        dp = np.dot(v, v.T)
        dp = np.maximum(dp, -1)
        dp = np.minimum(dp, 1)
        cos_theta = dp   # radians

        # Camera focal length
        f = [cameras[images[image_id].camera_id].params[0]
             for image_id in image_ids_]

        image_ids_ = [image_id_map[_] for _ in image_ids_]

        for i in range(len(image_ids_)):
            ind1 = image_ids_[i]
            for j in range(i + 1, len(image_ids_)):
                ind2 = image_ids_[j]

                # If theta is the angle between cameras in radians, and one
                # camera with focal length f nominally located at d from the
                # point changes it image-space position by one pixel, the ray
                # angle from that camera changes by 1/f, so the depth estimate
                # will change by cos(theta)*d/f
                d_acc1 = cos_theta[i, j]*d[i]/f[i]
                d_acc2 = cos_theta[i, j]*d[j]/f[j]
                d_acc = max([d_acc1, d_acc2])

                d_acc = max([min_d_acc, d_acc])

                if d_acc > max_d_acc:
                    continue

                if False:
                    jj = np.ravel_multi_index((ind1, ind2),
                                              (num_images, num_images))
                else:
                    jj = ind1*num_images + ind2

                inv_depth_accuracy[ii, jj] = 1/d_acc

        if show_timing:
            # Time spend so far.
            time_so_far = time.time() - t0
            time_per_iter = time_so_far/(ii + 1)
            iter_left = L - ii - 1
            print('Calculating pair depth accuracy....time left:',
                  time_per_iter*iter_left/60, 'minutes')

    inv_depth_accuracy = inv_depth_accuracy.tocsc()
    return image_ids, inv_depth_accuracy


@jit(nopython=True)
def get_mask_numba(inot, num_images):
    # Calculate the masks that should be applied to pair_angles to only
    # consider results coming from images that are not inot.
    mask = [1 for _ in range(num_images**2)]

    for i in range(num_images):
        if i == inot:
            for j in range(i + 1, num_images):
                mask[i*num_images + j] = 0
        else:
            for j in range(i + 1, num_images):
                if j == inot:
                    mask[i*num_images + j] = 0

    return mask


def get_mask(inot, num_images):
    # Calculate the masks that should be applied to pair_angles to only
    # consider results coming from images that are not inot.
    mask = np.ones(num_images**2, dtype=np.bool)

    for i in range(num_images):
        if i == inot:
            for j in range(i + 1, num_images):
                mask[i*num_images + j] = 0
        else:
            for j in range(i + 1, num_images):
                if j == inot:
                    mask[i*num_images + j] = 0

    return mask


def get_mask0(inot, num_images):
    # Calculate the masks that should be applied to pair_angles to only
    # consider results coming from images that are not inot.
    mask = np.ones(num_images*num_images, dtype=np.bool)
    for i in range(num_images):
        for j in range(i + 1, num_images):
            if i == inot or j == inot:
                ij = np.ravel_multi_index((i, j), (num_images, num_images))
                mask[ij] = 0
                ij = np.ravel_multi_index((j, i), (num_images, num_images))
                mask[ij] = 0

    return mask


class ThreadedProcessing(object):
    def __init__(self, images, inv_depth_accuracy, num_threads=1,
                 show_timing=False):
        self.inv_depth_accuracy = inv_depth_accuracy
        self.scores_without_image = np.zeros(len(images))
        self.images = images
        self.num_images = len(images)
        self.num_threads = num_threads
        self.finished = np.zeros(self.num_images, dtype=np.bool)
        self.show_timing = show_timing

    def process_all_images(self):
        t0 = time.time()
        thread_pool_dict = {}
        for inot in range(self.num_images):
            if self.num_threads > 1:
                # Multi-threaded.
                while True:
                    for key in list(thread_pool_dict.keys()):
                        if self.finished[key]:
                            del thread_pool_dict[key]

                    if len(thread_pool_dict) < self.num_threads:
                        break

                    time.sleep(.001)

                thread = threading.Thread(target=self.process_one_image,
                                          args=(inot,))
                thread.start()
                thread_pool_dict[inot] = thread
            else:
                # Single-threaded.
                self.process_one_image(inot)

            if self.show_timing:
                # Time spend so far.
                time_so_far = time.time() - t0
                time_per_iter = time_so_far/(inot + 1)
                iter_left = self.num_images - inot - 1
                print('Time left:', time_per_iter*iter_left/60, 'minutes')

        if self.show_timing:
            print('Processing took:', (time.time() - t0)/60, 'minutes')

    def get_mask(self, inot):
        # Calculate the masks that should be applied to pair_angles to only
        # consider results coming from images that are not inot.
        mask = np.ones(self.num_images*self.num_images, dtype=np.bool)
        for i in range(self.num_images):
            for j in range(i + 1, self.num_images):
                if i == inot or j == inot:
                    ij = np.ravel_multi_index((i, j), (self.num_images,
                                                       self.num_images))
                    mask[ij] = 0
                    ij = np.ravel_multi_index((j, i), (self.num_images,
                                                       self.num_images))
                    mask[ij] = 0

        return mask

    def process_one_image(self, inot):
        tic = time.time()
        mask = get_mask(inot, self.num_images)
        #print('Calculating mask took:', time.time() - tic)
        tic = time.time()
        without_i = np.max(inv_depth_accuracy[:, mask], axis=1)
        without_i = without_i.data
        self.scores_without_image[inot] = sum(without_i)
        #print('Analzing pair angles took:', time.time() - tic)
        self.finished[inot] = True


# Analyze scores when one image is left out.
scores_after_removal = []
removed_images = []
while len(images) > 100:
    print('Processing current image list of size=%i' % len(images))
    image_ids, inv_depth_accuracy = calculate_pair_point_inv_accuracy(images,
                                                                      min_d_acc=0.001,
                                                                      max_d_acc=0.1)
    threaded_processing = ThreadedProcessing(images, inv_depth_accuracy,
                                             num_threads=1)
    threaded_processing.process_all_images()
    ind = np.argmax(threaded_processing.scores_without_image)

    # Remove image ind.
    removed_images.append(images[image_ids[ind]])
    scores_after_removal.append(threaded_processing.scores_without_image[ind])
    del images[image_ids[ind]]

    perc = scores_after_removal[-1]/max(scores_after_removal)*100
    print('After reducing to %i iamges, the depth accuracy score is %0.5f%% of '
          'that from the full image set' % (len(images), perc))

plt.plot(np.array(scores_after_removal)/max(scores_after_removal))
# ----------------------------------------------------------------------------
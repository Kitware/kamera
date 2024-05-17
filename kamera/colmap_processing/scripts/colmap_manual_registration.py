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
import sys
import sqlite3
import numpy as np
import glob
import os
from scipy.spatial import distance_matrix
import cv2
import matplotlib.pyplot as plt
from shutil import copyfile


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))


# ----------------------------------------------------------------------------
# Open the database.

image_dir = '*.jpg'
database_path = 'colmap.db'
manual_points_path = 'manual_points/*.txt'
camera_id = 1

image_fnames = glob.glob(image_dir)
image_fnames.sort()


manual_points_fnames = glob.glob(manual_points_path)
manual_matches = {}
for manual_points_fname in manual_points_fnames:
    fname = os.path.splitext(os.path.split(manual_points_fname)[-1])[0]
    fname = fname.split('_')
    manual_matches[(int(fname[0]),int(fname[2]))] = np.loadtxt(manual_points_fname)


if False:
    # Draw manual key points.
    save_dir = '/'
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
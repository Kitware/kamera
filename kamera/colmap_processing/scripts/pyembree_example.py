from __future__ import division, print_function
import numpy as np
import os
import cv2
import subprocess
import matplotlib.pyplot as plt
import glob
import natsort
import trimesh
import math
import PIL
from osgeo import osr, gdal


from pyembree import rtcore_scene as rtcs
from pyembree.mesh_construction import TriangleMesh


# Determine the bounds of the model.
mesh_fname = 'coarse.ply'
mesh = trimesh.load(mesh_fname)
embree_scene = rtcs.EmbreeScene()
vertices = mesh.vertices.astype(np.float32)
faces = mesh.faces
embree_mesh = TriangleMesh(embree_scene, vertices, faces)


height, width, K, R, dist, cam_pos = height1, width1, K1, R1, dist1, cam_pos1
#height, width, K, R, dist, cam_pos = height2, width2, K2, R2, dist2, cam_pos2

X, Y = np.meshgrid(np.arange(width, dtype=np.float32) + 0.5,
                   np.arange(height, dtype=np.float32) + 0.5)
points = np.vstack([X.ravel(), Y.ravel()])
ray_dir = cv2.undistortPoints(np.expand_dims(points.T, 0), K, dist, None)
ray_dir = np.squeeze(ray_dir, 0).T
ray_dir = np.vstack([ray_dir, np.ones(ray_dir.shape[1])])
ray_dir = np.dot(R.T, ray_dir).astype(np.float32)
ray_dir /= np.sqrt(np.sum(ray_dir**2, 0))

origins = np.zeros((ray_dir.shape[1], 3), dtype=np.float32)
origins[:, 0] = cam_pos[0]
origins[:, 1] = cam_pos[1]
origins[:, 2] = cam_pos[2]
res = embree_scene.run(origins, ray_dir.T, output=1)

ray_inter = res['geomID'] >= 0
depth = np.zeros(ray_dir.shape[1], dtype=np.float32)
depth[~ray_inter] = np.nan
print('Intersection coordinates')
primID = res['primID'][ray_inter]
u = res['u'][ray_inter]
v = res['v'][ray_inter]
w = 1 - u - v

inters = np.atleast_2d(w).T * vertices[faces[primID][:, 0]] + \
         np.atleast_2d(u).T * vertices[faces[primID][:, 1]] + \
         np.atleast_2d(v).T * vertices[faces[primID][:, 2]]

# Vector from the intersection point back to the camera.
rays = inters - cam_pos

# Dot product with the z-axis of the camera is the depth.
depth[ray_inter] = np.dot(rays, R[2])
depth = np.reshape(depth, (height, width))



depth_map1 = depth




def unproject_from_camera_embree(im_pts, K, dist, R, cam_pos, embree_scene):
    # Unproject rays into the camera coordinate system.
    ray_dir = np.ones((3, len(im_pts)), dtype=np.float32)
    ray_dir0 = cv2.undistortPoints(np.expand_dims(im_pts, 0), K, dist, R=None)
    ray_dir[:2] = np.squeeze(ray_dir0, 0).T
    ray_dir = np.dot(R.T, ray_dir).astype(np.float32).T

    origins = np.zeros((ray_dir.shape[1], 3), dtype=np.float32)
    origins[:, 0] = cam_pos[0]
    origins[:, 1] = cam_pos[1]
    origins[:, 2] = cam_pos[2]
    res = embree_scene.run(origins, ray_dir, output=1)

    ray_inter = res['geomID'] >= 0
    depth = np.zeros(ray_dir.shape[1], dtype=np.float32)
    depth[~ray_inter] = np.nan
    primID = res['primID'][ray_inter]
    u = res['u'][ray_inter]
    v = res['v'][ray_inter]
    w = 1 - u - v

    points = np.zeros((ray_dir.shape[1], 3), dtype=np.float32)
    points[~ray_inter] = np.nan

    points[ray_inter] = np.atleast_2d(w).T * vertices[faces[primID][:, 0]] + \
                        np.atleast_2d(u).T * vertices[faces[primID][:, 1]] + \
                        np.atleast_2d(v).T * vertices[faces[primID][:, 2]]


    return points.T
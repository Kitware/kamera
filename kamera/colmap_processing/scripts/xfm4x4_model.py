#!/usr/bin/env python
# ckwg +31
# Copyright 2021 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither name of Kitware, Inc. nor the names of any contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
'''
Eugene_Borovikov@Kitware.com: read COLMAP model, apply a 4x4 similarity transform and save the transformed model, e.g. geo-registration.
'''
import argparse, logging, os, numpy as np
from colmap_processing.colmap_interface import read_model, write_model, rotmat2qvec, Image, Point3D


def sRt(M):
    U = M[:3,:3]
    s = np.linalg.norm(U[0,:])
    R = U/s
    t = M[:3,3]
    return s,R,t


def xfm_img(img, s, R, t):
    Ri = img.qvec2rotmat()
    U = Ri.dot(R.T)
    qi = rotmat2qvec(U)
    ti = s*img.tvec - U.dot(t)
    return Image(id=img.id, qvec=qi, tvec=ti,
                 camera_id=img.camera_id, name=img.name,
                 xys=img.xys, point3D_ids=img.point3D_ids)


def xfm_pt(pt, s, R, t):
    xyz = s*R.dot(pt.xyz)+t
    return Point3D(id=pt.id, xyz=xyz, rgb=pt.rgb,
                   error=pt.error, image_ids=pt.image_ids,
                   point2D_idxs=pt.point2D_idxs)


def run(args):
### load transform
    M = np.loadtxt(args.xfm)
    logging.info('xfm={}'.format(M))
### load model
    cameras, images, points3D = read_model(path=args.input_path, ext=args.input_ext)
    logging.info('num_cameras={}'.format(len(cameras)))
    logging.info('num_images={}'.format(len(images)))
    logging.info('num_points3D={}'.format(len(points3D)))
### transform
    s,R,t = sRt(M)
    images = {ID: xfm_img(img, s, R, t) for ID, img in images.items()}
    points3D = {ID: xfm_pt(pt, s, R, t) for ID, pt in points3D.items()}
### output
    if not os.path.isdir(args.output_path): os.makedirs(args.output_path)
    write_model(cameras, images, points3D, path=args.output_path, ext=args.output_ext)
    logging.info('written model{} to {}'.format(args.output_ext, args.output_path))


def CLI(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i:p', '--input_path', metavar='path',
                        help='path/to/colmap/model/folder; default=%(default)s')
    parser.add_argument('-i:x', '--input_ext', metavar='ext', choices=['.bin','.txt'], default='.bin', help='input model format: %(choices)s; default=%(default)s')
    parser.add_argument('-l', '--log', metavar='level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        default='WARNING', help='logging verbosity level: %(choices)s; default=%(default)s')
    parser.add_argument('-o:p', '--output_path', metavar='path',
                        help='path/to/output/folder; default=%(default)s')
    parser.add_argument('-o:x', '--output_ext', metavar='ext', default='.bin', help='output model format; default=%(default)s')
    parser.add_argument('-x', '--xfm', metavar='path',
                        help='path/to/4x4/similarity/transform; default=%(default)s')
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log)
    run(args)


if __name__ == '__main__': CLI()

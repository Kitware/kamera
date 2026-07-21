#!/usr/bin/env python
'''
Load and print camera parameters from YAML, and optionally render the view given a 3D model in PLY format.
'''
import logging, sys, numpy as np
import matplotlib.pyplot as plt

def run(args):
### load camera
    CamMdlFN = args.input_cam
    from colmap_processing.static_camera_model import load_static_camera_from_file
    height, width, K, dist, R, depth_map, latitude, longitude, altitude = load_static_camera_from_file(CamMdlFN)
### print camera parameters
    print(CamMdlFN)
    print('W,H={}'.format([width, height]))
    np.savetxt(sys.stdout, K, '%g', '\t', header='K')
    vFoV = np.arctan2(height/2, K[1,1])*360/np.pi
    print('vFoV={}'.format(vFoV))
    np.savetxt(sys.stdout, R, '%g', '\t', header='R')
    print('LLH={}'.format([latitude, longitude, altitude]))
    mesh_lat0, mesh_lon0, mesh_h0 = args.LLH0
    from colmap_processing.geo_conversions import llh_to_enu
    cam_pos = llh_to_enu(latitude, longitude, altitude, mesh_lat0, mesh_lon0, mesh_h0)
    print('cam_pos={}'.format(cam_pos))
    CCCP = np.eye(4)
    CCCP[:3,:4] = np.array([R[0],-R[1],-R[2], cam_pos]).T
    np.savetxt(sys.stdout, CCCP, '%g', '\t', header='CloudCompare camera pose')
    if not args.visual: return 0
### VTK rendering: limited to monitor resolution (width x height)
    monitor_resolution = (1920, 1080) # TODO: param/config
    import colmap_processing.vtk_util as vtk_util
    model_reader = vtk_util.load_world_model(args.input_mesh)
### render camera view
    render_resolution = list(monitor_resolution)
    vtk_camera = vtk_util.Camera(render_resolution[0], render_resolution[1], vFoV, cam_pos, R)
    img = vtk_camera.render_image(model_reader)
    plt.imshow(img)
    plt.show()
### save camera view to an image file
    import os
    CamViewFN = os.path.splitext(CamMdlFN)[0]+'.vtk.jpg'
    import cv2 as cv
    cv.imwrite(CamViewFN, img[:,:,::-1])

def CLI(argv=None):
    import argparse
    MeshFN = '../data/mesh.ply'
    CamModelFN = '../000128.cam.yml'
    # LLH0NSR = np.array([42.43722062, -84.02781521, 251.412]) # North Star Reach origin in LLH
    LLH0NSR = np.array([42.86453893, -73.77125128, 73]) # KHQ
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i:c', '--input_cam', metavar='path', default=CamModelFN,
                        help='path/to/input/camera/model.yml; default=%(default)s')
    parser.add_argument('-i:m', '--input_mesh', metavar='path', default=MeshFN,
                        help='path/to/input/mesh.ply; default=%(default)s')
    parser.add_argument('-l', '--log', metavar='level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        default='WARNING', help='logging verbosity level: %(choices)s; default=%(default)s')
    parser.add_argument('-o', '--LLH0', metavar='value', default=LLH0NSR,
                        help='assumed local GPS origin as an explicit vector [latitude,longitude,altitude]; default=%(default)s')
    parser.add_argument('-v', '--visual', action='store_true', help='display/save visuals')
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log)
    run(args)

if __name__ == '__main__': CLI()

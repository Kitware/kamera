#!/usr/bin/env python
import os
import os.path as osp
import json
import cv2
import PIL
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import ubelt as ub
from random import shuffle
from scipy.optimize import minimize, fminbound
from matplotlib.backends.backend_pdf import PdfPages

# Custom package imports.
from kamera.sensor_models import (
        quaternion_multiply,
        quaternion_from_matrix,
        quaternion_inverse,
        )
from kamera.sensor_models.nav_conversions import enu_to_llh
from kamera.sensor_models.nav_state import NavStateINSJson, NavStateFixed
from kamera.colmap_processing.camera_models import StandardCamera
from kamera.colmap_processing.colmap_interface import (
        read_images_binary,
        read_points3D_binary,
        read_cameras_binary,
        qvec2rotmat,
        )
from kamera.colmap_processing.image_renderer import render_view


def get_base_name(fname):
    """ Given an arbitrary filename (could be UV, IR, RGB, json),
    extract the portion of the filename that is just the time, flight,
    machine (C, L, R), and effort name.
    """
    # get base
    base = osp.basename(fname)
    # get it without an extension and modality
    modality_agnostic = "_".join(base.split("_")[:-1])
    return modality_agnostic


def get_modality(fname):
    base = osp.basename(fname)
    modality = base.split("_")[-1].split('.')[0]
    return modality

    
def get_channel(fname):
    base = osp.basename(fname)
    channel = base.split("_")[3]
    return channel


def get_basename_to_time(flight_dir) -> dict:
    # Establish correspondence between real-world exposure times base of file
    # names.
    basename_to_time = {}
    for json_fname in pathlib.Path(flight_dir).rglob('*_meta.json'):
        try:
            with open(json_fname) as json_file:
                d = json.load(json_file)
                # Time that the image was taken.
                basename = get_base_name(json_fname)
                basename_to_time[basename] = float(d['evt']['time'])
        except (OSError, IOError):
            pass
    return basename_to_time


def process_images(colmap_images, basename_to_time, nav_state_provider):
    """

    Returns:
    :param img_fnames: Image filename associated with each of the images in
        'colmap_images'.
    :type img_fnames: list of str

    :param img_times: INS-reported time associated with the trigger of each
        image in 'colmap_images'.
    :type img_times:

    :param ins_poses: INS-reported pose, (x, y, z) position and (x, y, z, w)
        quaternion, associated with the trigger of time of each image in
        'colmap_images'.
    :type ins_poses:

    :param sfm_poses: Colmap-reported reported pose, (x, y, z) position and
        (x, y, z, w) quaternion, associated with the trigger time of each image
        in 'colmap_images'.
    :type sfm_poses:

    """
    img_fnames = []
    img_times = []
    ins_poses = []
    sfm_poses = []
    llhs = []
    for image_num in colmap_images:
        image = colmap_images[image_num]
        base_name = get_base_name(image.name)
        try:
            t = basename_to_time[base_name]

            # Query the navigation state recorded by the INS for this time.
            pose = nav_state_provider.pose(t)
            llh = nav_state_provider.llh(t)

            # Query Colmaps pose for the camera.
            R = qvec2rotmat(image.qvec)
            pos = -np.dot(R.T, image.tvec)

            # The qvec used by Colmap is a (w, x, y, z) quaternion
            # representing the rotation of a vector defined in the world
            # coordinate system into the camera coordinate system. However,
            # the 'camera_models' module assumes (x, y, z, w) quaternions
            # representing a coordinate system rotation. Also, the quaternion
            # used by 'camera_models' represents a coordinate system rotation
            # versus the coordinate system transform of Colmap's convention,
            # so we need an inverse.

            #quat = transformations.quaternion_inverse(image.qvec)
            quat = image.qvec / np.linalg.norm(image.qvec)
            quat[0] = -quat[0]

            quat = [quat[1], quat[2], quat[3], quat[0]]

            sfm_pose = [pos, quat]

            img_times.append(t)
            ins_poses.append(pose)
            img_fnames.append(image.name)
            sfm_poses.append(sfm_pose)
            llhs.append(llh)
        except KeyError:
            print('Couldn\'t find a _meta.json file associated with \'%s\'' %
                  base_name)

    ind = np.argsort(img_fnames)
    img_fnames = [img_fnames[i] for i in ind]
    img_times = [img_times[i] for i in ind]
    ins_poses = [ins_poses[i] for i in ind]
    sfm_poses = [sfm_poses[i] for i in ind]
    llhs = [llhs[i] for i in ind]

    return img_fnames, img_times, ins_poses, sfm_poses, llhs


def write_image_locations(locations_fname, img_fnames, ins_poses):
    with open(locations_fname, 'w') as fo:
        for i in range(len(img_fnames)):
            name = img_fnames[i]
            pos = ins_poses[i][0]
            fo.write('%s %0.8f %0.8f %0.8f\n' % (name, pos[0], pos[1], pos[2]))


def get_colmap_data(colmap_images, colmap_cameras,
                    points3d, basename_to_time) -> tuple:
    # Load in all of the Colmap results into more-convenient structures.
    points_per_image = {}
    camera_from_camera_str = {}
    for image_num in colmap_images:
        image = colmap_images[image_num]
        camera_str = osp.basename(osp.dirname(image.name))
        camera_from_camera_str[camera_str] = colmap_cameras[image.camera_id]

        xys = image.xys
        pt_ids = image.point3D_ids
        ind = pt_ids != -1
        pt_ids = pt_ids[ind]
        xys = xys[ind]
        xyzs = np.array([points3d[pt_id].xyz for pt_id in pt_ids])
        base_name = get_base_name(image.name)
        try:
            t = basename_to_time[base_name]
            points_per_image[image.name] = (xys, xyzs, t)
        except KeyError:
            pass
    return points_per_image, camera_from_camera_str

    
def perform_error_analysis(camera_model, points_per_image_, save_dir, camera_str):
    """
    Perform error analysis and save all plots into a single PDF.

    Parameters:
    - camera_model: The calibrated camera model.
    - points_per_image_: List of tuples containing image points, corresponding 3D points, and timestamp.
    - save_dir: Directory where the PDF will be saved.
    - camera_str: String identifier for the camera (used in PDF filename).
    """
    err_meters = []
    err_pixels = []
    err_pixels_per_frame = []
    err_angle = []
    ifov = np.mean(camera_model.ifov())  # Assuming 'ifov' stands for 'instantaneous field of view'

    for xys, xyzs, t in points_per_image_:
        # Project 3D points to 2D image points
        xys2 = camera_model.project(xyzs.T, t)
        err_pixels_ = np.sqrt(np.sum((xys2 - xys.T)**2, axis=0))
        err_pixels_per_frame.append([t, err_pixels_.mean()])
        err_pixels.extend(err_pixels_.tolist())

        # Unproject image points to camera rays
        ray_pos, ray_dir = camera_model.unproject(xys.T, t)

        # Compute direction from camera to 3D points
        ray_dir2 = xyzs.T - ray_pos
        dist = np.linalg.norm(ray_dir2, axis=0)
        ray_dir2 /= dist

        # Calculate angular deviation
        dp = np.clip(np.sum(ray_dir * ray_dir2, axis=0), -1, 1)
        theta = np.arccos(dp)
        err_angle.extend(theta.tolist())

        # Calculate orthogonal distance in meters
        err_meters.extend((np.sin(theta) * dist).tolist())

    # Convert lists to numpy arrays for easier manipulation
    err_meters = np.array(err_meters)
    err_pixels = np.array(err_pixels)
    err_angle = np.array(err_angle)
    err_pixels_per_frame = np.array(err_pixels_per_frame).T

    # Sort the errors
    sorted_err_meters = np.sort(err_meters)
    sorted_err_pixels = np.sort(err_pixels)
    sorted_err_angle = np.sort(err_angle)

    # Initialize PdfPages object
    pdf_filename = f"{save_dir}/error_analysis_{camera_str}.pdf"
    with PdfPages(pdf_filename) as pdf:
        # --- Plot 1: Histogram of Pixel Errors ---
        plt.figure(figsize=(8, 6))
        plt.hist(sorted_err_pixels, bins=50, color='blue', alpha=0.7)
        plt.title('Pixel Errors')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Frequency')
        plt.grid(True)
        pdf.savefig()  # Save the current figure into the PDF
        plt.close()

        # --- Plot 2: Histogram of Meter Errors ---
        plt.figure(figsize=(8, 6))
        plt.hist(sorted_err_meters, bins=50, color='green', alpha=0.7)
        plt.title('Meter Errors')
        plt.xlabel('Error (meters)')
        plt.ylabel('Frequency')
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # --- Plot 3: Histogram of Angular Errors ---
        plt.figure(figsize=(8, 6))
        plt.hist(np.degrees(sorted_err_angle), bins=50, color='red', alpha=0.7)
        plt.title('Angular Errors')
        plt.xlabel('Error (degrees)')
        plt.ylabel('Frequency')
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # --- Plot 4: Pixel Errors per Frame ---
        plt.figure(figsize=(10, 6))
        plt.plot(err_pixels_per_frame[0], err_pixels_per_frame[1], marker='o', linestyle='-', color='purple')
        plt.title('Average Pixel Error per Frame')
        plt.xlabel('Frame Index or Timestamp')
        plt.ylabel('Average Pixel Error (pixels)')
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # --- Optional: Additional Plots ---
        # If you have more plots to include, add them here following the same pattern.

    print(f"Error analysis plots have been saved to {pdf_filename}")


def calibrate_rgb(rgb_camera_strs, img_fnames, ins_poses, sfm_poses,
                  points_per_image, camera_from_camera_str,
                  nav_state_provider, save_dir):
    for camera_str in rgb_camera_strs:
        ins_quat_ = []
        sfm_quat_ = []
        points_per_image_ = []
        for i in range(len(img_fnames)):
            fname = img_fnames[i]
            if osp.basename(osp.dirname(fname)) == camera_str:
                ins_quat_.append(ins_poses[i][1])
                sfm_quat_.append(sfm_poses[i][1])
                points_per_image_.append(points_per_image[fname])

        # Both quaternions are of the form (x, y, z, w) and represent a coordinate
        # system rotation.
        #q_sfm = quaternion_inverse(q_cam)*quaternion_inverse(q_ins)
        cam_quats = [quaternion_inverse(quaternion_multiply(sfm_quat_[k],
                                                            ins_quat_[k]))
                    for k in range(len(ins_quat_))]

        colmap_camera = camera_from_camera_str[camera_str]

        if colmap_camera.model == 'OPENCV':
                fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params
        elif colmap_camera.model == 'PINHOLE':
            fx, fy, cx, cy = colmap_camera.params
            d1 = d2 = d3 = d4 = 0

        K = K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([d1, d2, d3, d4])

        def cam_quat_error(cam_quat) -> float:
            cam_quat = cam_quat/np.linalg.norm(cam_quat)
            camera_model = StandardCamera(colmap_camera.width,
                                        colmap_camera.height,
                                        K, dist, [0, 0, 0], cam_quat,
                                        platform_pose_provider=nav_state_provider)

            err = []
            for xys, xyzs, t in points_per_image_:
                if False:
                    # Reprojection error.
                    xys2 = camera_model.project(xyzs.T, t)
                    err_ = np.sqrt(np.sum((xys2 - xys.T)**2, axis=0))
                else:
                    # Error in meters.

                    # Rays coming out of the camera in the direction of the imaged points.
                    ray_pos, ray_dir = camera_model.unproject(xys.T, t)

                    # Direction coming out of the camera pointing at the actual 3-D points'
                    # locatinos.
                    ray_dir2 = xyzs.T - ray_pos
                    d = np.sqrt(np.sum((ray_dir2)**2, axis=0))
                    ray_dir2 /= d

                    dp = np.minimum(np.sum(ray_dir*ray_dir2, axis=0), 1)
                    dp = np.maximum(dp, -1)
                    theta = np.arccos(dp)
                    err_ = np.sin(theta)*d
                    #err.append(np.percentile(err_, 90))
                    err.append(np.mean(err_))

                err = np.array(err)
                #err = err[err < np.percentile(err, 90)]

                err = np.mean(err)
                #print('RMS reproject error for quat', cam_quat, ': %0.8f' % err)
                return err

        print("Iterating through %s quaternion guesses." % len(cam_quats))
        shuffle(cam_quats)
        best_quat = None
        best_err = np.inf
        for i in range(len(cam_quats)):
            if True:
                cam_quat = cam_quats[i]
            else:
                cam_quat = np.random.rand(4)*2-1

            err = cam_quat_error(cam_quat)
            if err < best_err:
                best_err = err
                best_quat = cam_quat

            if best_err < 10:
                break

        print("Best error: ", best_err)
        print("Best quat: ")
        print(cam_quat)

        print("Minimizing error over camera quaternions")

        ret = minimize(cam_quat_error, best_quat)
        best_quat = ret.x/np.linalg.norm(ret.x)
        ret = minimize(cam_quat_error, best_quat, method='BFGS')
        best_quat = ret.x/np.linalg.norm(ret.x)
        ret = minimize(cam_quat_error, best_quat, method='Powell')
        best_quat = ret.x/np.linalg.norm(ret.x)

        # Sequential 1-D optimizations.
        for i in range(4):
            def set_x(x):
                quat = best_quat.copy()
                quat = quat/np.linalg.norm(quat)
                while abs(quat[i] - x) > 1e-6:
                    quat[i] = x
                    quat = quat/np.linalg.norm(quat)

                return quat

            def func(x):
                return cam_quat_error(set_x(x))

            x = np.linspace(-1, 1, 100);   x = sorted(np.hstack([x, best_quat[i]]))
            y = [func(x_) for x_ in x]
            x = fminbound(func, x[np.argmin(y) - 1], x[np.argmin(y) + 1], xtol=1e-8)
            best_quat = set_x(x)

        camera_model = StandardCamera(colmap_camera.width, colmap_camera.height,
                                    K, dist, [0, 0, 0], best_quat,
                                    platform_pose_provider=nav_state_provider)

        ub.ensuredir(save_dir)

        camera_model.save_to_file('%s/%s.yaml' % (save_dir, camera_str))

        perform_error_analysis(camera_model,
                               points_per_image_,
                               save_dir,
                               camera_str)


def create_time_modality_mapping(colmap_images, basename_to_time):
    print("Creating mapping between RGB and UV images...")
    time_to_modality = ub.AutoDict()
    for image in colmap_images.values():
        base_name = get_base_name(image.name)
        try:
            t = basename_to_time[base_name]
        except Exception as e:
            print(e)
            print(f"No ins time found for image {base_name}.")
            continue 
        modality = get_modality(image.name) 
        time_to_modality[t][modality] = image
    return time_to_modality


def create_fname_to_time_channel_modality(img_fnames, basename_to_time):
    print("Creating mapping between RGB and UV images...")
    time_to_modality = ub.AutoDict()
    for fname in img_fnames:
        base_name = get_base_name(fname)
        try:
            t = basename_to_time[base_name]
        except Exception as e:
            print(e)
            print(f"No ins time found for image {base_name}.")
            continue 
        modality = get_modality(fname) 
        channel = get_channel(fname) 
        time_to_modality[t][channel][modality] = fname
    return time_to_modality


def write_gifs(gif_dir, colmap_dir, img_fnames,
               fname_to_time_channel_modality,
               basename_to_time, rgb_str, camera_str,
               cm_rgb, cm_uv):
        print(f"Writing a registration gif for cameras {rgb_str} "
              f"and {camera_str}.")
        # Pick an image pair and register.
        ub.ensuredir(gif_dir)

        for k in range(10):
            inds = list(range(len(img_fnames)))
            shuffle(inds)
            for i in range(len(img_fnames)):
                uv_img = rgb_img = None
                fname1 = img_fnames[inds[i]]
                if osp.basename(osp.dirname(fname1)) != rgb_str:
                    continue
                t1 = basename_to_time[get_base_name(fname1)]
                channel = get_channel(fname1) # L/C/R
                try:
                    rgb_fname = fname_to_time_channel_modality[t1][channel]["rgb"]
                    abs_rgb_fname = os.path.join(colmap_dir, 'images0', rgb_fname)
                    rgb_img = cv2.imread(abs_rgb_fname, cv2.IMREAD_COLOR)[:, :, ::-1]
                except Exception as e:
                    print(f"No rgb image found at time {t1}")
                    continue
                try:
                    uv_fname = fname_to_time_channel_modality[t1][channel]["uv"]
                    abs_uv_fname = os.path.join(colmap_dir, 'images0', uv_fname)
                    uv_img = cv2.imread(abs_uv_fname, cv2.IMREAD_COLOR)[:, :, ::-1]
                    break
                except Exception as e:
                    print(f"No uv image found at time {t1}")
                    continue

            if uv_img is None or rgb_img is None:
                print("Failed to find matching image pair, skipping.")
                continue
            print(f"Writing {rgb_fname} and {uv_fname} to gif.")

            # Warps the color image img1 into the uv camera model cm_uv
            warped_rgb_img, mask = render_view(cm_rgb, rgb_img, 0,
                                               cm_uv, 0, block_size=10)

            ds_warped_rgb_img = PIL.Image.fromarray(cv2.pyrDown(
                                        cv2.pyrDown(cv2.pyrDown(warped_rgb_img))))
            ds_uv_img = PIL.Image.fromarray(cv2.pyrDown(
                                        cv2.pyrDown(cv2.pyrDown(uv_img))))
            fname_out = osp.join(gif_dir,
                                 f"{rgb_str}_to_{camera_str}_registration_{k+1}.gif")
            print(f"Writing gif to {fname_out}.")
            ds_uv_img.save(fname_out, save_all=True,
                           append_images=[ds_warped_rgb_img],
                           duration=350, loop=0)


def calibrate_uv(uv_camera_strs, img_fnames, colmap_images,
                 camera_from_camera_str, save_dir,
                 basename_to_time, time_to_modality,
                 fname_to_time_channel_modality,
                 colmap_dir, points_per_image):
    nav_state_fixed = NavStateFixed(np.zeros(3), [0, 0, 0, 1])
    skipped = 0
    total = 0
    for uv_str in uv_camera_strs:
        print(f"Matching images to camera {uv_str}.")
        rgb_str = uv_str.replace('uv', 'rgb')
        cm_rgb = StandardCamera.load_from_file(osp.join(save_dir, rgb_str + '.yaml'),
                                            platform_pose_provider=nav_state_fixed)
        im_pts_uv = []
        im_pts_rgb = []

        # Build up pairs of image coordinates between the two cameras from image
        # pairs acquired from the same time.
        image_nums = sorted(list(colmap_images.keys()))
        for image_num in image_nums:
            #print('%i/%i' % (image_num + 1, image_nums[-1]))
            image = colmap_images[image_num]
            im_str = osp.basename(osp.dirname(image.name))
            if im_str != uv_str:
                #print(f"{im_str} does not match {camera_str}, skipping.")
                continue

            # now we know it's uv
            image_uv = image
            base_name = get_base_name(image_uv.name)

            try:
                t1 = basename_to_time[base_name]
            except KeyError:
                print(f"No time found for {base_name}.")
                continue

            try:
                image_rgb = time_to_modality[t1]["rgb"]
            except KeyError:
                print(f"No rgb image found at {t1}.")
                continue

            # Both 'uv_image' and 'image_rgb' are from the same time.
            pt_ids1 = image_uv.point3D_ids
            ind = pt_ids1 != -1
            xys1 = dict(zip(pt_ids1[ind], image_uv.xys[ind]))

            pt_ids2 = image_rgb.point3D_ids
            ind = pt_ids2 != -1
            xys2 = dict(zip(pt_ids2[ind], image_rgb.xys[ind]))

            match_ids = set(xys1.keys()).intersection(set(xys2.keys()))
            total += 1
            if len(match_ids) < 1:
                #print("No match IDs found.")
                skipped += 1
                continue

            for match_id in match_ids:
                im_pts_uv.append(xys1[match_id])
                im_pts_rgb.append(xys2[match_id])

        print(f"Matched {total-skipped}/{total} image pairs, resulting in "
              f"{len(im_pts_uv)} matching UV and RGB points.")

        im_pts_uv = np.array(im_pts_uv)
        im_pts_rgb = np.array(im_pts_rgb)
        # Arbitrary cut off
        minimum_pts_required = 10
        if len(im_pts_rgb) < minimum_pts_required or \
                len(im_pts_uv) < minimum_pts_required:
            print("[ERROR] Not enough matching RGB/UV image points were found "
                  f"for camera {uv_str}.")
            continue

        if False:
            plt.subplot(121)
            plt.plot(im_pts_uv[:, 0], im_pts_uv[:, 1], 'ro')
            plt.subplot(122)
            plt.plot(im_pts_rgb[:, 0], im_pts_rgb[:, 1], 'bo')

        # Treat as co-located cameras (they are) and unproject out of RGB and into
        # the other camera.
        ray_pos, ray_dir = cm_rgb.unproject(im_pts_rgb.T)
        wrld_pts = ray_dir.T*1e4
        assert np.all(np.isfinite(wrld_pts)), "World points contain non-finite values."

        colmap_camera = camera_from_camera_str[uv_str]

        if colmap_camera.model == 'OPENCV':
            fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params
        elif colmap_camera.model == 'PINHOLE':
            fx, fy, cx, cy = colmap_camera.params
            d1 = d2 = d3 = d4 = 0

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([d1, d2, d3, d4], dtype=np.float32)

        flags = cv2.CALIB_ZERO_TANGENT_DIST
        flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
        flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
        flags = flags | cv2.CALIB_FIX_K1
        flags = flags | cv2.CALIB_FIX_K2
        flags = flags | cv2.CALIB_FIX_K3
        flags = flags | cv2.CALIB_FIX_K4
        flags = flags | cv2.CALIB_FIX_K5
        flags = flags | cv2.CALIB_FIX_K6

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30000,
                    0.0000001)

        ret = cv2.calibrateCamera([wrld_pts.astype(np.float32)],
                                [im_pts_uv.astype(np.float32)],
                                (colmap_camera.width, colmap_camera.height),
                                cameraMatrix=K.copy(), distCoeffs=dist.copy(),
                                flags=flags, criteria=criteria)

        err, _, _, rvecs, tvecs = ret

        R = np.identity(4)
        R[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
        cam_quat = quaternion_from_matrix(R.T)

        # Only optimize 3/4 components of the quaternion.
        static_quat_ind = np.argmax(np.abs(cam_quat))
        dynamic_quat_ind = [ i for i in range(4) if i != static_quat_ind ]
        #static_quat_ind = 3            # Fixing the 'w' component
        #dynamic_quat_ind = [0, 1, 2]   # Optimizing 'x', 'y', 'z' components
        dynamic_quat_ind = np.array(dynamic_quat_ind)
        cam_quat = np.asarray(cam_quat)
        cam_quat /= np.linalg.norm(cam_quat)
        x0 = cam_quat[dynamic_quat_ind].copy()  # [x, y, z]

        def get_cm(x):
            """
            Create a camera model with updated quaternion and intrinsic parameters.
        
            Parameters:
            - x: array-like, shape (N,)
                Optimization variables where the first 3 elements correspond to
                the dynamic quaternion components ('x', 'y', 'z'), optionally
                followed by intrinsic parameters ('fx', 'fy', etc.).
        
            Returns:
            - cm: StandardCamera instance
                Updated camera model with new parameters.
            """
            # Ensure 'x' has at least 3 elements for quaternion
            assert len(x) > 2, "Optimization variable 'x' must have at least 3 elements for quaternion."
            
            # Validate 'x[:3]' are finite numbers
            assert np.all(np.isfinite(x[:3])), "Quaternion components contain non-finite values."
            
            # Initialize quaternion with fixed 'w' component
            cam_quat_new = np.ones(4)
            
            # Assign dynamic components from optimization variables
            cam_quat_new[dynamic_quat_ind] = x[:3]
            
            # Normalize to ensure it's a unit quaternion
            norm = np.linalg.norm(cam_quat_new)
            assert norm > 1e-6, "Quaternion has zero or near-zero magnitude."
            cam_quat_new /= norm
        
            # Extract intrinsic parameters
            if len(x) > 3:
                fx_ = x[3]
                fy_ = x[4]
            else:
                fx_ = fx
                fy_ = fy
        
            if len(x) > 5:
                dist_ = x[5:]
            else:
                dist_ = dist
        
            # Construct the intrinsic matrix
            K = np.array([[fx_, 0, cx], [0, fy_, cy], [0, 0, 1]])
        
            # Create the camera model
            cm = StandardCamera(
                colmap_camera.width,
                colmap_camera.height,
                K,
                dist_,
                [0, 0, 0],
                cam_quat_new,
                platform_pose_provider=nav_state_fixed
            )
            return cm

        def error(x):
            try:
                cm = get_cm(x)
                projected_uv = cm.project(wrld_pts.T).T  # Shape: (N, 2)

                # Compute Euclidean distances
                err = np.sqrt(np.sum((im_pts_uv - projected_uv) ** 2, axis=1))

                # Apply Huber loss
                delta = 20
                ind = err < delta
                err[ind] = err[ind] ** 2
                err[~ind] = 2 * (err[~ind] - delta / 2) * delta

                # Sort and trim the error
                err = sorted(err)[:len(err) - len(err) // 5]

                # Compute mean error
                mean_err = np.sqrt(np.mean(err))

                # Add regularization term (e.g., L2 penalty)
                reg_strength = 1e-3  # Adjust as needed
                reg_term = reg_strength * np.linalg.norm(x[:3])**2

                total_error = mean_err + reg_term
                return total_error
            except Exception as e:
                print(f"Error in error function: {e}")
                return np.inf  # Assign a high error if computation fails

        # Optional: Define a callback function to monitor optimization
        def callback(xk):
            try:
                cm = get_cm(xk)
                projected_uv = cm.project(wrld_pts.T).T
                err = np.sqrt(np.sum((im_pts_uv - projected_uv) ** 2, axis=1))
                mean_err = np.mean(err)
                print(f"Current x: {xk}, Mean Error: {mean_err}")
            except Exception as e:
                print(f"Error in callback: {e}")

        def plot_results1(x):
            cm = get_cm(x)
            err = np.sqrt(np.sum((im_pts_uv - cm.project(wrld_pts.T).T)**2, 1))
            err = sorted(err)
            plt.plot(np.linspace(0, 100, len(err)), err)

        print("Optimizing error for UV models.")
        x = x0.copy()
        # Example bounds for [x, y, z] components
        bounds = [(-1.0, 1.0),  # x
                  (-1.0, 1.0),  # y
                  (-1.0, 1.0)]  # z
        print("First pass")
        # Perform optimization on [x, y, z]
        ret = minimize(
            error,
            x,
            method='L-BFGS-B',
            bounds=bounds,
            callback=None,  # Optional: Monitor progress
            options={'disp': False, 'maxiter': 30000, 'ftol': 1e-7}
        )
        assert ret.success, "Minimization of UV error failed."
        x = np.hstack([ret.x, fx, fy])
        print("Second pass")
        assert np.all(np.isfinite(x)), "Input quaternion with locked fx, fy, is not finite."
        ret = minimize(error, x, method='Powell')
        x = ret.x
        print("Third pass")
        assert np.all(np.isfinite(x)), "Input quaternion for BFGS is not finite."
        ret = minimize(error, x, method='BFGS')

        print("Final pass")
        if True:
            x = np.hstack([ret.x, dist])
            ret = minimize(error, x, method='Powell');    x = ret.x
            ret = minimize(error, x, method='BFGS');    x = ret.x

        assert np.all(np.isfinite(x)), "Input quaternion for final model is not finite."
        cm_uv = get_cm(x)
        cm_uv.save_to_file('%s/%s.yaml' % (save_dir, uv_str))

        perform_error_analysis_and_save_pdf(cm_uv, cm_rgb, points_per_image,
                                            save_dir,
                                            uv_str, rgb_str)
        gif_dir = osp.join(save_dir, 'registration_gifs')
        write_gifs(gif_dir, colmap_dir, img_fnames,
                   fname_to_time_channel_modality,
                   basename_to_time, rgb_str, uv_str,
                   cm_rgb, cm_uv)


def perform_error_analysis_and_save_pdf(camera_model_uv, camera_model_rgb,
                                        points_per_image,
                                        save_dir,
                                        uv_str,
                                        rgb_str):
    """
    Perform error analysis for both UV and RGB models and save all plots into a single PDF.

    Parameters:
    - camera_model_uv: Calibrated UV camera model.
    - camera_model_rgb: Calibrated RGB camera model.
    - points_per_image_uv: List of tuples containing (image points, corresponding 3D points, timestamp) for UV.
    - points_per_image_rgb: List of tuples containing (image points, corresponding 3D points, timestamp) for RGB.
    - save_dir: Directory where the PDF will be saved.
    - uv_str: String identifier for the UV camera.
    - rgb_str: String identifier for the RGB camera.
    """

    # Initialize error lists for UV
    err_meters_uv = []
    err_pixels_uv = []
    err_pixels_per_frame_uv = []
    err_angle_uv = []
    im_pts_uv = []
    im_pts_rgb = []

    # Mean IFOV for UV (assuming similar to RGB)
    ifov_uv = np.mean(camera_model_uv.ifov())

    #import ipdb; ipdb.set_trace()
    # Error analysis for UV
    for fname, (xys, xyzs, t) in points_per_image.items():
        im_str = osp.basename(osp.dirname(fname))
        if im_str != uv_str:
            continue
        im_pts_uv.extend(xys)
        # Project 3D points to 2D image points using UV model
        xys2 = camera_model_uv.project(xyzs.T, t)
        err_pixels_ = np.sqrt(np.sum((xys2 - xys.T)**2, axis=0))
        err_pixels_per_frame_uv.append([t, err_pixels_.mean()])
        err_pixels_uv.extend(err_pixels_.tolist())

        # Unproject image points to camera rays
        ray_pos, ray_dir = camera_model_uv.unproject(xys.T, t)

        # Compute direction from camera to 3D points
        ray_dir2 = xyzs.T - ray_pos
        dist = np.linalg.norm(ray_dir2, axis=0)
        ray_dir2 /= dist

        # Calculate angular deviation
        dp = np.clip(np.sum(ray_dir * ray_dir2, axis=0), -1, 1)
        theta = np.arccos(dp)
        err_angle_uv.extend(theta.tolist())

        # Calculate orthogonal distance in meters
        err_meters_uv.extend((np.sin(theta) * dist).tolist())

    # Initialize error lists for RGB
    err_meters_rgb = []
    err_pixels_rgb = []
    err_pixels_per_frame_rgb = []
    err_angle_rgb = []

    # Mean IFOV for RGB
    ifov_rgb = np.mean(camera_model_rgb.ifov())

    # Error analysis for RGB
    for fname, (xys, xyzs, t) in points_per_image.items():
        im_str = osp.basename(osp.dirname(fname))
        if im_str != rgb_str:
            continue
        im_pts_rgb.extend(xys)
        # Project 3D points to 2D image points using RGB model
        xys2 = camera_model_rgb.project(xyzs.T, t)
        err_pixels_ = np.sqrt(np.sum((xys2 - xys.T)**2, axis=0))
        err_pixels_per_frame_rgb.append([t, err_pixels_.mean()])
        err_pixels_rgb.extend(err_pixels_.tolist())

        # Unproject image points to camera rays
        ray_pos, ray_dir = camera_model_rgb.unproject(xys.T, t)

        # Compute direction from camera to 3D points
        ray_dir2 = xyzs.T - ray_pos
        dist = np.linalg.norm(ray_dir2, axis=0)
        ray_dir2 /= dist

        # Calculate angular deviation
        dp = np.clip(np.sum(ray_dir * ray_dir2, axis=0), -1, 1)
        theta = np.arccos(dp)
        err_angle_rgb.extend(theta.tolist())

        # Calculate orthogonal distance in meters
        err_meters_rgb.extend((np.sin(theta) * dist).tolist())

    # Convert lists to numpy arrays for easier manipulation
    err_meters_uv = np.array(err_meters_uv)
    err_pixels_uv = np.array(err_pixels_uv)
    err_angle_uv = np.array(err_angle_uv)
    err_pixels_per_frame_uv = np.array(err_pixels_per_frame_uv).T

    err_meters_rgb = np.array(err_meters_rgb)
    err_pixels_rgb = np.array(err_pixels_rgb)
    err_angle_rgb = np.array(err_angle_rgb)
    err_pixels_per_frame_rgb = np.array(err_pixels_per_frame_rgb).T

    # Sort the errors
    sorted_err_meters_uv = np.sort(err_meters_uv)
    sorted_err_pixels_uv = np.sort(err_pixels_uv)
    sorted_err_angle_uv = np.sort(err_angle_uv)

    sorted_err_meters_rgb = np.sort(err_meters_rgb)
    sorted_err_pixels_rgb = np.sort(err_pixels_rgb)
    sorted_err_angle_rgb = np.sort(err_angle_rgb)

    im_pts_uv = np.asarray(im_pts_uv)
    im_pts_rgb = np.asarray(im_pts_rgb)

    # Initialize PdfPages object
    pdf_filename = f"{save_dir}/error_analysis_{uv_str}_{rgb_str}.pdf"
    with PdfPages(pdf_filename) as pdf:
        # --- Plot 1: Summary Statistics for UV ---
        plt.figure(figsize=(11.69, 8.27))  # A4 size in inches
        plt.axis('off')  # Hide axes

        summary_text_uv = f"""
        Error Analysis Summary for UV Camera: {uv_str}

        Pixel Errors:
        - Mean: {np.mean(err_pixels_uv):.2f} pixels
        - Median: {np.median(err_pixels_uv):.2f} pixels
        - Max: {np.max(err_pixels_uv):.2f} pixels

        Meter Errors:
        - Mean: {np.mean(err_meters_uv):.2f} meters
        - Median: {np.median(err_meters_uv):.2f} meters
        - Max: {np.max(err_meters_uv):.2f} meters

        Angular Errors:
        - Mean: {np.degrees(np.mean(err_angle_uv)):.2f} degrees
        - Median: {np.degrees(np.median(err_angle_uv)):.2f} degrees
        - Max: {np.degrees(np.max(err_angle_uv)):.2f} degrees
        """

        plt.text(0.5, 0.5, summary_text_uv, fontsize=20, ha='center', va='center', wrap=True)
        plt.title('Error Analysis Summary for UV Camera', fontsize=24)
        pdf.savefig()
        plt.close()

        # --- Plot 2: Histogram of Pixel Errors for UV ---
        plt.figure(figsize=(11.69, 8.27))
        plt.hist(sorted_err_pixels_uv, bins=50, color='blue', alpha=0.7)
        plt.title('UV Camera - Pixel Errors', fontsize=24)
        plt.xlabel('Error (pixels)', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # --- Plot 3: Histogram of Meter Errors for UV ---
        plt.figure(figsize=(11.69, 8.27))
        plt.hist(sorted_err_meters_uv, bins=50, color='green', alpha=0.7)
        plt.title('UV Camera - Meter Errors', fontsize=24)
        plt.xlabel('Error (meters)', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # --- Plot 4: Histogram of Angular Errors for UV ---
        plt.figure(figsize=(11.69, 8.27))
        plt.hist(np.degrees(sorted_err_angle_uv), bins=50, color='red', alpha=0.7)
        plt.title('UV Camera - Angular Errors', fontsize=24)
        plt.xlabel('Error (degrees)', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # --- Plot 5: Average Pixel Error per Frame for UV ---
        plt.figure(figsize=(11.69, 8.27))
        plt.plot(err_pixels_per_frame_uv[0], err_pixels_per_frame_uv[1], marker='o', linestyle='-', color='purple')
        plt.title('UV Camera - Average Pixel Error per Frame', fontsize=24)
        plt.xlabel('Frame Index or Timestamp', fontsize=20)
        plt.ylabel('Average Pixel Error (pixels)', fontsize=20)
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # --- Plot 6: Scatter Plot of UV and RGB Points ---
        plt.figure(figsize=(11.69, 8.27))
        plt.subplot(1, 2, 1)
        plt.scatter(im_pts_uv[:, 0], im_pts_uv[:, 1], c='blue', marker='o', alpha=0.5, label='UV Observed')
        plt.title('UV Camera - Observed UV Points', fontsize=24)
        plt.xlabel('U', fontsize=20)
        plt.ylabel('V', fontsize=20)
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(im_pts_rgb[:, 0], im_pts_rgb[:, 1], c='green', marker='x', alpha=0.5, label='RGB Observed')
        plt.title('RGB Camera - Observed RGB Points', fontsize=24)
        plt.xlabel('R', fontsize=20)
        plt.ylabel('G', fontsize=20)
        plt.legend()
        plt.grid(True)

        plt.suptitle('Scatter Plots of Observed Points', fontsize=28)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig()
        plt.close()

        # --- Plot 7: Reprojection Error Histograms for Both UV and RGB ---
        plt.figure(figsize=(11.69, 8.27))
        plt.subplot(1, 2, 1)
        plt.hist(sorted_err_pixels_uv, bins=50, color='blue', alpha=0.7, label='UV Pixel Errors')
        plt.hist(sorted_err_pixels_rgb, bins=50, color='red', alpha=0.5, label='RGB Pixel Errors')
        plt.title('Reprojection Pixel Errors', fontsize=24)
        plt.xlabel('Error (pixels)', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(np.degrees(sorted_err_angle_uv), bins=50, color='blue', alpha=0.7, label='UV Angular Errors')
        plt.hist(np.degrees(sorted_err_angle_rgb), bins=50, color='red', alpha=0.5, label='RGB Angular Errors')
        plt.title('Reprojection Angular Errors', fontsize=24)
        plt.xlabel('Error (degrees)', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.legend()
        plt.grid(True)

        plt.suptitle('Reprojection Error Histograms for UV and RGB Cameras', fontsize=28)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig()
        plt.close()


def process_aligned_results(aligned_sparse_recon_subdir, colmap_dir, save_dir,
                            nav_state_provider, basename_to_time):
    # ---------------------------------------------------------------------------
    # Sanity check, pick the coordinates for a point in the 3-D model and
    # convert them to latitude and longitude.
    enu = np.array((640.446167, 822.111633, -9.576390))
    print(enu_to_llh(enu[0], enu[1], enu[2], nav_state_provider.lat0,
                    nav_state_provider.lon0, nav_state_provider.h0))

    # Read in the Colmap details of all images.
    images_bin_fname = osp.join(colmap_dir,
                                aligned_sparse_recon_subdir,
                                'images.bin')
    colmap_images = read_images_binary(images_bin_fname)
    points_bin_fname = osp.join(colmap_dir,
                                aligned_sparse_recon_subdir,
                                'points3D.bin')
    points3d = read_points3D_binary(points_bin_fname)
    camera_bin_fname = osp.join(colmap_dir,
                                aligned_sparse_recon_subdir,
                                'cameras.bin')
    colmap_cameras = read_cameras_binary(camera_bin_fname)

    """
        # For sanity checking that the original unadjusted results line up and the
        # code itself is sound.
        images_bin_fname = '%s/%s/images.bin' % (colmap_dir, sparse_recon_subdir)
        colmap_images = read_images_binary(images_bin_fname)
        points_bin_fname = '%s/%s/points3D.bin' % (colmap_dir, sparse_recon_subdir)
        points3d = read_points3D_binary(points_bin_fname)
        camera_bin_fname = '%s/%s/cameras.bin' % (colmap_dir, sparse_recon_subdir)
        colmap_cameras = read_cameras_binary(camera_bin_fname)
    """

    if False:
        pts_3d = []
        for pt_id in points3d:
            pts_3d.append(points3d[pt_id].xyz)

        pts_3d = np.array(pts_3d).T
        plt.plot(pts_3d[0], pts_3d[1], 'ro')


    points_per_image, camera_from_camera_str = get_colmap_data(colmap_images,
                                                               colmap_cameras,
                                                               points3d,
                                                               basename_to_time)

    img_fnames, img_times, ins_poses, sfm_poses, llhs = process_images(colmap_images,
                                                                       basename_to_time,
                                                                       nav_state_provider)

    if False:
        # Loop over all images and apply the camera model to project 3-D points
        # into the image and compare to the measured versions to calculate
        # reprojection error.
        err = []
        for i in range(len(img_fnames)):
            print('%i/%i' % (i + 1, len(img_fnames)))
            fname = img_fnames[i]
            sfm_pose = sfm_poses[i]
            camera_str = osp.basename(osp.dirname(fname))

            colmap_camera = camera_from_camera_str[camera_str]

            if colmap_camera.model == 'OPENCV':
                    fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params
            elif colmap_camera.model == 'PINHOLE':
                fx, fy, cx, cy = colmap_camera.params
                d1 = d2 = d3 = d4 = 0

            K = K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            dist = np.array([d1, d2, d3, d4])

            cm = StandardCamera(colmap_camera.width, colmap_camera.height, K, dist,
                                [0, 0, 0], [0, 0, 0, 1],
                                platform_pose_provider=NavStateFixed(*sfm_pose))
            xy, xyz, t = points_per_image[fname]
            err_ = np.sqrt(np.sum((xy - cm.project(xyz.T, t).T)**2, axis=1))
            err = err + err_.tolist()

        print("Errors: ")
        print(np.mean(err))
        print(np.median(err))
        #plt.hist(err, 1000)


    camera_strs = set([osp.basename(osp.dirname(fname)) for fname in img_fnames])
    rgb_camera_strs = set([ cam for cam in camera_strs if 'rgb' in cam ])
    uv_camera_strs = set([ cam for cam in camera_strs if 'uv' in cam ])

    print("Calibrating RGB cameras.")
    calibrate_rgb(rgb_camera_strs, img_fnames, ins_poses, sfm_poses,
                  points_per_image, camera_from_camera_str,
                  nav_state_provider, save_dir)

    time_to_modality = create_time_modality_mapping(colmap_images, basename_to_time)
    fname_to_time_channel_modality = create_fname_to_time_channel_modality(
                                    img_fnames, basename_to_time)

    print("Calibrating UV cameras.")
    calibrate_uv(uv_camera_strs, img_fnames, colmap_images,
                 camera_from_camera_str, save_dir,
                 basename_to_time, time_to_modality,
                 fname_to_time_channel_modality,
                 colmap_dir, points_per_image)
    print("Finished calibration!")


def main():
    # ---------------------------- Define Paths ----------------------------------
    # KAMERA flight directory where each sub-directory contains meta.json files.
    flight_dir = '/home/local/KHQ/adam.romlein/noaa/data/2024_AOC_AK_Calibration/fl09'

    # You should have a colmap directory where all of the Colmap-generated files
    # reside.
    colmap_dir = '/home/local/KHQ/adam.romlein/noaa/data/2024_AOC_AK_Calibration/colmap'

    # Sub-directory containing the images.bin and cameras.bin. Set to '' if in the
    # top-level Colmap directory.
    sparse_recon_subdir = 'sparse/1'
    aligned_sparse_recon_subdir = 'aligned/1'

    # Location to save KAMERA camera models.
    save_dir = osp.join(flight_dir, 'kamera_models')
    # ----------------------------------------------------------------------------

    basename_to_time = get_basename_to_time(flight_dir)

    json_glob = pathlib.Path(flight_dir).rglob('*_meta.json')
    try:
        next(json_glob)
    except StopIteration:
        raise SystemExit("No meta jsons were found, please check your filepaths.")
    nav_state_provider = NavStateINSJson(json_glob)

    # We take the INS-reported position (converted from latitude, longitude, and
    # altitude into easting/northing/up coordinates) and assign it to each image.
    print('Latiude of ENU coordinate system:', nav_state_provider.lat0, 'degrees')
    print('Longitude of ENU coordinate system:', nav_state_provider.lon0,
        'degrees')
    print('Height above the WGS84 ellipsoid of the ENU coordinate system:',
        nav_state_provider.h0, 'meters')

    # ----------------------------------------------------------------------------
    # Assemble the list of filenames with paths relative to the 'images0' directory
    # that we point Colmap to as the raw image directory. This may be a directory
    # of images, or it might be a directory of subdirectories, each of which
    # contains images from one camera.

    # Colmap then uses this pairing to solve for a similarity transform to best-
    # match the SfM poses it recovered into these positions. All Colmap coordinates
    # in this aligned version of its reconstruction will then be in easting/
    # northing/up meters coordinates
    align_fname = os.path.join(colmap_dir, 'image_locations.txt')
    print(align_fname)
    if osp.exists(align_fname) and osp.exists(osp.join(colmap_dir,
                                                       aligned_sparse_recon_subdir)):
        print(f"{align_fname} and {aligned_sparse_recon_subdir} exists,"
            " assuming model is aligned.")
    else:
        # Read in the Colmap details of all images.
        images_bin_fname = osp.join(colmap_dir, sparse_recon_subdir, 'images.bin')
        colmap_images = read_images_binary(images_bin_fname)

        img_fnames, img_times, ins_poses, sfm_poses, llhs = process_images(colmap_images,
                                                                           basename_to_time,
                                                                           nav_state_provider)
        write_image_locations(align_fname, img_fnames, ins_poses)
        ub.ensuredir(osp.join(colmap_dir, aligned_sparse_recon_subdir))
        print('Now run\nkamera/src/kitware-ros-pkg/postflight_scripts/scripts/'
            'colmap/model_aligner.sh %s %s %s %s' % (colmap_dir.replace('/host_filesystem', ''),
                                                    sparse_recon_subdir,
                                                    'image_locations.txt',
                                                    aligned_sparse_recon_subdir))
        return

    process_aligned_results(aligned_sparse_recon_subdir, colmap_dir,
                            save_dir, nav_state_provider, 
                            basename_to_time)

if __name__ == "__main__":
    main()

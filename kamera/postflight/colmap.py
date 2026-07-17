import os
import copy
import json
import pathlib
import cv2
import PIL.Image
import numpy as np
import os.path as osp
import ubelt as ub
from rich import print

import pycolmap as pc
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from kamera.sensor_models.nav_state import NavStateFixed, NavStateINSJson
from kamera.colmap_processing.camera_models import StandardCamera
from kamera.colmap_processing.image_renderer import render_view
from kamera.postflight.alignment import (
    VisiblePoint,
    iterative_alignment,
    manual_alignment,
    transfer_alignment,
)
from kamera.postflight.naming import (
    KameraCameraName,
    KameraImageName,
    swap_image_name_modality,
)


@dataclass(frozen=True)
class ColmapCalibrationData:
    # each sfm pose is a [position (3,), quaternion (4,)] pair
    sfm_poses: List[list]
    ins_poses: List[np.ndarray]
    img_fnames: List[str]
    img_times: List[float]
    llhs: List[np.ndarray]
    points_per_image: List[List[VisiblePoint]]
    basename_to_time: Dict[str, float]
    fname_to_time_channel_modality: Dict[float, Dict[str, Dict[str, str]]]
    best_cameras: Dict[str, pc.Camera]


class ColmapCalibration(object):
    def __init__(
            self, flight_dir: str | os.PathLike,
            recon_dir: str | os.PathLike,
            colmap_dir: str | os.PathLike
    ) -> None:
        self.flight_dir = flight_dir
        self.recon_dir = recon_dir
        self.colmap_dir = colmap_dir
        # contains the images, 3D points, and cameras of the colmap database
        self.R = pc.Reconstruction(self.recon_dir)
        # populated by prepare_calibration_data()
        self.ccd: Optional[ColmapCalibrationData] = None
        self.nav_state_provider = self.load_nav_state_provider()
        print(self.R.summary())

    def load_nav_state_provider(self):
        # Create navigation stream
        json_glob = pathlib.Path(self.flight_dir).rglob("*_meta.json")
        try:
            next(json_glob)
        except StopIteration:
            raise SystemExit("No meta jsons were found, please check your filepaths.")
        return NavStateINSJson((json_glob))

    def get_base_name(self, fname: str | os.PathLike) -> str:
        """Given an arbitrary filename (could be UV, IR, RGB, json),
        extract the portion of the filename that is just the time, flight,
        machine (C, L, R), and effort name.
        """
        return KameraImageName.parse(fname).base_name

    def get_modality(self, fname: str | os.PathLike) -> str:
        """Extract image modality (e.g. ir/meta) from a given kamera filename."""
        return KameraImageName.parse(fname).modality

    def get_channel(self, fname: str | os.PathLike) -> str:
        """Extract channel (e.g. L/R/C) from a given kamera filename."""
        return KameraImageName.parse(fname).channel

    def get_view(self, fname: str | os.PathLike) -> str:
        """Extract view (e.g. left_view, right_view, center_view) from
        a given kamera filename"""
        return KameraImageName.parse(fname).view

    def get_basename_to_time(self, flight_dir: str | os.PathLike) -> dict:
        # Establish correspondence between real-world exposure times base of file
        # names.
        basename_to_time = {}
        for json_fname in pathlib.Path(flight_dir).rglob("*_meta.json"):
            try:
                with open(json_fname) as json_file:
                    d = json.load(json_file)
                    # Time that the image was taken.
                    basename = self.get_base_name(json_fname)
                    basename_to_time[basename] = float(d["evt"]["time"])
            except (OSError, ValueError):
                pass
        return basename_to_time

    def prepare_calibration_data(self):
        img_fnames = []
        img_times = []
        ins_poses = []
        sfm_poses = []
        llhs = []
        points_per_image = []
        basename_to_time = self.get_basename_to_time(self.flight_dir)
        best_cameras = {}
        most_points = {}
        for image_num, image in self.R.images.items():
            try:
                base_name = self.get_base_name(image.name)
                t = basename_to_time[base_name]
            except (KeyError, ValueError):
                print(
                    "Couldn't find a _meta.json file associated with '%s'" % image.name
                )
                continue

            # Query the navigation state recorded by the INS for this time.
            ins_pose = self.nav_state_provider.pose(t)
            llh = self.nav_state_provider.llh(t)

            # Query Colmaps pose for the camera.
            pose = image.cam_from_world()
            R = pose.rotation.matrix()
            pos = -np.dot(R.T, pose.translation)

            pose.rotation.normalize()
            quat = pose.rotation.quat
            # invert the w (rotation) component,
            #  so we get the camera to world rotation
            quat[3] *= -1

            sfm_pose = [pos, quat]

            img_times.append(t)
            ins_poses.append(ins_pose)
            img_fnames.append(image.name)
            sfm_poses.append(sfm_pose)
            llhs.append(llh)
            points = []
            # associate all the 2D and 3D points seen by this image
            for pt in image.points2D:
                if pt.has_point3D():
                    point_2d = pt.xy
                    uncertainty = self.R.points3D[pt.point3D_id].error
                    point_3d = self.R.points3D[pt.point3D_id].xyz
                    visible = True
                    point_3d_id = pt.point3D_id
                    vpt = VisiblePoint(
                        point_3d, point_3d_id, point_2d, uncertainty, t, visible
                    )
                    points.append(vpt)
            points_per_image.append(points)
            camera_name = os.path.basename(os.path.dirname(image.name))
            # as a way to choose the first quaternion we check, choose the one
            # that has the most number of 3D points registered
            if camera_name in best_cameras:
                if image.num_points3D > most_points[camera_name]:
                    best_cameras[camera_name] = image.camera
                    most_points[camera_name] = image.num_points3D
            else:
                best_cameras[camera_name] = image.camera
                most_points[camera_name] = image.num_points3D

        # sort all entries
        ind = np.argsort(img_fnames)
        img_fnames = [img_fnames[i] for i in ind]
        img_times = [img_times[i] for i in ind]
        ins_poses = [ins_poses[i] for i in ind]
        sfm_poses = [sfm_poses[i] for i in ind]
        points_per_image = [points_per_image[i] for i in ind]
        llhs = [llhs[i] for i in ind]

        fname_to_time_channel_modality = self.create_fname_to_time_channel_modality(
            img_fnames, basename_to_time
        )

        ccd = ColmapCalibrationData(
            sfm_poses=sfm_poses,
            ins_poses=ins_poses,
            img_fnames=img_fnames,
            img_times=img_times,
            llhs=llhs,
            points_per_image=points_per_image,
            basename_to_time=basename_to_time,
            fname_to_time_channel_modality=fname_to_time_channel_modality,
            best_cameras=best_cameras,
        )
        self.ccd = ccd
        return ccd

    def calibrate_camera(
        self, camera_name: str
    ) -> Tuple[Optional[StandardCamera], Optional[float]]:
        assert self.ccd is not None, "call prepare_calibration_data() first"
        sfm_quats = np.array([pose[1] for pose in self.ccd.sfm_poses])
        ins_quats = np.array([pose[1] for pose in self.ccd.ins_poses])
        # Find all valid indices of current camera
        cam_idxs = [
            1 if camera_name == os.path.basename(os.path.dirname(im)) else 0
            for im in self.ccd.img_fnames
        ]
        print(f"Number of images in camera {camera_name}.")
        print(np.sum(cam_idxs))
        observations = [
            pts for i, pts in enumerate(self.ccd.points_per_image) if cam_idxs[i] == 1
        ]
        cam_sfm_quats = [q for i, q in enumerate(sfm_quats) if cam_idxs[i] == 1]
        cam_ins_quats = [q for i, q in enumerate(ins_quats) if cam_idxs[i] == 1]
        if len(observations) < 10:
            print(f"Only {len(observations)} seen for camera {camera_name}, skipping.")
            return None, None
        cam = self.ccd.best_cameras[camera_name]
        camera_model, error = iterative_alignment(
            cam_sfm_quats, cam_ins_quats, observations, cam, self.nav_state_provider
        )
        return camera_model, error

    def transfer_calibration(
        self,
        camera_name: str,
        calibrated_camera_model: StandardCamera,
        calibrated_modality: str,
    ) -> Tuple[Optional[StandardCamera], Optional[float]]:
        """
        Use the quaternion already solved of a colocated, calibrated camera
        to bootstrap the calibration process.
        """
        assert self.ccd is not None, "call prepare_calibration_data() first"
        # Find all valid indices of current cameras
        cam_idxs = [
            1 if camera_name == os.path.basename(os.path.dirname(im)) else 0
            for im in self.ccd.img_fnames
        ]
        print(f"Number of images in camera {camera_name}.")
        print(np.sum(cam_idxs))

        # Now we have to find the overlapping observations on a per-frame basis between
        # the camera to calibrate, and the colocated, calibrated, camera.
        colocated_observations = []
        observations = []
        for i, fname in enumerate(self.ccd.img_fnames):
            if cam_idxs[i] == 1:
                # Swap in the modality of the already calibrated camera
                calibrated_fname = swap_image_name_modality(fname, calibrated_modality)
                try:
                    ii = self.ccd.img_fnames.index(calibrated_fname)
                except ValueError:
                    print(f"Missing {calibrated_fname} from calibrated index.")
                    continue
                colocated_observations.append(self.ccd.points_per_image[ii])
                observations.append(self.ccd.points_per_image[i])
        if len(observations) < 10 or len(colocated_observations) < 10:
            print(
                f"Only {len(observations)} seen for camera {camera_name}, and "
                f" only {len(colocated_observations)} found for the calibrated camera, "
                " skipping."
            )
            return None, None

        cam = self.ccd.best_cameras[camera_name]
        camera_model, error = transfer_alignment(
            cam, calibrated_camera_model, observations, colocated_observations
        )
        return camera_model, error

    def manual_calibration(
        self,
        camera_model: StandardCamera,
        reference_camera_model: StandardCamera,
        point_pairs: dict,
    ) -> tuple[StandardCamera, float]:
        print("Refining camera model using manually defined point pairs.")
        refined_model, error = manual_alignment(
            camera_model, reference_camera_model, point_pairs
        )
        return refined_model, error

    def create_fname_to_time_channel_modality(
        self, img_fnames, basename_to_time
    ) -> Dict[float, Dict[str, Dict[str, str]]]:
        print("Creating mapping between RGB and UV images...")
        time_to_modality = ub.AutoDict()
        for fname in img_fnames:
            base_name = self.get_base_name(fname)
            try:
                t = basename_to_time[base_name]
            except Exception as e:
                print(e)
                print(f"No ins time found for image {base_name}.")
                continue
            modality = self.get_modality(fname)
            channel = self.get_channel(fname)
            time_to_modality[t][channel][modality] = fname
        return time_to_modality

    def _find_image_file(
        self,
        images_roots: List[str],
        cam_dir: str,
        bname: str,
    ) -> Optional[str]:
        """Find an image under any of the given images roots, trying common
        extensions if the exact basename is not present."""
        for root in images_roots:
            path = pathlib.Path(root) / cam_dir / bname
            candidates = [path] + [
                path.with_suffix(ext) for ext in (".jpg", ".jpeg", ".png", ".tiff")
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)
        return None

    def write_gifs(
        self,
        gif_dir: str | os.PathLike,
        camera_name: str,
        colocated_modality: str,
        camera_model: StandardCamera,
        colocated_camera_model: StandardCamera,
        num_gifs: int = 5,
    ) -> None:
        """Write registration gifs flipping between images of `camera_name`
        (its image-folder name, e.g. "85mm_25_5deg_center_uv") and the
        colocated camera's image warped into its view."""
        assert self.ccd is not None, "call prepare_calibration_data() first"
        cam_name = KameraCameraName.parse(camera_name)
        colocated_camera_name = cam_name.with_modality(colocated_modality).name
        print(
            f"Writing registration gifs for cameras {camera_name} "
            f"and {colocated_camera_name}."
        )
        ub.ensuredir(gif_dir)

        # Warp in the platform frame for both cameras. The models may arrive
        # with different pose providers (a freshly solved camera carries the
        # live INS provider, a loaded/transferred one an identity provider);
        # mixing them injects an arbitrary INS attitude into the warp even
        # when both mount quaternions are correct.
        nav_state_fixed = NavStateFixed(np.zeros(3), np.array([0, 0, 0, 1]))
        camera_model = copy.copy(camera_model)
        camera_model.platform_pose_provider = nav_state_fixed
        colocated_camera_model = copy.copy(colocated_camera_model)
        colocated_camera_model.platform_pose_provider = nav_state_fixed

        images_root = osp.join(self.colmap_dir, "images0")
        # The colocated modality may live in a different colmap workspace
        # (e.g. the rgb images paired with an ir model live under colmap_rgb).
        colocated_roots = [
            images_root,
            osp.join(self.flight_dir, f"colmap_{colocated_modality}", "images0"),
        ]

        img_fnames = self.ccd.img_fnames
        for k in range(num_gifs):
            inds = list(range(len(img_fnames)))
            np.random.shuffle(inds)
            img = colocated_img = None
            bname = colocated_bname = None
            for i in inds:
                fname = img_fnames[i]
                cam_dir = osp.basename(osp.dirname(fname))
                if cam_dir != camera_name:
                    continue
                bname = osp.basename(fname)
                abs_fname = self._find_image_file([images_root], cam_dir, bname)
                if abs_fname is None:
                    print(f"No {cam_name.modality} image found for {bname}")
                    continue
                img = cv2.imread(abs_fname, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Could not read image at {abs_fname}")
                    continue
                img = img[:, :, ::-1]

                # get the colocated image from the same time
                colocated_dir = (
                    KameraCameraName.parse(cam_dir)
                    .with_modality(colocated_modality)
                    .name
                )
                colocated_bname = (
                    KameraImageName.parse(bname)
                    .with_modality(colocated_modality)
                    .name
                )
                abs_colocated_fname = self._find_image_file(
                    colocated_roots, colocated_dir, colocated_bname
                )
                if abs_colocated_fname is None:
                    print(f"No {colocated_modality} image found for {bname}")
                    continue
                colocated_img = cv2.imread(abs_colocated_fname, cv2.IMREAD_COLOR)
                if colocated_img is None:
                    print(f"Could not read image at {abs_colocated_fname}")
                    continue
                colocated_img = colocated_img[:, :, ::-1]
                colocated_bname = osp.basename(abs_colocated_fname)
                # Once we find a matching pair, break
                break

            if img is None or colocated_img is None:
                print("Failed to find matching image pair, skipping.")
                continue
            print(f"Writing {bname} and {colocated_bname} to gif.")

            # warps the given colocated image into the view of the original camera model
            warped_colocated_img, mask = render_view(
                colocated_camera_model, colocated_img, 0, camera_model, 0, block_size=10
            )
            fname_out = osp.join(
                gif_dir,
                f"{camera_name}_to_{colocated_camera_name}_registration_{k}.gif",
            )
            print(f"Writing gif to {fname_out}.")
            # Make sure gifs are reasonable size
            h, w, _ = img.shape
            new_w = 1280
            new_h = int(new_w * h / w)
            pil_img = PIL.Image.fromarray(img).resize((new_w, new_h))
            pil_colocated_img = PIL.Image.fromarray(warped_colocated_img).resize(
                (new_w, new_h)
            )
            pil_img.save(
                fname_out,
                save_all=True,
                append_images=[pil_colocated_img],
                duration=350,
                loop=0,
            )

    def align_model(self, output_dir: str | os.PathLike) -> None:
        assert self.ccd is not None, "call prepare_calibration_data() first"
        img_fnames = [im.name for im in self.R.images.values()]
        points = []
        ins_poses = []
        for image_name in img_fnames:
            base_name = self.get_base_name(image_name)
            t = self.ccd.basename_to_time[base_name]
            # Query the navigation state recorded by the INS for this time.
            pose = self.nav_state_provider.pose(t)
            ins_poses.append(pose)
            x, y, z = pose[0]
            points.append([x, y, z])
        points = np.array(points)
        locations_txt = os.path.join(output_dir, "image_locations.txt")
        self.write_image_locations(locations_txt, img_fnames, ins_poses)
        print(
            f"Aligning model given {len(img_fnames)} images and their ENU coordinates."
        )
        ransac_options = pc.RANSACOptions(
            max_error=4.0,  # for example, the reprojection error in pixels
            min_inlier_ratio=0.01,
            confidence=0.9999,
            min_num_trials=1000,
            max_num_trials=100000,
        )
        min_common_observations = 5
        tform = pc.align_reconstruction_to_locations(
            self.R, img_fnames, points, min_common_observations, ransac_options
        )
        print("Transformation after alignment: ")
        print(tform.scale, tform.rotation, tform.translation)
        # Apply transform to self
        self.R.transform(tform)
        print(f"Saving aligned model to {output_dir}...")
        self.R.write(output_dir)

    def read_image_locations(
        self, fname: str | os.PathLike
    ) -> Tuple[List[str], np.ndarray]:
        with open(fname, "r") as f:
            lines = f.readlines()
        points = []
        image_fnames = []
        for line in lines:
            # Pull x,y,z out in ENU
            try:
                img_fname, x, y, z = line.split(" ")
            except Exception as e:
                print(e)
                print("Skipping location line.")
                continue
            image_fnames.append(img_fname.strip())
            points.append((float(x), float(y), float(z)))
        points = np.array(points)
        return image_fnames, points

    def write_image_locations(
        self,
        locations_fname: str | os.PathLike,
        img_fnames: List[str],
        ins_poses: List[np.ndarray],
    ) -> None:
        print(f"Writing image locations to {locations_fname}")
        img_fnames = sorted(img_fnames)
        with open(locations_fname, "w") as fo:
            for i in range(len(img_fnames)):
                name = img_fnames[i]
                pos = ins_poses[i][0]
                fo.write("%s %0.8f %0.8f %0.8f\n" % (name, pos[0], pos[1], pos[2]))


def find_best_sparse_model(sparse_dir: str | os.PathLike) -> str:
    all_models = sorted(os.listdir(sparse_dir)) if os.path.isdir(sparse_dir) else []
    print(f"All Models: {all_models}")
    best_model = ""
    most_images_aligned = 0
    for subdir in all_models:
        dir = os.path.join(sparse_dir, subdir)
        try:
            R = pc.Reconstruction(dir)
        except Exception as e:
            print(e)
            continue
        num_images = len(R.images.keys())
        print(f"Number of images in {dir}: {num_images}")
        if num_images > most_images_aligned:
            most_images_aligned = num_images
            best_model = dir
    if not best_model:
        raise SystemError(
            f"No valid sparse models found in {sparse_dir}, please verify "
            "your model built correctly."
        )
    print(f"Selecting {best_model} as the best model.")
    return best_model

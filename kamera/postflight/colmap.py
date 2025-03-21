import os
import json
import pathlib
import cv2
import PIL.Image
import numpy as np
import os.path as osp
import ubelt as ub
from posixpath import basename
from re import I
from rich import print

from shapely import points
import pycolmap as pc
from dataclasses import dataclass
from typing import Tuple, List, Dict
from kamera.sensor_models.nav_state import NavStateINSJson
from kamera.colmap_processing.camera_models import StandardCamera
from kamera.colmap_processing.image_renderer import render_view
from kamera.postflight.alignment import (
    VisiblePoint,
    iterative_alignment,
    manual_alignment,
    transfer_alignment,
)


@dataclass(frozen=True)
class ColmapCalibrationData:
    sfm_poses: List[np.ndarray]
    ins_poses: List[np.ndarray]
    img_fnames: List[np.ndarray]
    img_times: List[np.ndarray]
    llhs: List[np.ndarray]
    points_per_image: List[List[VisiblePoint]]
    basename_to_time: Dict[str, float]
    fname_to_time_channel_modality: Dict[float, Dict[str, str]]
    best_cameras: Dict[str, pc._core.Camera]


class ColmapCalibration(object):
    def __init__(
        self, flight_dir: str | os.PathLike, colmap_dir: str | os.PathLike
    ) -> None:
        self.flight_dir = flight_dir
        self.colmap_dir = colmap_dir
        # contains the images, 3D points, and cameras of the colmap database
        self.R = pc.Reconstruction(self.colmap_dir)
        self.ccd = ColmapCalibrationData
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
        # get base
        base = osp.basename(fname)
        # get it without an extension and modality
        modality_agnostic = "_".join(base.split("_")[:-1])
        return modality_agnostic

    def get_modality(self, fname: str | os.PathLike) -> str:
        """Extract image modality (e.g. ir/meta) from a given kamera filename."""
        base = osp.basename(fname)
        modality = base.split("_")[-1].split(".")[0]
        return modality

    def get_channel(self, fname: str | os.PathLike) -> str:
        """Extract channel (e.g. L/R/C) from a given kamera filename."""
        base = osp.basename(fname)
        channel = base.split("_")[3]
        return channel

    def get_view(self, fname: str | os.PathLike) -> str:
        """Extract view (e.g. left_view, right_view, center_view) from
        a given kamera filename"""
        base = osp.basename(fname)
        channel = base.split("_")[3]
        view = "null"
        if channel == "C":
            view = "center_view"
        elif channel == "L":
            view = "left_view"
        elif channel == "R":
            view = "right_view"
        return view

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
            except (OSError, IOError):
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
            base_name = self.get_base_name(image.name)
            try:
                t = basename_to_time[base_name]
            except KeyError:
                print(
                    "Couldn't find a _meta.json file associated with '%s'" % base_name
                )
                continue

            # Query the navigation state recorded by the INS for this time.
            pose = self.nav_state_provider.pose(t)
            llh = self.nav_state_provider.llh(t)

            # Query Colmaps pose for the camera.
            R = image.cam_from_world.rotation.matrix()
            pos = -np.dot(R.T, image.cam_from_world.translation)

            image.cam_from_world.rotation.normalize()
            quat = image.cam_from_world.rotation.quat
            # invert the w (rotation) component,
            #  so we get the camera to world rotation
            quat[3] *= -1

            sfm_pose = [pos, quat]

            img_times.append(t)
            ins_poses.append(pose)
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

    def calibrate_camera(self, camera_name: str) -> tuple[StandardCamera, float]:
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
    ) -> tuple[StandardCamera, float]:
        """
        Use the quaternion already solved of a colocated, calibrated camera
        to bootstrap the calibration process.
        """
        # Find all valid indices of current cameras
        channel, modality = camera_name.split("_")[2:]
        cam_idxs = [
            1 if camera_name == os.path.basename(os.path.dirname(im)) else 0
            for im in self.ccd.img_fnames
        ]
        print(f"Number of images in camera {camera_name}.")
        print(np.sum(cam_idxs))
        observations = [
            pts for i, pts in enumerate(self.ccd.points_per_image) if cam_idxs[i] == 1
        ]

        # Now we have to find the overlapping observations on a per-frame basis between
        # the camera to calibrate, and the colocated, calibrated, camera.
        colocated_observations = []
        observations = []
        for i, fname in enumerate(self.ccd.img_fnames):
            if cam_idxs[i] == 1:
                # Replace with the modality of the already calibrated modality
                calibrated_fname = fname.replace(modality, calibrated_modality)
                ii = self.ccd.img_fnames.index(calibrated_fname)
                colocated_observations.append(self.ccd.points_per_image[ii])
                observations.append(self.ccd.points_per_image[i])
        if len(observations) < 10 or len(colocated_observations) < 10:
            print(
                f"Only {len(observations)} seen for camera {camera_name}, and "
                f" only {len(colocated_observations)} found for the calibrated camera, "
                " skipping."
            )
            return

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
    ) -> Dict[float, Dict[str, str]]:
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

    def write_gifs(
        self,
        gif_dir: str | os.PathLike,
        camera_name: str,
        colocated_modality: str,
        camera_model: StandardCamera,
        colocated_camera_model: StandardCamera,
        num_gifs: int = 5,
    ) -> None:
        channel, modality = camera_name.split("_")[2:]
        colocated_camera_name = camera_name.replace(modality, colocated_modality)
        print(
            f"Writing registration gifs for cameras {camera_name} "
            f"and {colocated_camera_name}."
        )
        ub.ensuredir(gif_dir)

        img_fnames = self.ccd.img_fnames
        for k in range(num_gifs):
            inds = list(range(len(img_fnames)))
            np.random.shuffle(inds)
            for i in range(len(img_fnames)):
                colocated_img = img = None
                fname = img_fnames[inds[i]]
                if osp.basename(osp.dirname(fname)) != camera_name:
                    continue
                view = self.get_view(fname)
                try:
                    bname = osp.basename(fname)
                    abs_fname = osp.join(self.flight_dir, view, bname)
                    img = cv2.imread(abs_fname, cv2.IMREAD_COLOR)[:, :, ::-1]
                except Exception as e:
                    print(f"No {modality} image found at path {abs_fname}")
                    continue
                try:
                    # get colocated image from the same time
                    abs_colocated_bname = pathlib.Path(
                        abs_fname.replace(modality, colocated_modality)
                    )
                    abs_colocated_fname = None
                    if not abs_colocated_bname.is_file():
                        # can't find file, check other extensions
                        for ext in [".jpg", ".jpeg", ".png", ".tiff"]:
                            if abs_colocated_bname.with_suffix(ext).is_file():
                                abs_colocated_fname = str(
                                    abs_colocated_bname.with_suffix(ext)
                                )
                                break
                    else:
                        abs_colocated_fname = str(abs_colocated_bname)
                    colocated_bname = osp.basename(abs_colocated_fname)
                    colocated_img = cv2.imread(abs_colocated_fname, cv2.IMREAD_COLOR)[
                        :, :, ::-1
                    ]
                    # Once we find a matching pair, break
                    break
                except Exception as e:
                    print(
                        f"No {colocated_modality} image found at filepath {abs_colocated_fname}"
                    )
                    continue

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
            # ds_warped_rgb_img = PIL.Image.fromarray(
            #    cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(warped_rgb_img)))
            # )
            # ds_uv_img = PIL.Image.fromarray(
            #    cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(colocated_img)))
            # )
            # Make sure gifs are reasonable size
            w, h, _ = img.shape
            ratio = w / h
            new_w = 1280
            new_h = int(new_w * ratio)
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
        img_fnames = [im.name for im in self.R.images().values()]
        points = []
        ins_poses = []
        for image_name in img_fnames:
            base_name = self.get_base_name(image_name)
            t = self.basename_to_time[base_name]
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


def find_best_sparse_model(sparse_dir: str | os.PathLike):
    if os.listdir(sparse_dir) == 0:
        raise SystemError(
            f"Sparse directory given ({sparse_dir}) is empty, please verify your model built correctly."
        )
    all_models = sorted(os.listdir(sparse_dir))
    print(f"All Models: {all_models}")
    best_model = ""
    most_images_aligned = 0
    for subdir in all_models:
        R = pc.Reconstruction(subdir)
        num_images = len(R.images.keys())
        print(f"Number of images in {subdir}: {num_images}")
        if num_images > most_images_aligned:
            best_model = subdir
    print(f"Selecting {subdir} as the best model.")
    return best_model


def main():
    flight_dir = "/home/local/KHQ/adam.romlein/noaa/data/2024_AOC_AK_Calibration/fl09"
    colmap_dir = (
        "/home/local/KHQ/adam.romlein/noaa/data/2024_AOC_AK_Calibration/fl09/colmap_ir"
    )
    cc = ColmapCalibration(flight_dir, colmap_dir)
    cc.calibrate_ir()


if __name__ == "__main__":
    main()

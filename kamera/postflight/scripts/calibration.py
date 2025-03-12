import os
import time
import ubelt as ub

from kamera.colmap_processing.camera_models import load_from_file
from kamera.postflight.colmap import ColmapCalibration, find_best_sparse_model


def main():
    # REQUIREMENTS: Must already have built a reasonable 3-D model using Colmap
    # And have a flight directory populated by the kamera system
    flight_dir = "/home/local/KHQ/adam.romlein/noaa/data/2024_AOC_AK_Calibration/fl09"
    colmap_dir = os.path.join(flight_dir, "colmap")
    # Location to save KAMERA camera models.
    save_dir = os.path.join(flight_dir, "kamera_models")
    # Location to find / build aligned model
    align_dir = os.path.join(colmap_dir, "aligned")
    # This assumes that 2 modalities of cameras are within the 3-D model
    joint_calibration = True
    # Can switch to UV / IR if possible that they are better aligned
    main_modality = "rgb"
    # If camera models are already present, if true this will overwrite
    force_calibrate = False
    # Whether to calibrate IR models or not. A "colmap_ir" must be built
    # in the same folder
    calibrate_ir = True

    # Step 1: Make sure the model is aligned to the INS readings, if not,
    # find the best model and align it.
    if os.path.exists(align_dir) and len(os.listdir(align_dir)) == 1:
        print(
            f"The directory {align_dir} exists and is not empty,"
            " assuming model is aligned."
        )
        aligned_sparse_dir = os.path.join(align_dir, os.listdir(align_dir)[0])
        cc = ColmapCalibration(flight_dir, aligned_sparse_dir)
    elif os.path.exists(align_dir) and len(os.listdir(align_dir) > 1):
        raise SystemError(
            f"{len(os.listdir(align_dir))} aligned models are present, only 1 should exist in {align_dir}."
        )
    elif not os.path.exists(align_dir) or len(os.listdir(align_dir) == 0):
        print(
            f"The directory {align_dir} does not exist, or is empty, selecting and aligning the best sparse model."
        )
        # location of sparsely constructed colmap models (always sparse if using default colmap options)
        sparse_dir = os.path.join(colmap_dir, "sparse")
        # Step 1: Find the "best" sparse model (most images aligned)
        best_reconstruction_dir = find_best_sparse_model(sparse_dir)
        sparse_idx = os.path.split(best_reconstruction_dir)[-1]

        # Step 2: Align and save this model
        aligned_sparse_dir = os.path.join(align_dir, sparse_idx)
        ub.ensuredir(aligned_sparse_dir)
        cc = ColmapCalibration(flight_dir, best_reconstruction_dir)
        cc.align_model(aligned_sparse_dir)

    # We now have an aligned 3-D model, so we can calibrate the cameras
    print("Preparing calibration data...")
    cc.prepare_calibration_data()
    ub.ensuredir(save_dir)
    cameras = ub.AutoDict()
    camera_strs = list(cc.ccd.best_cameras.keys())
    modalities = [cs.split("_")[3] for cs in camera_strs]
    if len(modalities) < 1:
        print(
            "Warning: only 1 modality found in this 3D model, not joinly calibrating."
        )
        joint_calibration = False
    for camera_str in cc.ccd.best_cameras.keys():
        channel, modality = camera_str.split("_")[2:]
        # skip all modalities except the "main" ones
        if joint_calibration and modality != main_modality:
            continue
        out_path = os.path.join(save_dir, f"{camera_str}_v2.yaml")
        # skip compute if we don't have to recompute
        if os.path.exists(out_path) and not force_calibrate:
            print(
                f"Camera model path {out_path} exists and force_calibrate is turned off, loading model."
            )
            camera_model = load_from_file(out_path)
            cameras[channel][modality] = camera_model
        else:
            print(f"Calibrating camera {camera_str}.")
            tic = time.time()
            camera_model = cc.calibrate_camera(camera_str, error_threshold=100)
            toc = time.time()
            if camera_model is not None:
                print(f"Saving camera model to {out_path}")
                camera_model.save_to_file(out_path)
                print(f"Time to calibrate camera {camera_str} was {(toc - tic):.3f}s")
                print(camera_model)
                cameras[channel][modality] = camera_model
            else:
                print(f"Calibrating camera {camera_str} failed.")

    # If we have multiple cameras within one model, we can utilize the fact that these cameras
    # are colocated to use 3D points obtained from one to project into the other
    if joint_calibration:
        for camera_str in cc.ccd.best_cameras.keys():
            channel, modality = camera_str.split("_")[2:]
            # now calibrate all other modalities
            if modality == main_modality:
                continue
            print(f"Calibrating camera {camera_str}.")
            tic = time.perf_counter()
            camera_model = cc.transfer_calibration(
                camera_str,
                cameras[channel][main_modality],
                calibrated_modality=main_modality,
                error_threshold=100,
            )
            toc = time.perf_counter()
            if camera_model is not None:
                out_path = os.path.join(save_dir, f"{camera_str}_v2.yaml")
                print(f"Saving camera model to {out_path}")
                camera_model.save_to_file(out_path)
                print(
                    f"Time to transfer camera calibration to {camera_str} was {(toc - tic):.3f}s"
                )
                print(camera_model)
                cameras[channel][modality] = camera_model
            else:
                print(f"Transferring calibration to camera {camera_str} failed.")

    # IR calibration generally happens in a separate model, since SIFT has a hard
    # time matching between EO and long-wave IR
    if calibrate_ir:
        colmap_dir = os.path.join(flight_dir, "colmap")

    aircraft, angle = camera_str.split("_")[0:2]
    gif_dir = os.path.join(save_dir, "registration_gifs_v2")
    ub.ensuredir(gif_dir)
    for channel in cameras.keys():
        rgb_model = cameras[channel]["rgb"]
        uv_model = cameras[channel]["uv"]
        rgb_name = "_".join([aircraft, angle, channel, "rgb"])
        uv_name = "_".join([aircraft, angle, channel, "uv"])
        cc.write_gifs(gif_dir, colmap_dir, rgb_name, uv_name, rgb_model, uv_model)


if __name__ == "__main__":
    main()

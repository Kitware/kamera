import os
import time
import ubelt as ub

from kamera.postflight.colmap import ColmapCalibration, find_best_sparse_model


def main():
    # REQUIREMENTS: Must already have built a reasonable 3-D model using Colmap
    # And have a flight directory populated by the kamera system
    flight_dir = "<your_dir>"
    colmap_dir = os.path.join(flight_dir, "colmap")
    # Location to save KAMERA camera models.
    save_dir = os.path.join(flight_dir, "kamera_models")
    # Location to find / build aligned model
    align_dir = os.path.join(colmap_dir, "aligned")

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
    for camera_str in cc.ccd.best_cameras.keys():
        channel, modality = camera_str.split("_")[2:]
        # if modality != "rgb":
        #    continue
        print(f"Calibrating camera {camera_str}.")
        tic = time.time()
        camera_model = cc.calibrate_camera(camera_str, error_threshold=100)
        toc = time.time()
        if camera_model is not None:
            out_path = os.path.join(save_dir, f"{camera_str}_v2.yaml")
            print(f"Saving camera model to {out_path}")
            camera_model.save_to_file(out_path)
            print(f"Time to calibrate camera {camera_str} was {(toc - tic):.3f}s")
            print(camera_model)
            cameras[channel][modality] = camera_model
        else:
            print(f"Calibrating camera {camera_str} failed.")

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

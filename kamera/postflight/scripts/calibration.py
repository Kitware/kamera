import os
import time
import json
import pathlib
import ubelt as ub
from rich import print

from kamera.colmap_processing.camera_models import load_from_file
from kamera.postflight.colmap import ColmapCalibration, find_best_sparse_model


def align_model(
    flight_dir: os.PathLike | str,
    colmap_dir: os.PathLike | str,
    align_dir: os.PathLike | str,
) -> ColmapCalibration:
    if os.path.exists(align_dir) and len(os.listdir(align_dir)) == 1:
        print(
            f"The directory {align_dir} exists and is not empty,"
            " assuming model is aligned."
        )
        aligned_sparse_dir = os.path.join(align_dir, os.listdir(align_dir)[0])
        cc = ColmapCalibration(flight_dir, aligned_sparse_dir, colmap_dir)
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
        cc = ColmapCalibration(flight_dir, best_reconstruction_dir, colmap_dir)
        cc.prepare_calibration_data()
        cc.align_model(aligned_sparse_dir)
    return cc


def main():
    # REQUIREMENTS: Must already have built a reasonable 3-D model using Colmap
    # And have a flight directory populated by the kamera system
    flight_dir = "/home/local/KHQ/adam.romlein/noaa/data/052025_Calibration/"
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
    # open the point clicking UI to refine IR calibration?
    refine_ir = False

    # Step 1: Make sure the model is aligned to the INS readings, if not,
    # find the best model and align it.
    print("[blue] Loading 3D RGB/UV Model.[/blue]")
    cc = align_model(flight_dir, colmap_dir, align_dir)

    # We now have an aligned 3-D model, so we can calibrate the cameras
    print("Preparing calibration data...")
    cc.prepare_calibration_data()
    ub.ensuredir(save_dir)
    cameras = ub.AutoDict()
    errors = ub.AutoDict()
    camera_strs = list(cc.ccd.best_cameras.keys())
    modalities = [cs.split("_")[3] for cs in camera_strs]
    num_cameras_loaded = 0
    num_cameras_calibrated = 0
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
            num_cameras_loaded += 1
        else:
            print(f"Calibrating camera {camera_str}.")
            tic = time.time()
            camera_model, error = cc.calibrate_camera(camera_str)
            toc = time.time()
            if camera_model is not None:
                print(f"Saving camera model to {out_path}")
                camera_model.save_to_file(out_path)
                print(f"Time to calibrate camera {camera_str} was {(toc - tic):.3f}s")
                print(camera_model)
                cameras[channel][modality] = camera_model
                errors[channel][modality] = error
            else:
                print(f"Calibrating camera {camera_str} failed.")
            num_cameras_calibrated += 1

    # If we have multiple cameras within one model, we can utilize the fact that these cameras
    # are colocated to use 3D points obtained from one to project into the other
    if joint_calibration:
        for camera_str in cc.ccd.best_cameras.keys():
            channel, modality = camera_str.split("_")[2:]
            # now calibrate all other modalities
            if modality == main_modality:
                continue
            out_path = os.path.join(save_dir, f"{camera_str}_v2.yaml")
            if os.path.exists(out_path) and not force_calibrate:
                print(
                    f"Camera model path {out_path} exists and force_calibrate is turned off, loading model."
                )
                camera_model = load_from_file(out_path)
                cameras[channel][modality] = camera_model
                num_cameras_loaded += 1
            else:
                print(f"Calibrating camera {camera_str}.")
                tic = time.perf_counter()
                camera_model, error = cc.transfer_calibration(
                    camera_str,
                    cameras[channel][main_modality],
                    calibrated_modality=main_modality,
                )
                toc = time.perf_counter()
                if camera_model is not None:
                    print(f"Saving camera model to {out_path}")
                    camera_model.save_to_file(out_path)
                    print(
                        f"Time to transfer camera calibration to {camera_str} was {(toc - tic):.3f}s"
                    )
                    print(camera_model)
                    cameras[channel][modality] = camera_model
                    errors[channel][modality] = error
                    num_cameras_calibrated += 1
                else:
                    print(f"Transferring calibration to camera {camera_str} failed.")

    # IR calibration generally happens in a separate model, since SIFT has a hard
    # time matching between EO and long-wave IR
    ir_cameras_refined = 0
    if calibrate_ir:
        ir_colmap_dir = os.path.join(flight_dir, "colmap_ir")
        ir_align_dir = os.path.join(ir_colmap_dir, "aligned")
        print("[blue] Loading 3D IR Model.[/blue]")
        ir_cc = align_model(flight_dir, ir_colmap_dir, ir_align_dir)
        print("Preparing IR calibration data...")
        ir_cc.prepare_calibration_data()
        camera_strs = list(ir_cc.ccd.best_cameras.keys())
        for camera_str in ir_cc.ccd.best_cameras.keys():
            channel, modality = camera_str.split("_")[2:]
            out_path = os.path.join(save_dir, f"{camera_str}_v2.yaml")
            if os.path.exists(out_path) and not force_calibrate:
                print(
                    f"Camera model path {out_path} exists and force_calibrate is turned off, loading model."
                )
                camera_model = load_from_file(out_path)
                cameras[channel][modality] = camera_model
                num_cameras_loaded += 1
            else:
                print(f"Calibrating camera {camera_str}.")
                tic = time.perf_counter()
                camera_model, error = ir_cc.calibrate_camera(
                    camera_str, error_threshold=100
                )
                toc = time.perf_counter()
                if camera_model is not None:
                    print(f"Saving camera model to {out_path}")
                    camera_model.save_to_file(out_path)
                    print(
                        f"Time to calibrate camera {camera_str} was {(toc - tic):.3f}s"
                    )
                    print(camera_model)
                    cameras[channel][modality] = camera_model
                    errors[channel][modality] = error
                    num_cameras_calibrated += 1
                else:
                    print(f"Calibrating camera {camera_str} failed.")

            if refine_ir:
                # need to define a points per modality/view pair
                ir_points_json = pathlib.Path(
                    os.path.join(
                        flight_dir,
                        "manual_keypoints",
                        f"{channel}_ir_to_rgb_points.json",
                    )
                )
                if not ir_points_json.is_file():
                    # check to see if there are numpy points saved
                    ir_points_txt = ir_points_json.with_suffix(".txt")
                    if ir_points_txt.is_file():
                        with ir_points_txt.open() as f:
                            fpoints = f.readlines()
                        points = {"rightPoints": [], "leftPoints": []}
                        for line in fpoints:
                            vals = list(map(float, line.split(" ")))
                            points["rightPoints"].append(vals[:2])
                            points["leftPoints"].append(vals[2:4])
                    else:
                        print(
                            "[yellow]"
                            f"No matching points file was found! Please place your points file"
                            f" in [bold]{ir_points_txt}[/bold] if in text format, or [bold]{ir_points_json}[/bold] if in json format."
                            "[yellow]"
                        )
                        continue
                else:
                    with open(ir_points_json, "r") as f:
                        points = json.load(f)
                refined_camera_model, error = ir_cc.manual_calibration(
                    camera_model, cameras[channel][main_modality], points
                )
                cameras[channel][modality] = refined_camera_model
                errors[channel][modality] = error
                ir_cameras_refined += 1

    aircraft, angle = camera_str.split("_")[0:2]
    gif_dir = os.path.join(save_dir, "registration_gifs_v2")
    ub.ensuredir(gif_dir)

    # Write UV GIFs
    for channel in cameras.keys():
        rgb_model = cameras[channel]["rgb"]
        uv_model = cameras[channel]["uv"]
        uv_name = "_".join([aircraft, angle, channel, "uv"])
        cc.write_gifs(
            gif_dir,
            camera_name=uv_name,
            colocated_modality="rgb",
            camera_model=uv_model,
            colocated_camera_model=rgb_model,
            num_gifs=5,
        )

    # Write IR GIFs
    if calibrate_ir:
        for channel in cameras.keys():
            rgb_model = cameras[channel]["rgb"]
            ir_model = cameras[channel]["ir"]
            ir_name = "_".join([aircraft, angle, channel, "ir"])
            ir_cc.write_gifs(
                gif_dir,
                camera_name=ir_name,
                colocated_modality="rgb",
                camera_model=ir_model,
                colocated_camera_model=rgb_model,
                num_gifs=5,
            )

    print("=" * 80)
    print("[bold] Quick Summary: ")
    print(f"Number of cameras calibrated: {num_cameras_calibrated}.")
    print(f"Number of cameras loaded from disk: {num_cameras_loaded}.")
    print(f"Number of IR cameras refined {ir_cameras_refined}.")
    print("Errors: ")
    print(errors)


if __name__ == "__main__":
    main()

"""Command-line configuration for the rig calibration."""

from __future__ import annotations

import scriptconfig as scfg


class CalibrateConfig(scfg.DataConfig):
    __command__ = "kamera-calibrate"
    flight_dir = scfg.Value(
        None,
        position=1,
        help="KAMERA flight directory (contains <sys_cfg>/<view>_view/*_meta.json)",
    )
    work_dir = scfg.Value(
        None, help="Scratch and output root (default: <flight_dir>/calibration)"
    )
    rig_name = scfg.Value(
        None,
        help="Name used in output file names (default: sys_cfg from the meta json)",
    )
    reference_camera = scfg.Value(
        "C_rgb",
        help="Rig reference sensor; every other camera is expressed relative to it",
    )
    frame_start = scfg.Value(0, help="Index of the first synchronized frame to use")
    frame_stride = scfg.Value(1, help="Use every Nth frame")
    max_frames = scfg.Value(0, help="Cap on frames used (0 = all)")
    focal_px = scfg.Value(
        {"rgb": 31363.0, "uv": 14120.0, "ir": 1712.0},
        help="Initial focal length per modality in pixels; refined by SfM",
    )
    max_image_size = scfg.Value(
        3200, help="Images are downsampled to this longest side for SIFT"
    )
    num_features = scfg.Value(8192, help="Max SIFT features per image")
    match_distance_m = scfg.Value(
        250.0, help="Spatial matching radius from INS positions"
    )
    match_neighbors = scfg.Value(
        90,
        help="Spatial matching neighbours per image "
        "(about 10 frames times the number of cameras)",
    )
    prior_std_m = scfg.Value(
        2.0, help="Standard deviation assigned to INS position priors"
    )
    registration_range_m = scfg.Value(
        0.0,
        help="Ground range the homographies are exact at; 0 = median scene range of "
        "the calibration model. Set to the survey AGL",
    )
    gif_frames = scfg.Value(5, help="Registration GIFs written per camera pair")
    force = scfg.Value(
        False, isflag=True, help="Rerun stages whose outputs already exist"
    )

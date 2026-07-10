import json

import numpy as np

from kamera.postflight import registration_homography as rh


def test_write_registration_homographies_per_camera(tmp_path, monkeypatch):
    h = np.diag([1.0, 2.0, 1.0])
    monkeypatch.setattr(rh, "pixel_homography", lambda src, dst: h)

    # models are only looked up, never called, thanks to the stub
    models = {
        "21deg_N56RF_center_rgb": "center-rgb",
        "21deg_N56RF_center_uv": "center-uv",
        "21deg_N56RF_center_ir": "center-ir",
        "25deg_N56RF_left_rgb": "left-rgb",
        "25deg_N56RF_left_uv": "left-uv",
        "30deg_N56RF_right_uv": "right-uv",  # no rgb reference -> skipped
    }
    written = rh.write_registration_homographies(models, str(tmp_path))

    assert sorted(written) == [
        str(tmp_path / name)
        for name in [
            "21deg_N56RF_center_ir_to_21deg_N56RF_center_rgb_registration.json",
            "21deg_N56RF_center_uv_to_21deg_N56RF_center_rgb_registration.json",
            "25deg_N56RF_left_uv_to_25deg_N56RF_left_rgb_registration.json",
        ]
    ]

    with open(written[0]) as f:
        d = json.load(f)
    assert d["camera"] == "21deg_N56RF_center_ir"
    assert d["reference_camera"] == "21deg_N56RF_center_rgb"
    assert np.allclose(d["camera_to_reference"], h)
    assert np.allclose(d["reference_to_camera"], np.linalg.inv(h))

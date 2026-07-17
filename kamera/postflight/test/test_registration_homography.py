import json
from types import SimpleNamespace

import numpy as np

from kamera.postflight import registration_homography as rh


def test_write_registration_homographies_per_station(tmp_path, monkeypatch):
    h = np.diag([1.0, 2.0, 1.0])
    monkeypatch.setattr(rh, "pixel_homography", lambda src, dst: h)

    cam = SimpleNamespace(width=64, height=32)
    models = {
        "21deg_N56RF_center_rgb": cam,
        "21deg_N56RF_center_uv": cam,
        "21deg_N56RF_center_ir": cam,
        "25deg_N56RF_left_rgb": cam,
        "25deg_N56RF_left_uv": cam,
        "30deg_N56RF_right_uv": cam,  # no rgb reference -> skipped
    }
    written = rh.write_registration_homographies(
        models, str(tmp_path), flight="fl07"
    )

    assert sorted(written) == [
        str(tmp_path / name)
        for name in [
            "21deg_N56RF_center_registration.json",
            "25deg_N56RF_left_registration.json",
        ]
    ]

    with open(tmp_path / "21deg_N56RF_center_registration.json") as f:
        d = json.load(f)
    assert d["type"] == "dive-camera-registration"
    assert d["version"] == 1
    assert d["source"] == {"model": "kamera-v3", "flight": "fl07"}
    assert [(p["left"], p["right"]) for p in d["pairs"]] == [
        ("eo", "ir"),
        ("eo", "uv"),
    ]

    pair = d["pairs"][0]
    assert pair["transformType"] == "homography"
    # rightToLeft is the camera->reference homography; leftToRight its
    # inverse, normalized so [2][2] == 1.
    assert np.allclose(pair["rightToLeft"], h)
    assert np.allclose(pair["leftToRight"], np.linalg.inv(h))
    assert np.array(pair["leftToRight"])[2, 2] == 1.0

    # points are [x_left, y_left, x_right, y_right] correspondences that
    # satisfy the homography exactly, gridded over the right image.
    pts = np.array(pair["points"])
    assert pts.shape == (9, 4)
    assert pts[:, 2].min() == 0 and pts[:, 2].max() == cam.width - 1
    assert pts[:, 3].min() == 0 and pts[:, 3].max() == cam.height - 1
    right_h = np.vstack([pts[:, 2], pts[:, 3], np.ones(len(pts))])
    left = h @ right_h
    assert np.allclose(pts[:, :2], (left[:2] / left[2]).T)

    with open(tmp_path / "25deg_N56RF_left_registration.json") as f:
        d = json.load(f)
    assert [(p["left"], p["right"]) for p in d["pairs"]] == [("eo", "uv")]

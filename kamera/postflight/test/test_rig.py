import json
from datetime import datetime, timezone

import numpy as np
import pycolmap
import pytest

from kamera.postflight.rig import filter_same_trigger_matches, write_pose_priors

BASE = "test_fl09_{ch}_20200830_{time}_{mod}.jpg"


def _make_db(path):
    """Two triggers seen by two co-located cameras: the same-trigger
    pairs must be emptied, the cross-trigger pair kept."""
    db = pycolmap.Database.open(str(path))
    cam = pycolmap.Camera()
    cam.model = pycolmap.CameraModelId.OPENCV
    cam.width, cam.height = 100, 100
    cam.params = [100, 100, 50, 50, 0, 0, 0, 0]
    cam_id = db.write_camera(cam)

    names = [
        ("rig_C_rgb", BASE.format(ch="C", time="020748.058907", mod="rgb")),
        ("rig_C_uv", BASE.format(ch="C", time="020748.058907", mod="uv")),
        ("rig_C_rgb", BASE.format(ch="C", time="020750.058907", mod="rgb")),
        ("rig_C_uv", BASE.format(ch="C", time="020750.058907", mod="uv")),
    ]
    ids = []
    for folder, base in names:
        im = pycolmap.Image()
        im.name = f"{folder}/{base}"
        im.camera_id = cam_id
        ids.append(db.write_image(im))
        db.write_keypoints(ids[-1], np.random.rand(20, 2).astype(np.float32) * 90)

    matches = np.stack([np.arange(20)] * 2, 1).astype(np.uint32)
    pairs = [
        (ids[0], ids[1]),  # same trigger (rgb-uv)
        (ids[2], ids[3]),  # same trigger (rgb-uv)
        (ids[0], ids[2]),  # same camera, consecutive triggers -- keep
    ]
    for id1, id2 in pairs:
        db.write_matches(id1, id2, matches)
        geom = pycolmap.TwoViewGeometry()
        geom.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
        geom.inlier_matches = matches
        db.write_two_view_geometry(id1, id2, geom)
    db.close()
    return ids


def test_filter_same_trigger_matches(tmp_path):
    db_path = tmp_path / "database.db"
    ids = _make_db(db_path)

    assert filter_same_trigger_matches(db_path) == 2

    db = pycolmap.Database.open(str(db_path))
    try:
        # same-trigger pairs are emptied but still present, so a matcher
        # re-run skips them instead of recreating them
        assert db.exists_matches(ids[0], ids[1])
        assert len(db.read_matches(ids[0], ids[1])) == 0
        assert len(db.read_two_view_geometry(ids[0], ids[1]).inlier_matches) == 0
        # the along-track pair is untouched
        assert len(db.read_matches(ids[0], ids[2])) == 20
        assert len(db.read_two_view_geometry(ids[0], ids[2]).inlier_matches) == 20
    finally:
        db.close()

    # idempotent: nothing left to empty
    assert filter_same_trigger_matches(db_path) == 0


class _FakeNav:
    """pose(t) encodes t in the east coordinate so tests can check which
    exposure time each prior was interpolated at."""

    def pose(self, t):
        return [np.array([t, 0.0, 100.0]), np.array([0.0, 0.0, 0.0, 1.0])]


def test_write_pose_priors_falls_back_to_filename_time(tmp_path):
    db_path = tmp_path / "database.db"
    ids = _make_db(db_path)

    # meta.json only for the first trigger; the second must fall back to
    # the filename's <date>_<time> (a missing prior aborts COLMAP's
    # spatial matcher, so every image needs one)
    meta = "test_fl09_C_20200830_020748.058907_meta.json"
    (tmp_path / meta).write_text(json.dumps({"evt": {"time": 123.0}}))

    db = pycolmap.Database.open(str(db_path))
    try:
        assert write_pose_priors(db, tmp_path, _FakeNav()) == 4
        by_image = {p.corr_data_id.id: p for p in db.read_all_pose_priors()}
        t_meta = by_image[ids[0]].position[0]
        t_name = by_image[ids[2]].position[0]
    finally:
        db.close()

    assert t_meta == 123.0
    expected = (
        datetime.strptime("20200830_020750.058907", "%Y%m%d_%H%M%S.%f")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    assert t_name == expected


def test_write_pose_priors_rejects_non_finite_position(tmp_path):
    db_path = tmp_path / "database.db"
    _make_db(db_path)

    class _NanNav:
        def pose(self, t):
            return [np.array([np.nan, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])]

    db = pycolmap.Database.open(str(db_path))
    try:
        with pytest.raises(ValueError, match="Non-finite"):
            write_pose_priors(db, tmp_path, _NanNav())
    finally:
        db.close()

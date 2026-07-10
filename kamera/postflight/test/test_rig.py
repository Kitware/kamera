import numpy as np
import pycolmap

from kamera.postflight.rig import filter_same_trigger_matches

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

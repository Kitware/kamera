import json
import os

import cv2
import numpy as np
import pytest

from kamera.postflight.flight_prep import (
    find_raw_dir,
    modality_group,
    stage_flight,
    stretch_to_uint8,
)


def test_modality_group():
    assert modality_group("rgb") == "rgb"
    assert modality_group("uv") == "rgb"
    assert modality_group("ir") == "ir"


def test_find_raw_dir_nested(tmp_path):
    (tmp_path / "images_21deg" / "center_view").mkdir(parents=True)
    (tmp_path / "colmap_rgb").mkdir()
    assert find_raw_dir(tmp_path) == str(tmp_path / "images_21deg")


def test_find_raw_dir_flight_is_raw(tmp_path):
    (tmp_path / "left_view").mkdir()
    assert find_raw_dir(tmp_path) == str(tmp_path)


def test_find_raw_dir_bare_view_names(tmp_path):
    (tmp_path / "center").mkdir()  # fl09-style raw layout
    assert find_raw_dir(tmp_path) == str(tmp_path)


def test_find_raw_dir_none_or_ambiguous(tmp_path):
    with pytest.raises(SystemError, match="none found"):
        find_raw_dir(tmp_path)
    (tmp_path / "a" / "center_view").mkdir(parents=True)
    (tmp_path / "b" / "center_view").mkdir(parents=True)
    with pytest.raises(SystemError, match="several found"):
        find_raw_dir(tmp_path)


def test_stretch_top_is_count_based():
    # 30 hot pixels far above a dim ramp: with top_px=30 the hot target
    # itself sets the high bound and saturates, where any percentile top
    # would either include it with thousands of background pixels or be
    # set by the background entirely.
    img = np.linspace(0, 1000, 10000).astype(np.uint16).reshape(100, 100)
    img.ravel()[:30] = 60000
    out = stretch_to_uint8(img, low_pct=0.0, top_px=30)
    assert out.dtype == np.uint8
    assert np.count_nonzero(out == 255) == 30
    assert out.min() == 0


def test_stretch_keeps_tiny_warm_target_in_range():
    # cold ice around 27000 counts, a 25-px seal at 29000: the seal must
    # end up near saturation with the background spread over the range,
    # not compressed into a couple of gray levels.
    rng = np.random.default_rng(0)
    img = rng.normal(27000, 50, (512, 640)).astype(np.uint16)
    img[100:105, 200:205] = 29000
    out = stretch_to_uint8(img, low_pct=1.0, top_px=30)
    assert out[102, 202] >= 250
    assert np.std(out[out < 250]) > 5  # background contrast survives


def test_stretch_constant_image_is_safe():
    img = np.full((8, 8), 1234, dtype=np.uint16)
    out = stretch_to_uint8(img)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def _write_raw_trigger(raw_dir, base="test_fl09_C_20200830_020748.058907"):
    view = raw_dir / "center_view"
    view.mkdir(parents=True)
    ir = np.full((32, 32), 27000, dtype=np.uint16)
    ir[4:6, 4:6] = 29000
    cv2.imwrite(str(view / f"{base}_ir.tif"), ir)
    cv2.imwrite(
        str(view / f"{base}_rgb.jpg"), np.zeros((16, 16, 3), dtype=np.uint8)
    )
    (view / f"{base}_meta.json").write_text(json.dumps({"evt": {"time": 1.0}}))
    return base


def test_stage_flight_stretches_ir(tmp_path):
    raw, flight = tmp_path / "raw", tmp_path / "flight"
    base = _write_raw_trigger(raw)
    stage_flight(raw, flight, "p")

    ir_dst = flight / "colmap_ir" / "images0" / "p_center_ir" / f"{base}_ir.tif"
    rgb_dst = flight / "colmap_rgb" / "images0" / "p_center_rgb" / f"{base}_rgb.jpg"
    assert ir_dst.is_file() and not ir_dst.is_symlink()
    assert rgb_dst.is_symlink()
    staged = cv2.imread(str(ir_dst), cv2.IMREAD_UNCHANGED)
    assert staged.dtype == np.uint8
    assert staged.max() == 255  # the warm blob saturates

    # rerun trusts the stretched file instead of rewriting it
    mtime = os.path.getmtime(ir_dst)
    stage_flight(raw, flight, "p")
    assert os.path.getmtime(ir_dst) == mtime


def test_stage_flight_heals_previously_symlinked_ir(tmp_path):
    raw, flight = tmp_path / "raw", tmp_path / "flight"
    base = _write_raw_trigger(raw)
    ir_dst = flight / "colmap_ir" / "images0" / "p_center_ir" / f"{base}_ir.tif"
    ir_dst.parent.mkdir(parents=True)
    ir_dst.symlink_to(raw / "center_view" / f"{base}_ir.tif")

    stage_flight(raw, flight, "p")
    assert ir_dst.is_file() and not ir_dst.is_symlink()
    assert cv2.imread(str(ir_dst), cv2.IMREAD_UNCHANGED).dtype == np.uint8


def test_stage_flight_stretch_disabled(tmp_path):
    raw, flight = tmp_path / "raw", tmp_path / "flight"
    base = _write_raw_trigger(raw)
    stage_flight(raw, flight, "p", stretch_modalities=())
    ir_dst = flight / "colmap_ir" / "images0" / "p_center_ir" / f"{base}_ir.tif"
    assert ir_dst.is_symlink()

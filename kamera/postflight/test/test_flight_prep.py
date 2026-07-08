import numpy as np
import pytest

from kamera.postflight.flight_prep import (
    TriggerFrame,
    find_raw_dir,
    modality_group,
    select_by_spacing,
)


def _line_frames(n, step_m):
    """n frames along +E at a fixed spacing."""
    return [
        TriggerFrame(
            key=str(i),
            time=float(i),
            position=np.array([i * step_m, 0.0, 100.0]),
        )
        for i in range(n)
    ]


def test_spacing_thins_dense_frames():
    frames = _line_frames(100, step_m=10)  # 10 m apart
    kept = select_by_spacing(frames, spacing_m=50)
    # keep first, then every ~50 m -> ~ every 5th
    assert 18 <= len(kept) <= 21
    gaps = np.diff([f.position[0] for f in kept])
    assert gaps.min() >= 50 - 1e-9


def test_spacing_zero_keeps_all():
    frames = _line_frames(30, step_m=10)
    assert len(select_by_spacing(frames, spacing_m=0)) == 30


def test_every_and_max_frames():
    frames = _line_frames(100, step_m=10)
    assert len(select_by_spacing(frames, every=4)) == 25
    assert len(select_by_spacing(frames, max_frames=10)) == 10


def test_spacing_preserves_order_and_endpoints():
    frames = _line_frames(50, step_m=10)
    kept = select_by_spacing(frames, spacing_m=100)
    assert kept[0] is frames[0]
    times = [f.time for f in kept]
    assert times == sorted(times)


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

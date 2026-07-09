import pytest

from kamera.postflight.flight_prep import find_raw_dir, modality_group


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

import pytest

from kamera.postflight.naming import (
    KameraCameraName,
    KameraImageName,
    swap_image_name_modality,
)


def test_parse_meta_json():
    n = KameraImageName.parse(
        "test_seattle_2020_fl09_R_20200830_020748.058907_meta.json"
    )
    assert n.prefix == "test_seattle_2020"
    assert n.flight == "fl09"
    assert n.channel == "R"
    assert n.date == "20200830"
    assert n.time == "020748.058907"
    assert n.modality == "meta"
    assert n.ext == "json"
    assert n.base_name == "test_seattle_2020_fl09_R_20200830_020748.058907"
    assert n.name == "test_seattle_2020_fl09_R_20200830_020748.058907_meta.json"
    assert n.view == "right_view"


def test_parse_image_from_full_path():
    n = KameraImageName.parse(
        "/some/dir/nefsc_2026_narw_fl003_C_20260121_163532.006601_rgb.jpg"
    )
    assert n.prefix == "nefsc_2026_narw"
    assert n.channel == "C"
    assert n.modality == "rgb"
    assert n.ext == "jpg"
    assert n.view == "center_view"


def test_base_name_matches_meta_key():
    # An image and its meta json must share the same base_name key.
    img = KameraImageName.parse(
        "test_seattle_2020_fl09_R_20200830_020748.058907_uv.jpg"
    )
    meta = KameraImageName.parse(
        "test_seattle_2020_fl09_R_20200830_020748.058907_meta.json"
    )
    assert img.base_name == meta.base_name


def test_with_modality_only_touches_modality():
    n = KameraImageName.parse(
        "test_seattle_2020_fl09_R_20200830_020748.058907_ir.jpg"
    )
    swapped = n.with_modality("rgb")
    assert swapped.name == "test_seattle_2020_fl09_R_20200830_020748.058907_rgb.jpg"
    # A name whose *effort prefix* contains the modality substring must
    # survive the swap untouched (str.replace would corrupt it).
    n = KameraImageName.parse("first_flight_2020_fl01_C_20200830_010203.4_ir.jpg")
    assert (
        n.with_modality("rgb").name
        == "first_flight_2020_fl01_C_20200830_010203.4_rgb.jpg"
    )


def test_parse_rejects_non_kamera_names():
    with pytest.raises(ValueError):
        KameraImageName.parse("image_locations.txt")
    with pytest.raises(ValueError):
        KameraImageName.parse("a_b_c_notadate_notatime_rgb.jpg")


def test_camera_name_roundtrip():
    c = KameraCameraName.parse("85mm_25_5deg_center_rgb")
    assert c.prefix == "85mm_25_5deg"
    assert c.channel == "center"
    assert c.modality == "rgb"
    assert c.name == "85mm_25_5deg_center_rgb"
    assert c.with_modality("uv").name == "85mm_25_5deg_center_uv"


def test_camera_name_from_path():
    c = KameraCameraName.parse("images0/85mm_25_5deg_left_ir")
    assert c.channel == "left"
    assert c.modality == "ir"


def test_swap_image_name_modality():
    name = (
        "85mm_25_5deg_center_uv/"
        "test_seattle_2020_fl09_C_20200830_020748.058907_uv.jpg"
    )
    assert swap_image_name_modality(name, "rgb") == (
        "85mm_25_5deg_center_rgb/"
        "test_seattle_2020_fl09_C_20200830_020748.058907_rgb.jpg"
    )
    # bare basename, no directory component
    assert swap_image_name_modality(
        "test_seattle_2020_fl09_C_20200830_020748.058907_uv.jpg", "rgb"
    ) == "test_seattle_2020_fl09_C_20200830_020748.058907_rgb.jpg"

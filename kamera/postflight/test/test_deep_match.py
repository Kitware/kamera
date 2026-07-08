import numpy as np
import pytest
import scipy.ndimage as ndi

from kamera.postflight.deep_match import to_uint8


def test_to_uint8_passthrough():
    img = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)
    assert to_uint8(img) is img


def test_to_uint8_stretches_uint16():
    img = np.linspace(1000, 3000, 64 * 64, dtype=np.uint16).reshape(64, 64)
    out = to_uint8(img)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_to_uint8_constant_image_no_divide_by_zero():
    img = np.full((32, 32), 500, dtype=np.uint16)
    out = to_uint8(img)
    assert out.dtype == np.uint8


def test_to_uint8_clahe():
    img = np.linspace(0, 4000, 64 * 64, dtype=np.uint16).reshape(64, 64)
    out = to_uint8(img, clahe=True)
    assert out.dtype == np.uint8 and out.shape == img.shape


def _textured_image(rng, shape=(512, 640)):
    img = ndi.gaussian_filter(rng.uniform(0, 1, shape), 3)
    return ((img - img.min()) / np.ptp(img) * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def matcher():
    pytest.importorskip("vismatch")
    from kamera.postflight.deep_match import DeepMatcher

    return DeepMatcher(device="cpu")


def test_match_identity(matcher):
    """Guards the coordinate contract: keypoints must come back in the
    frame of the arrays passed in, regardless of internal resizing."""
    img = _textured_image(np.random.default_rng(0))
    k0, k1 = matcher.match(img, img)
    assert len(k0) > 100
    assert np.median(np.linalg.norm(k0 - k1, axis=1)) < 1.0


def test_match_known_shift(matcher):
    rng = np.random.default_rng(1)
    img = _textured_image(rng)
    shift = 32
    k0, k1 = matcher.match(img, np.roll(img, shift, axis=1))
    assert len(k0) > 100
    dx, dy = np.median(k1 - k0, axis=0)
    assert abs(dx - shift) < 1.5
    assert abs(dy) < 1.5

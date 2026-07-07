import numpy as np

from kamera.postflight.flight_prep import (
    TriggerFrame,
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

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kamera.postflight.boresight import average_quaternions


def test_average_recovers_true_rotation():
    rng = np.random.default_rng(0)
    true = Rotation.from_euler("xyz", [10, -20, 55], degrees=True)
    noise = Rotation.from_rotvec(rng.normal(0, np.radians(0.2), size=(200, 3)))
    quats = (noise * true).as_quat()
    mean = Rotation.from_quat(average_quaternions(quats))
    err = (mean * true.inv()).magnitude()
    assert np.degrees(err) < 0.1


def test_average_handles_sign_flips():
    true = Rotation.from_euler("z", 45, degrees=True)
    q = true.as_quat()
    quats = np.array([q, -q, q, -q])
    mean = Rotation.from_quat(average_quaternions(quats))
    assert np.degrees((mean * true.inv()).magnitude()) < 1e-6


def test_average_weights():
    a = Rotation.from_euler("z", 0, degrees=True).as_quat()
    b = Rotation.from_euler("z", 10, degrees=True).as_quat()
    # all weight on b -> mean is b
    mean = Rotation.from_quat(
        average_quaternions(np.array([a, b]), weights=np.array([0.0, 1.0]))
    )
    assert np.degrees((mean * Rotation.from_quat(b).inv()).magnitude()) < 1e-6


def test_average_rejects_too_few():
    with pytest.raises(Exception):
        average_quaternions(np.zeros((0, 4)))

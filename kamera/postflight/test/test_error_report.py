import numpy as np
from scipy.spatial.transform import Rotation

from kamera.postflight.boresight import BoresightEstimate, average_quaternions
from kamera.postflight.error_report import (
    residual_components_deg,
    sweep_time_offset,
    write_error_report,
)

TRUE_BORESIGHT = Rotation.from_euler("xyz", [1.0, -2.0, 30.0], degrees=True)


class OscillatingNav:
    """ENU attitude wobbling in roll/pitch with a slow heading drift --
    enough attitude rate for a clock offset to leave a signature."""

    def pose(self, t):
        rot = Rotation.from_euler(
            "xyz",
            [
                5.0 * np.sin(0.8 * t),
                3.0 * np.sin(0.53 * t + 1.0),
                20.0 * np.sin(0.05 * t),
            ],
            degrees=True,
        )
        return np.zeros(3), rot.as_quat()


def make_estimate(
    nav,
    clock_offset_s=0.0,
    heading_noise_deg=0.0,
    tilt_noise_deg=0.01,
    n=300,
    seed=0,
):
    """Synthesize solve_rig_boresight's per-frame samples: exposures
    actually happen at t_rec + clock_offset_s, but nav is interpolated
    at the recorded time t_rec, exactly like the real solve."""
    rng = np.random.default_rng(seed)
    t_rec = np.sort(rng.uniform(0.0, 600.0, n))
    quats = []
    for t in t_rec:
        noise = Rotation.from_euler(
            "xyz",
            [
                rng.normal(0.0, tilt_noise_deg),
                rng.normal(0.0, tilt_noise_deg),
                rng.normal(0.0, heading_noise_deg),
            ],
            degrees=True,
        )
        enu_from_ins_true = Rotation.from_quat(nav.pose(t + clock_offset_s)[1])
        enu_from_rig = enu_from_ins_true * noise * TRUE_BORESIGHT
        enu_from_ins_rec = Rotation.from_quat(nav.pose(t)[1])
        quats.append((enu_from_ins_rec.inv() * enu_from_rig).as_quat())
    quats = np.asarray(quats)
    mean = average_quaternions(quats)
    residuals = np.degrees(
        (Rotation.from_quat(quats) * Rotation.from_quat(mean).inv()).magnitude()
    )
    return BoresightEstimate(
        ins_from_rig=mean,
        lever_arm_ins=np.zeros(3),
        num_frames=n,
        num_rejected=0,
        residuals_deg=residuals,
        sample_times=t_rec,
        sample_quats=quats,
        inlier_mask=np.ones(n, dtype=bool),
    )


def test_sweep_recovers_clock_offset():
    nav = OscillatingNav()
    est = make_estimate(nav, clock_offset_s=0.05)
    offsets, medians = sweep_time_offset(est, nav, max_offset_s=0.2, step_s=0.005)
    best = offsets[np.argmin(medians)]
    assert abs(best - 0.05) <= 0.0075
    # the corrected residual should be dramatically smaller
    assert medians.min() < 0.25 * medians[np.searchsorted(offsets, 0.0)]


def test_sweep_flat_without_offset():
    nav = OscillatingNav()
    est = make_estimate(nav, clock_offset_s=0.0)
    offsets, medians = sweep_time_offset(est, nav, max_offset_s=0.2, step_s=0.01)
    best = offsets[np.argmin(medians)]
    assert abs(best) <= 0.015


def test_heading_noise_lands_on_heading_axis():
    nav = OscillatingNav()
    est = make_estimate(nav, heading_noise_deg=0.3, tilt_noise_deg=0.03)
    comps = residual_components_deg(est)
    rms = np.sqrt(np.mean(comps**2, axis=0))
    assert rms[2] > 3 * max(rms[0], rms[1])


class FakeImage:
    has_pose = True

    def projection_center(self):
        return np.array([0.0, 0.0, 305.0])


class FakeRec:
    images = {i: FakeImage() for i in range(5)}


class FakeModel:
    fx = 8000.0


def test_write_error_report(tmp_path):
    nav = OscillatingNav()
    est = make_estimate(nav, clock_offset_s=0.02, heading_noise_deg=0.1)
    mount = TRUE_BORESIGHT.as_quat()
    groups = {
        "rgb": {
            "estimate": est,
            "rec": FakeRec(),
            "models": {"cam_rgb": FakeModel()},
            "camera_records": {
                "cam_rgb": {
                    "name": "cam_rgb",
                    "is_reference": True,
                    "num_images": 5,
                    "reprojection_error_px": 1.7,
                    "camera_quaternion_xyzw": [float(x) for x in mount],
                }
            },
            "ref_folder": "cam_rgb",
        }
    }
    path = write_error_report(tmp_path, "fl_test", groups, nav)
    assert path.endswith("calibration_error_report.pdf")
    assert (tmp_path / "calibration_error_report.pdf").stat().st_size > 10_000

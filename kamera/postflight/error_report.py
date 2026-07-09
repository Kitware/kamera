"""Post-run calibration error report.

Quantifies each modality group's boresight residual, attributes it to
the two error sources the data can distinguish -- a constant
camera-to-INS time offset and INS heading noise -- and writes a PDF
(``calibration_error_report.pdf``) with per-camera error tables next to
the exported yamls.

The input is the per-frame boresight samples that `solve_rig_boresight`
retains (``BoresightEstimate.sample_*``). Each sample is an independent
single-frame measurement of the same physical rotation, so structure in
their residuals is diagnostic:

- Time sync: the INS attitude is interpolated at the recorded exposure
  time, so a constant clock offset converts attitude *rate* into
  rotation error. Sweeping a trial offset and re-evaluating the
  residual spread locates the offset the data supports; a flat sweep
  curve means timing does not explain the residual.
- Heading: single-antenna INS heading is typically several times worse
  than roll/pitch. Expressing each residual as a rotation vector in the
  INS body frame (aerospace NED: x forward, y right, z down -- see the
  ``rzyx`` euler order in ``nav_state``) splits it per axis; a dominant
  z component points at heading noise.

At runtime imagery is projected through the INS pose plus the static
mount, so the per-frame residual here is the per-frame pointing error
production will see -- ``altitude * tan(residual)`` of ground error.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.transform import Rotation

from kamera.postflight.boresight import BoresightEstimate, average_quaternions

__all__ = [
    "residual_components_deg",
    "sweep_time_offset",
    "write_error_report",
]

_AXIS_LABELS = ["roll (body x)", "pitch (body y)", "heading (body z)"]


def _nav_attitudes(nav_state_provider, times: np.ndarray) -> Rotation:
    """Batched enu_from_ins attitude at each time."""
    return Rotation.from_quat(
        np.asarray([nav_state_provider.pose(t)[1] for t in times])
    )


def _spread_deg(quats: np.ndarray) -> np.ndarray:
    """Per-sample angular distance (deg) from the chordal mean."""
    mean = Rotation.from_quat(average_quaternions(quats))
    return np.degrees((Rotation.from_quat(quats) * mean.inv()).magnitude())


def sweep_time_offset(
    estimate: BoresightEstimate,
    nav_state_provider,
    max_offset_s: float = 0.25,
    step_s: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median boresight residual as a function of a constant offset added
    to the exposure times before interpolating the INS attitude.

    Returns (offsets_s, median_residual_deg). The minimum is the
    camera-to-INS clock offset best supported by the data.
    """
    ts = np.asarray(estimate.sample_times)[estimate.inlier_mask]
    quats = np.asarray(estimate.sample_quats)[estimate.inlier_mask]
    # Undo the offset-zero nav attitude baked into each sample to recover
    # the frame's SfM-side rotation, which does not depend on the offset:
    # sample = enu_from_ins(t)^-1 . enu_from_rig
    enu_from_rig = _nav_attitudes(nav_state_provider, ts) * Rotation.from_quat(quats)
    offsets = np.arange(-max_offset_s, max_offset_s + step_s / 2, step_s)
    medians = np.empty(len(offsets))
    for i, dt in enumerate(offsets):
        shifted = _nav_attitudes(nav_state_provider, ts + dt)
        medians[i] = float(np.median(_spread_deg((shifted.inv() * enu_from_rig).as_quat())))
    return offsets, medians


def residual_components_deg(estimate: BoresightEstimate) -> np.ndarray:
    """Inlier residual rotations as rotation vectors (deg) in the INS
    body frame, one row per frame: [roll(x), pitch(y), heading(z)]."""
    mean = Rotation.from_quat(estimate.ins_from_rig)
    quats = np.asarray(estimate.sample_quats)[estimate.inlier_mask]
    return np.degrees((Rotation.from_quat(quats) * mean.inv()).as_rotvec())


def _median_altitude_m(reconstruction) -> Optional[float]:
    """Median camera height above the ENU origin, from posed images."""
    zs = [
        float(im.projection_center()[2])
        for im in reconstruction.images.values()
        if im.has_pose
    ]
    return float(np.median(zs)) if zs else None


def _ground_m(residual_deg: float, altitude_m: Optional[float]) -> Optional[float]:
    if altitude_m is None:
        return None
    return altitude_m * np.tan(np.radians(residual_deg))


def _analyze_group(group: Dict, nav_state_provider) -> Dict:
    """All derived error numbers for one _calibrate_group result."""
    est: BoresightEstimate = group["estimate"]
    out = {
        "estimate": est,
        "altitude_m": _median_altitude_m(group["rec"]),
        "median_deg": float(np.median(est.residuals_deg)),
        "p90_deg": float(np.percentile(est.residuals_deg, 90)),
    }
    if len(est.sample_times) == 0:
        return out
    offsets, medians = sweep_time_offset(est, nav_state_provider)
    best = int(np.argmin(medians))
    comps = residual_components_deg(est)
    out.update(
        offsets_s=offsets,
        sweep_medians_deg=medians,
        best_offset_s=float(offsets[best]),
        best_offset_median_deg=float(medians[best]),
        components_deg=comps,
        axis_rms_deg=np.sqrt(np.mean(comps**2, axis=0)),
        inlier_times=np.asarray(est.sample_times)[est.inlier_mask],
    )
    return out


def _summary_lines(name: str, a: Dict) -> List[str]:
    est = a["estimate"]
    alt = a["altitude_m"]
    lines = [
        f"Group '{name}': {est.num_frames} frames used, "
        f"{est.num_rejected} rejected",
        f"  Boresight residual: median {a['median_deg']:.3f} deg, "
        f"p90 {a['p90_deg']:.3f} deg",
    ]
    if alt is not None:
        lines.append(
            f"  At {alt:.0f} m above the ENU origin that is "
            f"{_ground_m(a['median_deg'], alt):.2f} m (median) / "
            f"{_ground_m(a['p90_deg'], alt):.2f} m (p90) on the ground"
        )
    if "best_offset_s" in a:
        gain = a["median_deg"] - a["best_offset_median_deg"]
        lines.append(
            f"  Time offset: residual minimized at {a['best_offset_s']*1e3:+.0f} ms "
            f"(median {a['best_offset_median_deg']:.3f} deg, "
            f"{gain:.3f} deg better than at 0 ms)"
        )
        rms = a["axis_rms_deg"]
        lines.append(
            "  Residual RMS by axis: "
            + ", ".join(f"{lbl} {v:.3f} deg" for lbl, v in zip(_AXIS_LABELS, rms))
        )
        if rms[2] > 2 * max(rms[0], rms[1]):
            lines.append(
                "  -> heading dominates: consistent with INS heading noise"
            )
    lines.append(
        f"  Lever arm (INS frame, m): {np.round(est.lever_arm_ins, 3).tolist()}"
    )
    return lines


_EXPLANATION = """\
How to read this report

The boresight solve treats every synchronized frame as an independent
measurement of the same rig-to-INS rotation. The residual is each
frame's disagreement with the averaged boresight. Because production
projects imagery through the INS pose plus this static mount, the
per-frame residual IS the pointing error to expect at runtime:
ground error = altitude x tan(residual).

Time-offset sweep: a constant camera-to-INS clock offset turns attitude
rate into rotation error. The sweep re-solves the residual with the nav
attitude sampled at t + offset; a clear minimum away from 0 ms measures
the clock offset, while a flat curve rules timing out.

Heading decomposition: each residual is split into rotations about the
INS body axes. Roll/pitch from the INS are usually good to a few
hundredths of a degree; heading is often several times worse. A heading
(body z) component that dominates the other two points at INS heading
noise rather than the camera calibration.

Per-camera errors: the reprojection error measures how well each
camera's intrinsics + mount reproject the reconstruction's own 3D
points through the INS pose chain (median pixels over sampled images).
Its ground equivalent uses that camera's focal length at the flight
altitude; the boresight residual adds on top of it.
"""


def _summary_page(pdf: PdfPages, flight_name: str, analyses: Dict[str, Dict]):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.95, f"Calibration error report -- {flight_name}",
             fontsize=15, weight="bold")
    text: List[str] = []
    for name, a in analyses.items():
        text.extend(_summary_lines(name, a))
        text.append("")
    fig.text(0.08, 0.90, "\n".join(text), fontsize=9, family="monospace",
             va="top")
    fig.text(0.08, 0.02, _EXPLANATION, fontsize=8, va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def _group_page(pdf: PdfPages, name: str, a: Dict):
    if "components_deg" not in a:
        return
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(f"Boresight residual analysis -- group '{name}'", fontsize=13)
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(grid[0, :])
    t = a["inlier_times"]
    minutes = (t - t.min()) / 60.0
    for i, lbl in enumerate(_AXIS_LABELS):
        ax.plot(minutes, a["components_deg"][:, i], ".", ms=3, label=lbl)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("flight time (min)")
    ax.set_ylabel("residual (deg)")
    ax.set_title("Per-frame residual by INS body axis")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(a["offsets_s"] * 1e3, a["sweep_medians_deg"])
    ax.axvline(a["best_offset_s"] * 1e3, color="r", lw=0.8,
               label=f"best {a['best_offset_s']*1e3:+.0f} ms")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("camera-to-INS time offset (ms)")
    ax.set_ylabel("median residual (deg)")
    ax.set_title("Time-offset sweep")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, 1])
    ax.bar(range(3), a["axis_rms_deg"], color=["C0", "C1", "C2"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["roll", "pitch", "heading"], fontsize=9)
    ax.set_ylabel("residual RMS (deg)")
    ax.set_title("Heading vs tilt attribution")

    pdf.savefig(fig)
    plt.close(fig)


def _camera_table_page(pdf: PdfPages, groups: Dict[str, Dict],
                       analyses: Dict[str, Dict]):
    rows = []
    for name, group in groups.items():
        alt = analyses[name]["altitude_m"]
        for folder, record in group.get("camera_records", {}).items():
            model = group.get("models", {}).get(folder)
            reproj = record.get("reprojection_error_px")
            ground = None
            if reproj is not None and alt is not None and model is not None:
                # one pixel subtends ~altitude/fx meters at nadir
                ground = reproj * alt / float(model.fx)
            ypr = Rotation.from_quat(
                record["camera_quaternion_xyzw"]
            ).as_euler("ZYX", degrees=True)
            rows.append([
                folder,
                name,
                "yes" if record.get("is_reference") else "",
                str(record.get("num_images", "")),
                f"{reproj:.2f}" if reproj is not None else "n/a",
                f"{ground:.2f}" if ground is not None else "n/a",
                " ".join(f"{v:.2f}" for v in ypr),
            ])
    if not rows:
        return
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Per-camera errors", fontsize=13)
    ax = fig.add_subplot(111)
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["camera", "group", "ref", "images",
                   "reproj (px)", "ground (m)", "mount YPR (deg)"],
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.text(
        0.08, 0.06,
        "reproj: median reprojection of the model's own 3D points through "
        "the INS pose + mount, per camera.\nground: that error at the "
        "flight altitude (reproj x altitude / fx); the group boresight "
        "residual adds on top.",
        fontsize=8,
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_error_report(
    save_dir: str | os.PathLike,
    flight_name: str,
    groups: Dict[str, Dict],
    nav_state_provider,
) -> str:
    """Write calibration_error_report.pdf for a finished run.

    `groups` maps group name to a `_calibrate_group` result dict
    (estimate, rec, models, camera_records, ref_folder).
    """
    analyses = {
        name: _analyze_group(group, nav_state_provider)
        for name, group in groups.items()
    }
    path = os.path.join(str(save_dir), "calibration_error_report.pdf")
    with PdfPages(path) as pdf:
        _summary_page(pdf, flight_name, analyses)
        for name, a in analyses.items():
            _group_page(pdf, name, a)
        _camera_table_page(pdf, groups, analyses)
    return path

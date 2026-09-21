"""PDF report: camera table, rig geometry, INS boresight residuals, homography overlays, error budget."""

from __future__ import annotations

import textwrap

import matplotlib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kamera.calibration.rig import RigCalibration

PAGE = (11, 8.5)

ERROR_NOTES = """\
Error budget and what limits it

INS attitude at the trigger. Each meta.json carries one 100 Hz INS sample taken before the
event, so the attitude used here is up to 10 ms stale (median gap reported above). At the
turn rates of a figure-eight (about 5 deg/s) that is up to 0.05 deg, or roughly 25 RGB
pixels, and it enters every frame's boresight estimate as noise. A hardware event-stamped
INS sample or a full-rate log removes it; InsTrajectory accepts either without code changes.

SfM drift. Bundle adjustment with INS position priors pins scale, heading and position to
the INS, but the relative orientation drift of the model over the flight is what dominates the
per-frame boresight scatter. The rig constraint removes the intra-frame freedom entirely, so
the relative camera geometry (and therefore the homographies) is far better determined than
the absolute boresight.

Exposure timing. A camera whose exposure midpoint differs from the reference camera's sees
the ground further along track by ground speed x time difference, and a bundle adjustment on a
translating rig cannot tell that from a camera mounted that far forward. The rig table's
"exposure vs ref" column reads each camera's forward offset back into a time difference at
the flight's ground speed (negative = earlier than the reference). Only the relative timing is
observable: the position priors absorb any delay shared by the whole rig. The camera yaml
positions carry these offsets, which is correct at similar ground speeds.

Lever arms. Beyond that timing signal, at 400 to 900 m a 30 cm baseline subtends less than one
IR pixel, so the rig translations are weakly determined and the reported standard deviations
should be read as such. The INS lever arm is the median offset of the rig origin from the INS
position over all frames.

Homographies. A homography maps one camera onto another exactly only for a plane at one
range, and the timing baseline above makes the range matter. Each pair is fit for the range
in its title (the survey AGL if given, else the calibration flight's median scene range); the
fit residual (rms and p95, in right-image pixels) then measures the lens distortion a single
matrix cannot carry, and the warped overlays show it visually.
"""


def _text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.94, title, fontsize=16, weight="bold", va="top")
    fig.text(
        0.06, 0.88, body, fontsize=9.5, va="top", family="monospace", linespacing=1.4
    )
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(
    pdf: PdfPages, title: str, header: list[str], rows: list[list], widths=None
) -> None:
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold", loc="left", pad=20)
    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="upper center",
        cellLoc="center",
        colWidths=widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    pdf.savefig(fig)
    plt.close(fig)


def camera_page(pdf: PdfPages, cal: RigCalibration) -> None:
    header = [
        "camera",
        "size",
        "fx",
        "fy",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
        "frames",
        "rms px",
        "ifov deg",
    ]
    rows = []
    for name in sorted(cal.cameras):
        c = cal.cameras[name]
        rows.append(
            [
                name,
                f"{c.width}x{c.height}",
                f"{c.K[0, 0]:.1f}",
                f"{c.K[1, 1]:.1f}",
                f"{c.K[0, 2]:.1f}",
                f"{c.K[1, 2]:.1f}",
                *[f"{v:.5f}" for v in c.dist],
                c.frames,
                f"{c.reproj_rms_px:.2f}",
                f"{np.degrees(1 / c.K[0, 0]):.5f}",
            ]
        )
    _table_page(pdf, f"{cal.rig}: camera intrinsics ({cal.flight})", header, rows)


def rig_page(pdf: PdfPages, cal: RigCalibration) -> None:
    ref = cal.cameras[cal.reference]
    fig = plt.figure(figsize=PAGE)
    fig.suptitle(
        f"Rig geometry relative to {cal.reference}",
        fontsize=15,
        weight="bold",
        x=0.06,
        ha="left",
    )
    ax = fig.add_subplot(1, 2, 1)
    ax.axis("off")
    rows = []
    for name in sorted(cal.cameras):
        rel = ref.rig_from_cam.inv() * cal.cameras[name].rig_from_cam
        rv, c = rel.as_rotvec(degrees=True), cal.cameras[name].center_in_rig
        rows.append(
            [
                name,
                f"{np.degrees(rel.magnitude()):.3f}",
                f"{rv[0]:+.3f} {rv[1]:+.3f} {rv[2]:+.3f}",
                f"{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f}",
                f"{cal.implied_delay_ms(name):+.0f}",
            ]
        )
    t = ax.table(
        cellText=rows,
        colLabels=[
            "camera",
            "angle deg",
            "rotvec deg (ref axes)",
            "centre m (rig)",
            "exposure vs ref ms",
        ],
        loc="center",
        cellLoc="center",
        colWidths=[0.14, 0.14, 0.36, 0.3, 0.14],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.6)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    for i, name in enumerate(sorted(cal.cameras)):
        z = cal.cameras[name].rig_from_cam.apply([0, 0, 1])
        ax3.quiver(
            0, 0, 0, *z, length=1.0, label=name, arrow_length_ratio=0.08, color=f"C{i}"
        )
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(0, 1)
    ax3.set_xlabel("rig x")
    ax3.set_ylabel("rig y")
    ax3.set_zlabel("rig z (optical)")
    ax3.set_title("optical axes in the rig frame", fontsize=10)
    ax3.legend(fontsize=6, loc="upper left")
    pdf.savefig(fig)
    plt.close(fig)


def boresight_page(pdf: PdfPages, cal: RigCalibration) -> None:
    keep = cal.inlier
    t = cal.frame_times - cal.frame_times[0]
    res, pos = cal.rotation_residual_deg, cal.position_residual_m
    mag = np.linalg.norm(res[keep], axis=1)
    fig, axes = plt.subplots(2, 2, figsize=PAGE)
    e = cal.ins_from_rig.as_euler("ZYX", degrees=True)
    fig.suptitle(
        f"INS boresight: ins_from_rig euler ZYX = ({e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}) deg, lever arm = "
        f"({cal.lever_arm_m[0]:.2f}, {cal.lever_arm_m[1]:.2f}, {cal.lever_arm_m[2]:.2f}) m; "
        f"{keep.sum()} frames, {(~keep).sum()} rejected",
        fontsize=10,
        weight="bold",
    )
    for i, lbl in enumerate("xyz"):
        axes[0, 0].plot(t[keep], res[keep, i], ".", ms=2, label=f"rot {lbl}")
        axes[1, 0].plot(t[keep], pos[keep, i], ".", ms=2, label=f"pos {lbl}")
    axes[0, 0].set(
        title="per-frame boresight residual (deg, rig axes)",
        xlabel="s since first frame",
    )
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].set(
        title="rig origin vs INS minus lever arm (m, body axes)",
        xlabel="s since first frame",
    )
    axes[1, 0].legend(fontsize=7)
    axes[0, 1].hist(mag, bins=50, color="gray")
    axes[0, 1].set(
        title=f"residual magnitude: median {np.median(mag):.3f}, p90 {np.percentile(mag, 90):.3f} deg",
        xlabel="deg",
    )
    axes[1, 1].hist(cal.ins_gap_s * 1000, bins=40, color="gray")
    axes[1, 1].set(
        title=f"INS sample staleness: median {np.median(cal.ins_gap_s) * 1000:.1f} ms",
        xlabel="ms",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def homography_page(pdf: PdfPages, pair: dict) -> None:
    s = pair["stats"]
    fig = plt.figure(figsize=PAGE)
    fig.suptitle(
        f"{pair['left']} -> {pair['right']} at {s['rangeM']:.0f} m: fit rms {s['rmsPx']:.2f} px, p95 {s['p95Px']:.2f} px, max {s['maxPx']:.2f} px, "
        f"coverage {100 * s['coverage']:.0f}%",
        fontsize=11,
        weight="bold",
    )
    for i, (key, title) in enumerate(
        [
            ("warped_img", f"{pair['left']} warped into {pair['right']}"),
            ("right_img", pair["right"]),
            ("overlay_img", "overlay (magenta/green)"),
        ]
    ):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(pair[key])
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.text(
        0.06,
        0.04,
        "H (left -> right) = "
        + np.array2string(
            np.asarray(pair["h"]), precision=5, suppress_small=True, max_line_width=200
        ).replace("\n", " "),
        fontsize=7,
        family="monospace",
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_report(
    path: str, cal: RigCalibration, pairs: list[dict], notes: str = ""
) -> None:
    with PdfPages(path) as pdf:
        _text_page(
            pdf,
            f"KAMERA rig calibration: {cal.rig}",
            textwrap.dedent(f"""\
            flight:            {cal.flight}
            reference camera:  {cal.reference}
            cameras:           {", ".join(sorted(cal.cameras))}
            frames used:       {int(cal.inlier.sum())}
            """)
            + notes,
        )
        camera_page(pdf, cal)
        rig_page(pdf, cal)
        boresight_page(pdf, cal)
        for pair in pairs:
            homography_page(pdf, pair)
        _text_page(pdf, "Error sources", ERROR_NOTES)

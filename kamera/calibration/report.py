"""PDF report: flight summary, cameras, rig geometry, registration overlays."""

from __future__ import annotations

import datetime
import textwrap
from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kamera.calibration.flight import Frame, InsTrajectory
from kamera.calibration.rig import RigCalibration

PAGE = (11, 8.5)

FRAME_NOTES = """A frame is one trigger event. Every camera fires on it and writes one image, so the
images of a trigger share a single rig position and orientation, and that shared pose
is what the rig bundle adjustment enforces. Only triggers where every camera wrote an
image are used. A frame is registered when structure from motion placed it; the
boresight uses the registered frames whose INS-to-rig rotation is not an outlier."""


@dataclass
class FlightSummary:
    """What the flight folder held and which of it went into the calibration."""

    discovered: int  # triggers found in the flight folder
    complete: int  # triggers with an image from every camera
    selected: list[Frame]  # complete frames handed to structure from motion
    selection: str  # how they were picked, in words
    images_on_disk: dict[str, int]  # per camera, over every discovered trigger
    ins: InsTrajectory


def _utc(t: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc)


def _table_page(
    pdf: PdfPages,
    title: str,
    header: list[str],
    rows: list[list],
    widths=None,
    note: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold", loc="left", pad=20)
    if note:
        fig.text(
            0.06,
            0.86 - 0.033 * (len(rows) + 1) - 0.04,
            "\n".join(textwrap.wrap(note, 120)),
            fontsize=8,
            va="top",
            linespacing=1.4,
        )
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


def summary_page(pdf: PdfPages, cal: RigCalibration, fs: FlightSummary) -> None:
    """Page 1: the flight, the frames, and how many of them each camera contributed."""
    sel = fs.selected
    t0, t1 = sel[0].time, sel[-1].time
    ins_t0, ins_t1 = fs.ins.times[0], fs.ins.times[-1]
    pos = np.array([fs.ins.pose(f.time)[0] for f in sel])
    track_km = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() / 1000
    window = (fs.ins.times >= t0) & (fs.ins.times <= t1)
    alt = fs.ins.llh[window if window.any() else slice(None), 2]
    interval = np.median(np.diff([f.time for f in sel]))
    registered = np.isin(
        np.round([f.time for f in sel], 3), np.round(cal.frame_times, 3)
    )
    facts = [
        ("flight", cal.flight),
        ("rig", f"{cal.rig}: {len(cal.cameras)} cameras, reference {cal.reference}"),
        (
            "date (UTC)",
            f"{_utc(ins_t0):%Y-%m-%d}, {_utc(ins_t0):%H:%M} to {_utc(ins_t1):%H:%M} "
            f"({(ins_t1 - ins_t0) / 60:.0f} min of INS samples)",
        ),
        (
            "frames on disk",
            f"{fs.discovered} triggers, {fs.complete} with every camera",
        ),
        (
            "frames selected",
            f"{len(sel)} ({fs.selection}), {_utc(t0):%H:%M} to {_utc(t1):%H:%M}, "
            f"{(t1 - t0) / 60:.0f} min",
        ),
        (
            "frames registered",
            f"{registered.sum()} placed by SfM, {int(cal.inlier.sum())} used for "
            "the boresight",
        ),
        ("trigger interval", f"median {interval:.2f} s"),
        (
            "ground speed",
            f"median {cal.ground_speed_mps:.0f} m/s, {track_km:.1f} km flown over "
            "the selected frames",
        ),
        (
            "altitude",
            f"INS {alt.min():.0f} to {alt.max():.0f} m above the ellipsoid, "
            f"median scene range {cal.scene_range_m:.0f} m",
        ),
    ]
    fig = plt.figure(figsize=PAGE)
    fig.text(
        0.05, 0.94, f"KAMERA rig calibration: {cal.rig}", fontsize=16, weight="bold"
    )
    width = max(len(k) for k, _ in facts)
    fig.text(
        0.05,
        0.88,
        "\n".join(f"{k:<{width}}  {v}" for k, v in facts),
        fontsize=8.5,
        va="top",
        family="monospace",
        linespacing=1.5,
    )
    fig.text(0.05, 0.66, "What a frame is", fontsize=11, weight="bold", va="top")
    fig.text(
        0.05,
        0.625,
        "\n".join(textwrap.wrap(" ".join(FRAME_NOTES.split()), 78)),
        fontsize=8.5,
        va="top",
        linespacing=1.4,
    )
    ax = fig.add_axes((0.05, 0.08, 0.5, 0.38))
    ax.axis("off")
    ax.set_title("images per camera", fontsize=11, weight="bold", loc="left")
    rows = [
        [
            name,
            f"{c.width}x{c.height}",
            fs.images_on_disk.get(name, 0),
            len(sel),
            c.frames,
            c.observations,
        ]
        for name, c in sorted(cal.cameras.items())
    ]
    table = ax.table(
        cellText=rows,
        colLabels=[
            "camera",
            "size",
            "on disk",
            "selected",
            "registered",
            "observations",
        ],
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    ax = fig.add_axes((0.63, 0.42, 0.33, 0.46))
    enu = fs.ins.enu / 1000
    ax.plot(enu[:, 0], enu[:, 1], "-", color="0.8", lw=0.8, label="whole flight")
    minutes = (np.array([f.time for f in sel]) - t0) / 60
    sc = ax.scatter(
        pos[:, 0] / 1000, pos[:, 1] / 1000, c=minutes, s=6, cmap="viridis", zorder=3
    )
    if not registered.all():
        ax.plot(
            pos[~registered, 0] / 1000,
            pos[~registered, 1] / 1000,
            ".",
            color="red",
            ms=3,
            zorder=4,
            label="selected, not registered",
        )
    # Zoom to the registered frames: the ferry legs run off the plot.
    area = pos[registered] if registered.any() else pos
    lo, hi = area[:, :2].min(0) / 1000, area[:, :2].max(0) / 1000
    margin = 0.1 * max(hi - lo) + 0.1
    ax.set(
        xlim=(lo[0] - margin, hi[0] + margin),
        ylim=(lo[1] - margin, hi[1] + margin),
        xlabel="east (km)",
        ylabel="north (km)",
        title="flight track",
    )
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label(
        "minutes since first selected frame", fontsize=7
    )

    ax = fig.add_axes((0.63, 0.08, 0.33, 0.22))
    ax.plot((fs.ins.times - ins_t0) / 60, fs.ins.llh[:, 2], color="0.4", lw=0.8)
    ax.axvspan((t0 - ins_t0) / 60, (t1 - ins_t0) / 60, color="C0", alpha=0.2)
    ax.set(
        xlabel="minutes since first INS sample",
        ylabel="altitude (m)",
        title="INS altitude, selected window shaded",
    )
    pdf.savefig(fig)
    plt.close(fig)


MODALITY_ORDER = {"rgb": 0, "uv": 1, "ir": 2}
CHANNEL_ORDER = {"L": 0, "C": 1, "R": 2}


def camera_order(name: str) -> tuple[int, int]:
    """Sort key: modality first so focal lengths sit side by side, then L, C, R."""
    channel, modality = name.split("_")
    return MODALITY_ORDER.get(modality, 9), CHANNEL_ORDER.get(channel, 9)


def camera_page(pdf: PdfPages, cal: RigCalibration) -> None:
    header = [
        "camera",
        "fx",
        "fy",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
        "fov deg (h x v)",
        "gsd cm",
        "obs",
        "rms px",
    ]
    rows = []
    for name in sorted(cal.cameras, key=camera_order):
        c = cal.cameras[name]
        fov_h = 2 * np.degrees(np.arctan(c.width / 2 / c.K[0, 0]))
        fov_v = 2 * np.degrees(np.arctan(c.height / 2 / c.K[1, 1]))
        rows.append(
            [
                name,
                f"{c.K[0, 0]:.1f}",
                f"{c.K[1, 1]:.1f}",
                f"{c.K[0, 2]:.1f}",
                f"{c.K[1, 2]:.1f}",
                f"{c.dist[0]:.3f}",
                f"{c.dist[1]:.3f}",
                f"{c.dist[2]:.4f}",
                f"{c.dist[3]:.4f}",
                f"{fov_h:.1f} x {fov_v:.1f}",
                f"{100 * cal.scene_range_m / c.K[0, 0]:.1f}",
                c.observations,
                f"{c.reproj_rms_px:.2f}",
            ]
        )
    _table_page(
        pdf,
        f"{cal.rig}: camera intrinsics ({cal.flight})",
        header,
        rows,
        widths=[
            0.07,
            0.07,
            0.07,
            0.07,
            0.07,
            0.065,
            0.065,
            0.065,
            0.065,
            0.11,
            0.06,
            0.07,
            0.06,
        ],
        note=(
            "OpenCV model: fx, fy, cx, cy in pixels; k1, k2 radial and p1, p2 "
            "tangential distortion. fov is the full field of view from the focal "
            "length and image size. gsd is the ground footprint of one pixel at the "
            f"flight's median scene range of {cal.scene_range_m:.0f} m. obs is the "
            "number of features with a 3D point; rms is their reprojection error."
        ),
    )


def swathe_order(name: str) -> tuple[int, int]:
    """Sort key: L, C, R first so each swathe's three cameras sit together."""
    channel, modality = name.split("_")
    return CHANNEL_ORDER.get(channel, 9), MODALITY_ORDER.get(modality, 9)


def rig_page(pdf: PdfPages, cal: RigCalibration) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.text(
        0.06,
        0.94,
        f"Rig geometry relative to {cal.reference}",
        fontsize=15,
        weight="bold",
    )
    ax = fig.add_axes((0.06, 0.5, 0.88, 0.4))
    ax.axis("off")
    rows = []
    for name in sorted(cal.cameras, key=swathe_order):
        rel = cal.rotation_from_reference(name)
        rv, c = rel.as_rotvec(degrees=True), cal.cameras[name].center_in_rig
        rows.append(
            [
                name,
                f"{np.degrees(rel.magnitude()):.3f}",
                *[f"{v:+.3f}" for v in rv],
                *[f"{v:+.2f}" for v in c],
                f"{cal.implied_delay_ms(name):+.0f}",
            ]
        )
    header = [
        "camera",
        "angle (deg)",
        "rot x (deg)",
        "rot y (deg)",
        "rot z (deg)",
        "lever arm x (m)",
        "lever arm y (m)",
        "lever arm z (m)",
        "exposure offset (ms)",
    ]
    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="upper center",
        cellLoc="center",
        colWidths=[0.08, 0.09, 0.09, 0.09, 0.09, 0.11, 0.11, 0.11, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.text(
        0.06,
        0.6,
        "\n".join(
            textwrap.wrap(
                "Rotation of each camera relative to the reference, as a rotation "
                "vector in the reference camera's axes (x right, y down the image, "
                "z along the optical axis); angle is its magnitude. Lever arm is the "
                "camera centre in that frame. Exposure offset reads the along-track "
                "part of the lever arm as a timing difference at the flight's ground "
                "speed, positive when the camera exposes after the reference; a "
                "bundle adjustment on a moving rig cannot separate the two. Lever arms "
                "are weakly determined at these ranges and should be read as such.",
                125,
            )
        ),
        fontsize=8,
        va="top",
        linespacing=1.4,
    )

    ax3 = fig.add_axes((0.2, 0.0, 0.6, 0.46), projection="3d")
    # Draw the rig as mounted, in INS body axes (forward, right, down) via the
    # boresight: the cameras hang from the mount plate and look down, so the down
    # axis is inverted to point down the page.
    grid = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    ax3.plot_trisurf(grid[:, 0], grid[:, 1], np.zeros(4), color="0.85", alpha=0.5)
    colours = {"rgb": "C0", "uv": "C2", "ir": "C3"}
    for name in sorted(cal.cameras, key=swathe_order):
        z = cal.ins_from_rig.apply(cal.cameras[name].rig_from_cam.apply([0, 0, 1]))
        ax3.quiver(
            0,
            0,
            0,
            *z,
            length=1.0,
            label=name,
            arrow_length_ratio=0.06,
            color=colours.get(name.split("_")[1], "k"),
        )
    ax3.set(
        xlim=(-1, 1),
        ylim=(-1, 1),
        zlim=(1, 0),
        xticks=[],
        yticks=[],
        zticks=[],
    )
    ax3.set_xlabel("forward", fontsize=8)
    ax3.set_ylabel("right (starboard)", fontsize=8)
    ax3.set_zlabel("down", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.view_init(elev=22, azim=20)
    ax3.set_title(
        "optical axes in aircraft body axes, seen from behind the aircraft",
        fontsize=10,
        y=0.98,
    )
    ax3.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.12, 0.5))
    pdf.savefig(fig)
    plt.close(fig)


def homography_page(pdf: PdfPages, cal: RigCalibration, pair: dict) -> None:
    s, left, right = pair["stats"], pair["left"], pair["right"]
    # The fit residual is in right-camera pixels; restate it in the left camera's own
    # pixels and on the ground, since one IR pixel is many RGB pixels.
    scale = cal.cameras[right].K[0, 0] / cal.cameras[left].K[0, 0]
    gsd_cm = 100 * s["rangeM"] / cal.cameras[right].K[0, 0]
    fig = plt.figure(figsize=PAGE)
    fig.text(
        0.03,
        0.965,
        f"{left} -> {right} at {s['rangeM']:.0f} m: "
        f"fit rms {s['rmsPx']:.2f} px, p95 {s['p95Px']:.2f} px, "
        f"max {s['maxPx']:.2f} px in {right} pixels, coverage {100 * s['coverage']:.0f}%",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.03,
        0.94,
        f"in {left} pixels: rms {s['rmsPx'] / scale:.2f}, "
        f"p95 {s['p95Px'] / scale:.2f}, max {s['maxPx'] / scale:.2f} "
        f"(one {left} pixel = {scale:.1f} {right} pixels); "
        f"rms {s['rmsPx'] * gsd_cm:.0f} cm on the ground",
        fontsize=9,
    )
    ax = fig.add_axes((0.03, 0.06, 0.94, 0.835))
    # No GIF frame had both images (or --gif_frames 0): keep the page for its fit.
    if "overlay_img" in pair:
        ax.imshow(pair["overlay_img"])
        ax.set_title(
            f"{pair['right']} in colour; inside the {pair['left']} footprint, "
            f"{pair['left']} warped in magenta over {pair['right']} in green",
            fontsize=8,
        )
    ax.axis("off")
    h_text = np.array2string(
        np.asarray(pair["h"]), precision=5, suppress_small=True, max_line_width=200
    ).replace("\n", " ")
    fig.text(
        0.03, 0.03, "H (left -> right) = " + h_text, fontsize=7, family="monospace"
    )
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def write_report(
    path: str, cal: RigCalibration, pairs: list[dict], flight: FlightSummary
) -> None:
    with PdfPages(path) as pdf:
        summary_page(pdf, cal, flight)
        camera_page(pdf, cal)
        rig_page(pdf, cal)
        for pair in pairs:
            homography_page(pdf, cal, pair)

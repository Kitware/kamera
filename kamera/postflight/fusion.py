"""Fuse IR images into the EO reconstruction with deep cross-modal matches.

The per-group pipeline leaves EO and IR as two independent ENU models
whose mutual consistency comes only from both boresights referencing the
same INS -- an INS-attitude-noise-limited link. This module registers
every IR image directly into the EO model instead (MINIMA-LoFTR against
each image's co-located EO partner, pre-warped into the IR view so the
matcher faces only the modality gap), then re-derives the mounts from
that single multimodal reconstruction. `fuse_ir_into_eo` orchestrates;
the per-stage functions carry their own contracts.

Two constraints shape the design:

- Matched EO pixels are lifted to 3D at the interpolated depth of the
  EO image's own triangulated points, not snapped to them -- EO images
  carry only a few hundred SIFT points, far too sparse for thousands of
  dense matches. For co-located cameras a depth error moves the point
  along the EO ray, which barely changes its bearing from the IR
  camera, so the recovered rotation is insensitive to the interpolation.
- Per-image PnP is only an outlier filter, never the pose: over
  near-planar terrain a single image has a strong tilt/translation
  ambiguity (degrees of rotation wobble). All inlier correspondences
  jointly solve one Sim3 aligning the IR reconstruction to the EO
  reconstruction (`align_ir_to_eo`), so every fused pose inherits the
  IR model's own multi-view relative geometry.

The pre-warp uses the two-boresight mounts only as an initialization (a
0.27 deg mount error is ~4 IR px of warp offset, well within matcher
tolerance); the fused geometry comes entirely from the matches.
"""

import bisect
import functools
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycolmap
from rich import print
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from kamera.colmap_processing.camera_models import StandardCamera
from kamera.postflight.boresight import _folder_camera
from kamera.postflight.deep_match import to_uint8
from kamera.postflight.naming import KameraCameraName, KameraImageName
from kamera.postflight.registration_homography import pixel_homography
from kamera.postflight.rig import _frame_key

__all__ = [
    "EoPartner",
    "IrMatchResult",
    "align_ir_to_eo",
    "eo_partners",
    "fuse_ir_into_eo",
    "insert_ir_image",
    "lift_eo_pixels",
    "match_ir_image",
    "mount_delta_deg",
    "pnp_ir_image",
    "prewarp_eo",
    "refine_fused",
    "snap_index",
]


def prewarp_eo(
    eo_img: np.ndarray,
    h_eo2ir: np.ndarray,
    ir_size: Tuple[int, int],
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a full-res EO image into the IR view.

    Returns (warped, H) where warped is `scale` x the IR size and H maps
    a full-res EO pixel to a warped pixel; a matched warped pixel maps
    back to full-res EO coordinates exactly through inv(H).

    The ~20x EO->IR downscale would alias badly through warpPerspective
    alone, so the image is first area-resized close to the target scale
    (keeping ~2x oversampling) and the homography adjusted for OpenCV's
    half-pixel-center resize convention.
    """
    w_ir, h_ir = ir_size
    out_w, out_h = round(w_ir * scale), round(h_ir * scale)
    H = np.diag([scale, scale, 1.0]) @ h_eo2ir

    corners = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float64,
    )
    eo_corners = cv2.perspectiveTransform(corners[None], np.linalg.inv(H))[0]
    span = max(np.ptp(eo_corners[:, 0]), np.ptp(eo_corners[:, 1]))
    r = min(1.0, 2.0 * max(out_w, out_h) / max(span, 1.0))
    if r < 1.0:
        small = cv2.resize(eo_img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        # OpenCV resize maps src pixel x to r*x + (r-1)/2 (pixel centers).
        A = np.array([[r, 0, (r - 1) / 2], [0, r, (r - 1) / 2], [0, 0, 1.0]])
        warped = cv2.warpPerspective(small, H @ np.linalg.inv(A), (out_w, out_h))
    else:
        warped = cv2.warpPerspective(eo_img, H, (out_w, out_h))
    return warped, H


INVALID_POINT3D = np.uint64(np.iinfo(np.uint64).max)  # pycolmap's invalid id


@dataclass
class _SnapIndex:
    """KD-tree over an EO image's triangulated observations."""

    tree: cKDTree
    point3D_ids: np.ndarray
    xyz: np.ndarray  # [K, 3] world coordinates of the anchors


def snap_index(
    image: "pycolmap.Image",
    reconstruction: "pycolmap.Reconstruction",
    min_points: int = 25,
) -> Optional[_SnapIndex]:
    """Index of the image's points2D that carry a 3D point: the depth
    anchors for `lift_eo_pixels` and the track targets for snapping.
    None if too few to be useful."""
    xy, pids, xyz = [], [], []
    for p2 in image.points2D:
        if p2.has_point3D():
            xy.append(p2.xy)
            pids.append(p2.point3D_id)
            xyz.append(reconstruction.points3D[p2.point3D_id].xyz)
    if len(xy) < min_points:
        return None
    return _SnapIndex(
        cKDTree(np.asarray(xy)),
        np.asarray(pids, dtype=np.uint64),
        np.asarray(xyz),
    )


def lift_eo_pixels(
    camera: "pycolmap.Camera",
    image: "pycolmap.Image",
    snap: _SnapIndex,
    eo_px: np.ndarray,
    k: int = 4,
    max_anchor_px: float = 600.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """World points for EO pixels, by casting each pixel's ray from the
    image's reconstructed pose to the median camera-frame depth of its
    `k` nearest triangulated observations.

    Returns (xyz [N,3], valid [N]); valid is False where the nearest
    depth anchor is further than `max_anchor_px` (depth would be a
    guess)."""
    cfw = image.cam_from_world()
    R, t = cfw.rotation.matrix(), cfw.translation
    anchor_depth = (snap.xyz @ R.T + t)[:, 2]

    kk = min(k, len(snap.point3D_ids))
    dist, idx = snap.tree.query(eo_px, k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    depth = np.median(anchor_depth[idx], axis=1)
    valid = dist[:, 0] <= max_anchor_px

    rays = camera.cam_from_img(np.asarray(eo_px, dtype=np.float64))
    xyz_cam = np.column_stack([rays * depth[:, None], depth])
    return (xyz_cam - t) @ R, valid


def eo_partners(
    ir_images: Dict[int, str],
    eo_images: Dict[int, str],
    pairs_per_ir: int = 1,
    ref_modality: str = "rgb",
    max_dt_s: float = 15.0,
) -> Dict[int, List[int]]:
    """EO partner image ids for each IR image, both given as
    {image_id: colmap name (folder/basename)}.

    Partners come from the co-located station (the IR camera folder with
    its modality swapped to `ref_modality`), keyed by trigger frame_key
    -- never by rebuilding filenames, whose extensions differ across
    modalities. The first partner is the temporally closest trigger
    (the exact same-trigger image when it is registered); `pairs_per_ir`
    walks outward to neighboring triggers. Candidates further than
    `max_dt_s` seconds are dropped: past a few triggers the platform has
    moved so far that the views no longer overlap (an IR trigger whose
    EO images all fell out of the reconstruction gets no partner).
    """
    eo_by_folder: Dict[str, Dict[str, int]] = defaultdict(dict)
    for iid, name in eo_images.items():
        folder, _, base = name.rpartition("/")
        try:
            eo_by_folder[folder][_frame_key(base)] = iid
        except ValueError:
            continue
    keys_by_folder = {f: sorted(d) for f, d in eo_by_folder.items()}

    out: Dict[int, List[int]] = {}
    for iid, name in ir_images.items():
        folder, _, base = name.rpartition("/")
        try:
            eo_folder = KameraCameraName.parse(folder).with_modality(ref_modality).name
            key = _frame_key(base)
        except ValueError:
            continue
        keys = keys_by_folder.get(eo_folder)
        if not keys:
            continue
        # walk outward from the insertion point, temporally nearest first
        pos = bisect.bisect_left(keys, key)
        picked: List[int] = []
        lo, hi = pos - 1, pos
        while len(picked) < pairs_per_ir and (lo >= 0 or hi < len(keys)):
            if hi < len(keys) and (lo < 0 or _key_dist(keys[hi], key) <= _key_dist(keys[lo], key)):
                candidate, hi = keys[hi], hi + 1
            else:
                candidate, lo = keys[lo], lo - 1
            if _key_dist(candidate, key) > max_dt_s:
                break  # walking outward only gets further away
            picked.append(eo_by_folder[eo_folder][candidate])
        if picked:
            out[iid] = picked
    return out


def _key_dist(a: str, b: str) -> float:
    """Temporal distance between two date_time frame keys, seconds."""

    def parse(k: str) -> float:
        date, _, t = k.partition("_")
        return float(date) * 86400 + float(t)

    try:
        return abs(parse(a) - parse(b))
    except ValueError:
        return float("inf")


@dataclass
class EoPartner:
    """One EO image prepared for matching against an IR image."""

    img: np.ndarray  # grayscale uint8, full resolution
    time: float
    model: StandardCamera  # exported mount model, for the warp
    image: "pycolmap.Image"  # reconstructed pose, for the depth lift
    camera: "pycolmap.Camera"
    snap: _SnapIndex


@dataclass
class IrMatchResult:
    """2D(IR) <-> 3D correspondences for one IR image.

    Every row has a depth-lifted 3D point (for PnP); rows that also
    landed within `snap_px` of a triangulated observation carry that
    point's id in `point3D_ids` (else INVALID_POINT3D) with the snap
    distance in `snap_dist` -- those become track observations.
    """

    ir_xy: np.ndarray  # [N, 2]
    xyz: np.ndarray  # [N, 3]
    point3D_ids: np.ndarray  # [N] uint64, INVALID_POINT3D where unsnapped
    snap_dist: np.ndarray  # [N]
    num_raw: int = 0


def _concat(rows: List[np.ndarray], empty_shape: Tuple, dtype="float64") -> np.ndarray:
    return np.concatenate(rows) if rows else np.empty(empty_shape, dtype=dtype)


def match_ir_image(
    ir_img: np.ndarray,
    ir_model: StandardCamera,
    t_ir: float,
    partners: List[EoPartner],
    matcher,
    ground_z: float,
    snap_px: float = 16.0,
    warp_scale: float = 1.0,
) -> IrMatchResult:
    """Match one IR image against its EO partners.

    Per partner: homography pre-warp -> deep match -> map EO keypoints
    back to full resolution -> lift to 3D at interpolated depth (see
    `lift_eo_pixels`), tagging the matches within `snap_px` (full-res EO
    px) of a triangulated observation with that observation's point id.
    """
    rows_xy, rows_xyz, rows_pid, rows_dist = [], [], [], []
    num_raw = 0
    ir_u8 = to_uint8(ir_img)
    h, w = ir_u8.shape[:2]
    for p in partners:
        try:
            H = pixel_homography(p.model, ir_model, p.time, t_ir, ground_z)
        except ValueError:
            continue  # views no longer overlap (e.g. turn between triggers)
        warped, H_used = prewarp_eo(p.img, H, (w, h), warp_scale)
        eo_kpts, ir_kpts = matcher.match(warped, ir_u8)
        num_raw += len(eo_kpts)
        if not len(eo_kpts):
            continue
        eo_px = cv2.perspectiveTransform(
            eo_kpts[None], np.linalg.inv(H_used)
        )[0]
        xyz, valid = lift_eo_pixels(p.camera, p.image, p.snap, eo_px)
        dist, idx = p.snap.tree.query(eo_px)
        pid = np.where(
            dist <= snap_px, p.snap.point3D_ids[idx], INVALID_POINT3D
        )
        rows_xy.append(ir_kpts[valid])
        rows_xyz.append(xyz[valid])
        rows_pid.append(pid[valid])
        rows_dist.append(dist[valid])

    return IrMatchResult(
        _concat(rows_xy, (0, 2)),
        _concat(rows_xyz, (0, 3)),
        _concat(rows_pid, (0,), "uint64"),
        _concat(rows_dist, (0,)),
        num_raw,
    )


def pnp_ir_image(
    result: IrMatchResult,
    ir_camera: "pycolmap.Camera",
    ransac_px: float = 3.0,
    min_inliers: int = 12,
) -> Optional[Tuple["pycolmap.Rigid3d", np.ndarray]]:
    """PnP+RANSAC an IR image into the EO/ENU frame from its depth-lifted
    3D points. Distortion is handled by the passed pycolmap Camera; raw
    IR pixels go in directly. Returns (cam_from_world, inlier_mask) or
    None when registration is unreliable."""
    if len(result.ir_xy) < min_inliers:
        return None
    est = pycolmap.AbsolutePoseEstimationOptions()
    est.ransac.max_error = ransac_px
    ans = pycolmap.estimate_and_refine_absolute_pose(
        result.ir_xy, result.xyz, ir_camera, est,
        pycolmap.AbsolutePoseRefinementOptions(),
    )
    if ans is None or ans["num_inliers"] < min_inliers:
        return None
    return ans["cam_from_world"], np.asarray(ans["inlier_mask"], dtype=bool)


@dataclass
class AlignmentEntry:
    """One IR image's contribution to the model alignment: its pose in
    the IR reconstruction and its PnP-inlier cross-modal correspondences."""

    cam_from_irworld: "pycolmap.Rigid3d"
    camera: "pycolmap.Camera"
    ir_xy: np.ndarray  # [M, 2] inlier IR pixels
    xyz_eo: np.ndarray  # [M, 3] their EO-model 3D points


def align_ir_to_eo(
    entries: List[AlignmentEntry],
    cap_per_image: int = 500,
    f_scale_px: float = 2.0,
) -> Tuple[Rotation, np.ndarray, float, Dict]:
    """Solve the Sim3 (irworld <- eoworld) that best reprojects every
    EO 3D point into every IR image through the IR reconstruction's own
    poses: X_ir = s * R @ X_eo + t.

    Both reconstructions are prior-mapped into the same ENU frame, so
    the transform starts at identity and stays small; solving it jointly
    over all images defeats the per-image planar-PnP tilt ambiguity.
    Returns (R, t, s, stats)."""
    poses, cams, pxs, xyzs = [], [], [], []
    rng = np.random.default_rng(0)
    for e in entries:
        n = len(e.ir_xy)
        sel = (
            rng.choice(n, cap_per_image, replace=False)
            if n > cap_per_image
            else slice(None)
        )
        poses.append(
            (e.cam_from_irworld.rotation.matrix(), e.cam_from_irworld.translation)
        )
        cams.append(e.camera)
        pxs.append(np.asarray(e.ir_xy[sel], dtype=np.float64))
        xyzs.append(np.asarray(e.xyz_eo[sel], dtype=np.float64))

    def residuals(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        t, s = p[3:6], np.exp(p[6])
        out = []
        for (Ri, ti), cam, px, xyz in zip(poses, cams, pxs, xyzs):
            xc = (s * xyz @ R.T + t) @ Ri.T + ti
            norm = np.column_stack(
                [xc[:, 0] / xc[:, 2], xc[:, 1] / xc[:, 2], np.ones(len(xc))]
            )
            out.append((cam.img_from_cam(norm) - px).ravel())
        return np.concatenate(out)

    fit = least_squares(
        residuals, np.zeros(7), loss="soft_l1", f_scale=f_scale_px, x_scale="jac"
    )
    R = Rotation.from_rotvec(fit.x[:3])
    t, s = fit.x[3:6], float(np.exp(fit.x[6]))
    res = residuals(fit.x).reshape(-1, 2)
    err = np.linalg.norm(res, axis=1)
    stats = {
        "num_images": len(entries),
        "num_correspondences": int(len(err)),
        "rotation_deg": float(np.degrees(R.magnitude())),
        "translation_m": [float(x) for x in t],
        "scale": s,
        "reproj_px_median": float(np.median(err)),
        "reproj_px_p90": float(np.percentile(err, 90)),
    }
    print(
        f"IR->EO model alignment over {stats['num_images']} images / "
        f"{stats['num_correspondences']} matches: rot {stats['rotation_deg']:.3f} deg, "
        f"trans {np.round(t, 2)} m, scale {s:.5f}, "
        f"reproj median {stats['reproj_px_median']:.2f} px."
    )
    return R, t, s, stats


def aligned_ir_pose(
    cam_from_irworld: "pycolmap.Rigid3d",
    R: Rotation,
    t: np.ndarray,
    s: float,
) -> "pycolmap.Rigid3d":
    """The IR image's pose in the EO world, given the Sim3 from
    `align_ir_to_eo`. The scale is absorbed into the translation so the
    projection is unchanged (X_cam and s*X_cam project identically)."""
    Ri = cam_from_irworld.rotation.matrix()
    ti = cam_from_irworld.translation
    rot = Rotation.from_matrix(Ri @ R.as_matrix())
    return pycolmap.Rigid3d(
        pycolmap.Rotation3d(rot.as_quat()), (Ri @ t + ti) / s
    )


def _fresh_image_id(rec: "pycolmap.Reconstruction") -> int:
    # Trivial frames share their image's id, so stay clear of both.
    return max(max(rec.images, default=0), max(rec.frames, default=0)) + 1


def add_ir_camera(
    rec: "pycolmap.Reconstruction", camera: "pycolmap.Camera"
) -> int:
    """Copy an IR camera (from the IR group model) into the EO
    reconstruction under a fresh id, with its trivial rig."""
    cid = max(rec.cameras, default=0) + 1
    rec.add_camera_with_trivial_rig(
        pycolmap.Camera(
            model=camera.model.name,
            width=camera.width,
            height=camera.height,
            params=camera.params,
            camera_id=cid,
        )
    )
    return cid


def insert_ir_image(
    rec: "pycolmap.Reconstruction",
    camera_id: int,
    name: str,
    result: IrMatchResult,
    inlier_mask: np.ndarray,
    cam_from_world: "pycolmap.Rigid3d",
) -> Tuple[int, int]:
    """Add a posed IR image; its PnP-inlier keypoints that snapped onto
    an existing EO 3D point become observations of that point, extending
    the track across the modality gap (at most one observation per point
    -- COLMAP's track invariant -- keeping the smallest snap distance).
    Returns (image id, number of observations added)."""
    image_id = _fresh_image_id(rec)
    image = pycolmap.Image(
        name=name,
        keypoints=np.asarray(result.ir_xy, dtype=np.float64),
        camera_id=camera_id,
        image_id=image_id,
    )
    rec.add_image_with_trivial_frame(image, cam_from_world)
    best: Dict[int, Tuple[float, int]] = {}
    for idx, (pid, d, ok) in enumerate(
        zip(result.point3D_ids, result.snap_dist, inlier_mask)
    ):
        if ok and pid != INVALID_POINT3D:
            key = int(pid)
            if key not in best or d < best[key][0]:
                best[key] = (float(d), idx)
    for pid, (_, idx) in best.items():
        rec.add_observation(pid, pycolmap.TrackElement(image_id, idx))
    return image_id, len(best)


def refine_fused(
    rec: "pycolmap.Reconstruction",
    ir_image_ids: List[int],
    refine_ir_intrinsics: bool = False,
    ir_camera_ids: Tuple[int, ...] = (),
) -> "pycolmap.BundleAdjuster":
    """Bundle-adjust the fused model with the EO side frozen.

    EO frame poses and all intrinsics are held constant (IR intrinsics
    optionally free); 3D points and IR poses refine, so the multimodal
    tracks pull the IR registrations against the full EO structure
    without touching the EO gauge or geometry.
    """
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_principal_point = False
    options.refine_focal_length = refine_ir_intrinsics
    options.refine_extra_params = refine_ir_intrinsics
    options.print_summary = False

    config = pycolmap.BundleAdjustmentConfig()
    ir_ids = set(ir_image_ids)
    ir_cams = set(ir_camera_ids)
    for iid, im in rec.images.items():
        if not im.has_pose:
            continue
        if iid in ir_ids and im.num_points3D < 6:
            continue  # too few observations to constrain; PnP pose stands
        config.add_image(iid)
        if iid not in ir_ids:
            config.set_constant_rig_from_world_pose(im.frame_id)
    for cid in rec.cameras:
        if not (refine_ir_intrinsics and cid in ir_cams):
            config.set_constant_cam_intrinsics(cid)

    adjuster = pycolmap.create_default_bundle_adjuster(options, config, rec)
    summary = adjuster.solve()
    print(
        f"Fused bundle adjustment: {summary.termination_type}, "
        f"{summary.num_residuals} residuals."
    )
    return adjuster


def mount_delta_deg(a: StandardCamera, b: StandardCamera) -> float:
    """Angle between two exported camera mounts, degrees."""
    ra = Rotation.from_quat(np.asarray(a.cam_quat))
    rb = Rotation.from_quat(np.asarray(b.cam_quat))
    return float(np.degrees((ra * rb.inv()).magnitude()))


def _load_gray(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None if img is None else to_uint8(img)


def fuse_ir_into_eo(
    eo_rec: "pycolmap.Reconstruction",
    ir_rec: "pycolmap.Reconstruction",
    models: Dict[str, StandardCamera],
    image_dirs: Dict[str, str],
    times: Dict[str, float],
    matcher,
    pairs_per_ir: int = 1,
    max_dt_s: float = 15.0,
    snap_px: float = 16.0,
    ransac_px: float = 3.0,
    min_inliers: int = 12,
    warp_scale: float = 1.0,
    run_ba: bool = False,
    refine_ir_intrinsics: bool = False,
    max_images: Optional[int] = None,
) -> Dict:
    """Register the IR group's images into the EO reconstruction
    (mutating `eo_rec` into the fused model) and return a report.

    `models` are the per-group exported StandardCameras by camera folder
    (both the warp initialization and the IR intrinsics come from that
    stage); `image_dirs` maps folders to their images0 directories;
    `times` maps base names to exposure times. `max_images` uniformly
    subsamples the IR images for smoke runs.

    `run_ba` is off by default: the fused poses come from the joint
    model alignment, and per-image bundle adjustment against the sparse
    multimodal track observations would re-introduce the very per-image
    tilt wobble the alignment defeats. `refine_ir_intrinsics` implies it.
    """
    run_ba = run_ba or refine_ir_intrinsics
    ground_z = float(
        np.median([p.xyz[2] for p in eo_rec.points3D.values()])
    )
    print(f"Fusion ground plane z = {ground_z:.1f} m (median EO point).")

    ir_images = {
        iid: im.name
        for iid, im in ir_rec.images.items()
        if im.has_pose and "/" in im.name
    }
    eo_images = {
        iid: im.name
        for iid, im in eo_rec.images.items()
        if im.has_pose and "/" in im.name
    }
    partners_by_ir = eo_partners(
        ir_images, eo_images, pairs_per_ir, max_dt_s=max_dt_s
    )

    ordered = sorted(ir_images, key=lambda i: ir_images[i])
    if max_images is not None and len(ordered) > max_images:
        step = len(ordered) / max_images
        ordered = [ordered[int(i * step)] for i in range(max_images)]

    ir_folder_camera = _folder_camera(ir_rec)
    camera_ids: Dict[str, int] = {}

    @functools.lru_cache(maxsize=None)
    def _snap(eo_id: int) -> Optional[_SnapIndex]:
        return snap_index(eo_rec.images[eo_id], eo_rec)

    @functools.lru_cache(maxsize=16)  # full-res EO images are large
    def _gray(eo_id: int) -> Optional[np.ndarray]:
        folder, _, base = eo_images[eo_id].rpartition("/")
        return _load_gray(os.path.join(image_dirs[folder], base))

    def eo_partner(eo_id: int) -> Optional[EoPartner]:
        name = eo_images[eo_id]
        folder, _, base = name.rpartition("/")
        if folder not in models or folder not in image_dirs:
            return None
        try:
            t = times[KameraImageName.parse(base).base_name]
        except (ValueError, KeyError):
            return None
        snap, img = _snap(eo_id), _gray(eo_id)
        if snap is None or img is None:
            return None
        image = eo_rec.images[eo_id]
        return EoPartner(
            img, t, models[folder], image, eo_rec.cameras[image.camera_id], snap
        )

    skipped = defaultdict(int)
    matched = []  # (name, folder, result, inlier_mask, alignment_entry)
    per_image: List[dict] = []
    inlier_counts: Dict[str, List[int]] = defaultdict(list)
    for n, ir_id in enumerate(ordered, 1):
        name = ir_images[ir_id]
        folder, _, base = name.rpartition("/")
        partners = [
            p
            for p in map(eo_partner, partners_by_ir.get(ir_id, []))
            if p is not None
        ]
        if not partners:
            skipped["no_partner"] += 1
            continue
        if folder not in models or folder not in image_dirs:
            skipped["no_ir_model"] += 1
            continue
        try:
            t_ir = times[KameraImageName.parse(base).base_name]
        except (ValueError, KeyError):
            skipped["no_time"] += 1
            continue
        ir_img = cv2.imread(
            os.path.join(image_dirs[folder], base), cv2.IMREAD_UNCHANGED
        )
        if ir_img is None:
            skipped["unreadable"] += 1
            continue

        result = match_ir_image(
            ir_img, models[folder], t_ir, partners, matcher, ground_z,
            snap_px=snap_px, warp_scale=warp_scale,
        )
        # PnP filters outliers and gates reliability; the pose itself
        # comes from the joint model alignment below.
        pnp = pnp_ir_image(
            result, ir_folder_camera[folder],
            ransac_px=ransac_px, min_inliers=min_inliers,
        )
        if pnp is None:
            skipped["pnp_failed" if len(result.ir_xy) >= min_inliers else "few_matches"] += 1
            continue
        _, inlier_mask = pnp
        entry = AlignmentEntry(
            ir_rec.images[ir_id].cam_from_world(),
            ir_folder_camera[folder],
            result.ir_xy[inlier_mask],
            result.xyz[inlier_mask],
        )
        result.xyz = np.empty((0, 3))  # the entry holds the inlier copy
        matched.append((name, folder, result, inlier_mask, entry))
        inlier_counts[folder].append(int(inlier_mask.sum()))
        if n % 25 == 0 or n == len(ordered):
            print(
                f"  [{n}/{len(ordered)}] matched {len(matched)}, "
                f"skipped {sum(skipped.values())}"
            )

    fused_ids: List[int] = []
    align_stats = None
    if matched:
        R, t, s, align_stats = align_ir_to_eo([m[4] for m in matched])
        for name, folder, result, inlier_mask, entry in matched:
            pose = aligned_ir_pose(entry.cam_from_irworld, R, t, s)
            if folder not in camera_ids:
                camera_ids[folder] = add_ir_camera(
                    eo_rec, ir_folder_camera[folder]
                )
            image_id, num_obs = insert_ir_image(
                eo_rec, camera_ids[folder], name, result, inlier_mask, pose
            )
            fused_ids.append(image_id)
            per_image.append(
                {
                    "name": name,
                    "num_raw": result.num_raw,
                    "num_lifted": len(result.ir_xy),
                    "num_inliers": int(inlier_mask.sum()),
                    "num_track_observations": num_obs,
                }
            )

    if fused_ids and run_ba:
        refine_fused(
            eo_rec, fused_ids,
            refine_ir_intrinsics=refine_ir_intrinsics,
            ir_camera_ids=tuple(camera_ids.values()),
        )

    report = {
        "matcher": getattr(matcher, "name", str(matcher)),
        "num_ir_candidates": len(ordered),
        "num_fused": len(fused_ids),
        "skipped": dict(skipped),
        "model_alignment": align_stats,
        "bundle_adjusted": bool(fused_ids and run_ba),
        "per_folder": {
            f: {
                "fused": len(c),
                "median_inliers": float(np.median(c)),
            }
            for f, c in sorted(inlier_counts.items())
        },
        "per_image": per_image,
    }
    print(
        f"Fused {report['num_fused']}/{report['num_ir_candidates']} IR images "
        f"into the EO model (skipped: {report['skipped'] or 'none'})."
    )
    return report

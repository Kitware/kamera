"""Tests for cross-modal fusion, on synthetic nadir geometry.

The fixture is a pair of colocated nadir cameras at 500 m over a flat
ground plane (z=0): a narrow-FOV high-res "EO" and a wide-FOV low-res
"IR", built both as StandardCameras (warp/matching side) and as a
pycolmap reconstruction (snap/PnP/surgery side) from the same numbers,
so the two projection stacks agree exactly. No torch required -- the
matcher is faked where needed.
"""

import os

import cv2
import numpy as np
import pycolmap
import pytest
from scipy.spatial.transform import Rotation

from kamera.colmap_processing.camera_models import StandardCamera
from kamera.colmap_processing.platform_pose import PlatformPoseFixed
from kamera.postflight.fusion import (
    INVALID_POINT3D,
    AlignmentEntry,
    EoPartner,
    IrMatchResult,
    add_ir_camera,
    align_ir_to_eo,
    aligned_ir_pose,
    eo_partners,
    fuse_ir_into_eo,
    insert_ir_image,
    lift_eo_pixels,
    match_ir_image,
    mount_delta_deg,
    pnp_ir_image,
    prewarp_eo,
    refine_fused,
    snap_index,
    warp_homography,
)

ALT = 500.0
# camera z (optical axis) down, x east: camera -> world (= INS here)
NADIR = Rotation.from_euler("x", 180, degrees=True)
EO = dict(width=2000, height=1500, f=3000.0)
IR = dict(width=640, height=512, f=700.0)

EO_NAME = "pre_center_rgb/pre_fl00_C_20260101_00000{i}.000000_rgb.jpg"
IR_NAME = "pre_center_ir/pre_fl00_C_20260101_00000{i}.000000_ir.png"


def platform(pos=(0, 0, ALT)):
    return PlatformPoseFixed(
        pos=np.asarray(pos, dtype=float), quat=np.array([0, 0, 0, 1.0]),
        lat0=0.0, lon0=0.0, h0=0.0,
    )


def standard_camera(spec, pos=(0, 0, ALT)):
    K = np.array(
        [[spec["f"], 0, spec["width"] / 2], [0, spec["f"], spec["height"] / 2], [0, 0, 1]]
    )
    return StandardCamera(
        spec["width"], spec["height"], K, np.zeros(4), np.zeros(3),
        NADIR.as_quat(), platform_pose_provider=platform(pos),
    )


def cam_from_world(center=(0, 0, ALT)):
    R = NADIR.inv()  # world -> camera (Rx(180) is its own inverse)
    t = -R.apply(np.asarray(center, dtype=float))
    return pycolmap.Rigid3d(pycolmap.Rotation3d(R.as_quat()), t)


def project(spec, pose, xyz):
    pc = pose.rotation.matrix() @ np.asarray(xyz).T + pose.translation[:, None]
    return np.column_stack(
        [
            spec["f"] * pc[0] / pc[2] + spec["width"] / 2,
            spec["f"] * pc[1] / pc[2] + spec["height"] / 2,
        ]
    )


def ground_points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(-110, 110, n), rng.uniform(-85, 85, n), np.zeros(n)]
    )


def build_eo_rec(n_pts=200, centers=((0, 0, ALT), (30, 0, ALT))):
    """EO reconstruction: one PINHOLE camera, one image per trigger, and
    ground points with a track element in every image."""
    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(
        pycolmap.Camera(
            model="PINHOLE", width=EO["width"], height=EO["height"],
            params=[EO["f"], EO["f"], EO["width"] / 2, EO["height"] / 2],
            camera_id=1,
        )
    )
    xyz = ground_points(n_pts)
    ids = []
    for i, center in enumerate(centers, start=1):
        pose = cam_from_world(center)
        image = pycolmap.Image(
            name=EO_NAME.format(i=i),
            keypoints=project(EO, pose, xyz),
            camera_id=1,
            image_id=i,
        )
        rec.add_image_with_trivial_frame(image, pose)
        ids.append(i)
    pids = []
    for j in range(n_pts):
        pid = rec.add_point3D(xyz[j], pycolmap.Track())
        for i in ids:
            rec.add_observation(pid, pycolmap.TrackElement(i, j))
        pids.append(pid)
    return rec, xyz, np.asarray(pids)


class FakeMatcher:
    """Returns preset correspondences, ignoring the images."""

    name = "fake"

    def __init__(self, kpts0, kpts1):
        self.kpts0, self.kpts1 = kpts0, kpts1

    def match(self, img0, img1):
        return np.asarray(self.kpts0, float), np.asarray(self.kpts1, float)


# ---------------------------------------------------------------- geometry


def test_warp_homography_roundtrip():
    eo, ir = standard_camera(EO), standard_camera(IR)
    H = warp_homography(eo, 0.0, ir, 0.0, ground_z=0.0)
    xyz = ground_points(50, seed=1)
    eo_px = eo.project(xyz.T, 0.0)
    ir_px = ir.project(xyz.T, 0.0)
    mapped = cv2.perspectiveTransform(eo_px.T[None], H)[0]
    assert np.max(np.linalg.norm(mapped - ir_px.T, axis=1)) < 0.5


def test_warp_homography_moving_platform():
    # neighbor-trigger partner: EO exposed 40 m away from the IR trigger
    eo, ir = standard_camera(EO, pos=(40, 10, ALT)), standard_camera(IR)
    H = warp_homography(eo, 0.0, ir, 0.0, ground_z=0.0)
    xyz = ground_points(50, seed=2)
    eo_px = eo.project(xyz.T, 0.0)
    ir_px = ir.project(xyz.T, 0.0)
    inside = (eo_px[0] >= 0) & (eo_px[0] < EO["width"]) & (eo_px[1] >= 0) & (
        eo_px[1] < EO["height"]
    )
    mapped = cv2.perspectiveTransform(eo_px.T[inside][None], H)[0]
    assert np.max(np.linalg.norm(mapped - ir_px.T[inside], axis=1)) < 0.5


def test_warp_homography_no_overlap():
    eo = standard_camera(EO, pos=(10000, 0, ALT))
    ir = standard_camera(IR)
    with pytest.raises(ValueError):
        warp_homography(eo, 0.0, ir, 0.0, ground_z=0.0)


def test_prewarp_backmapping():
    eo, ir = standard_camera(EO), standard_camera(IR)
    H = warp_homography(eo, 0.0, ir, 0.0, ground_z=0.0)
    # smooth intensity field so resampling error stays small
    xx, yy = np.meshgrid(np.arange(EO["width"]), np.arange(EO["height"]))
    eo_img = ((xx / EO["width"] + yy / EO["height"]) * 127).astype(np.uint8)
    warped, H_used = prewarp_eo(eo_img, H, (IR["width"], IR["height"]))
    assert warped.shape == (IR["height"], IR["width"])

    xyz = ground_points(30, seed=3)
    eo_px = eo.project(xyz.T, 0.0).T
    wpx = cv2.perspectiveTransform(eo_px[None], H_used)[0]
    inside = (
        (wpx[:, 0] > 2)
        & (wpx[:, 0] < IR["width"] - 2)
        & (wpx[:, 1] > 2)
        & (wpx[:, 1] < IR["height"] - 2)
    )
    assert inside.sum() > 10
    # the warped image at H(p) shows what the EO image shows at p
    for p, wp in zip(eo_px[inside], wpx[inside]):
        got = warped[int(round(wp[1])), int(round(wp[0]))]
        want = eo_img[int(round(p[1])), int(round(p[0]))]
        assert abs(int(got) - int(want)) <= 3
    # and back-mapping through inv(H_used) recovers the EO pixel exactly
    back = cv2.perspectiveTransform(wpx[None], np.linalg.inv(H_used))[0]
    assert np.max(np.linalg.norm(back - eo_px, axis=1)) < 1e-6


def test_snap_index():
    rec, xyz, pids = build_eo_rec()
    image = rec.images[1]
    snap = snap_index(image, rec)
    kp5 = image.points2D[5].xy
    dist, idx = snap.tree.query(kp5)
    assert dist < 1e-9 and int(snap.point3D_ids[idx]) == int(pids[5])
    assert np.allclose(snap.xyz[idx], xyz[5])
    dist, _ = snap.tree.query(kp5 + [20.0, 0.0])
    assert 19 < dist <= 20.5


def test_snap_index_too_few():
    rec, _, _ = build_eo_rec(n_pts=10)
    assert snap_index(rec.images[1], rec, min_points=25) is None


def test_lift_eo_pixels_recovers_ground_points():
    rec, xyz, pids = build_eo_rec()
    image = rec.images[1]
    snap = snap_index(image, rec)
    # lift at pixels midway between keypoints: planar scene -> exact depth
    query_xyz = ground_points(40, seed=9)
    eo_px = project(EO, image.cam_from_world(), query_xyz)
    lifted, valid = lift_eo_pixels(rec.cameras[1], image, snap, eo_px)
    assert valid.all()
    assert np.max(np.linalg.norm(lifted - query_xyz, axis=1)) < 1e-6


def test_lift_eo_pixels_rejects_far_from_anchors():
    rec, xyz, pids = build_eo_rec()
    image = rec.images[1]
    snap = snap_index(image, rec)
    # a pixel in the image corner, far outside the anchor cloud
    _, valid = lift_eo_pixels(
        rec.cameras[1], image, snap, np.array([[1.0, 1.0]]), max_anchor_px=50
    )
    assert not valid.any()


def test_eo_partners_colocated_and_neighbors():
    ir = {1: IR_NAME.format(i=2)}
    eo = {10 + i: EO_NAME.format(i=i) for i in (1, 2, 3)}
    assert eo_partners(ir, eo, pairs_per_ir=1) == {1: [12]}
    got = eo_partners(ir, eo, pairs_per_ir=3)[1]
    assert got[0] == 12 and set(got) == {11, 12, 13}


def test_eo_partners_missing_station():
    ir = {1: "pre_left_ir/pre_fl00_L_20260101_000001.000000_ir.png"}
    eo = {10: EO_NAME.format(i=1)}  # only the center station has EO
    assert eo_partners(ir, eo) == {}


def test_eo_partners_unregistered_trigger_uses_nearest():
    ir = {1: IR_NAME.format(i=2)}
    eo = {10: EO_NAME.format(i=1), 13: EO_NAME.format(i=3)}
    got = eo_partners(ir, eo, pairs_per_ir=1)[1]
    assert len(got) == 1 and got[0] in (10, 13)


def test_eo_partners_caps_temporal_distance():
    # nearest surviving EO trigger is 39 s away -> no partner at all
    ir = {1: "pre_center_ir/pre_fl00_C_20260101_000001.000000_ir.png"}
    eo = {10: "pre_center_rgb/pre_fl00_C_20260101_000040.000000_rgb.jpg"}
    assert eo_partners(ir, eo, pairs_per_ir=3, max_dt_s=15.0) == {}
    assert eo_partners(ir, eo, pairs_per_ir=3, max_dt_s=60.0) == {1: [10]}


# ------------------------------------------------------------ PnP + surgery


def ir_camera(camera_id=50):
    return pycolmap.Camera(
        model="OPENCV", width=IR["width"], height=IR["height"],
        params=[IR["f"], IR["f"], IR["width"] / 2, IR["height"] / 2, 0, 0, 0, 0],
        camera_id=camera_id,
    )


def match_result(px, xyz, pids=None, snapped=True):
    """IrMatchResult from explicit correspondences (test helper)."""
    n = len(px)
    if pids is None or not snapped:
        pids = np.full(n, INVALID_POINT3D, dtype=np.uint64)
    return IrMatchResult(
        np.asarray(px, float), np.asarray(xyz, float),
        np.asarray(pids, dtype=np.uint64), np.zeros(n),
    )


def test_pnp_recovers_pose_with_outliers():
    rec, xyz, pids = build_eo_rec()
    truth = cam_from_world((5, -3, ALT))
    px = project(IR, truth, xyz)
    rng = np.random.default_rng(4)
    px += rng.normal(0, 0.3, px.shape)
    n_out = 60  # 30% gross outliers
    px[:n_out] = rng.uniform(0, IR["width"], (n_out, 2))
    result = match_result(px, xyz, pids)

    pose, inliers = pnp_ir_image(result, ir_camera())
    assert inliers.sum() >= len(xyz) - n_out - 5
    assert np.linalg.norm(pose.translation - truth.translation) < 0.5
    dq = Rotation.from_quat(pose.rotation.quat) * Rotation.from_quat(
        truth.rotation.quat
    ).inv()
    # the scene is exactly planar, so tilt trades off against lateral
    # translation; single-image rotation is the noisy one (the pipeline
    # averages it over hundreds of frames)
    assert np.degrees(dq.magnitude()) < 0.5


def test_pnp_too_few_matches():
    rec, xyz, pids = build_eo_rec()
    truth = cam_from_world()
    result = match_result(project(IR, truth, xyz[:5]), xyz[:5], pids[:5])
    assert pnp_ir_image(result, ir_camera()) is None


def test_insert_ir_image_invariants(tmp_path):
    rec, xyz, pids = build_eo_rec()
    truth = cam_from_world((5, -3, ALT))
    result = match_result(project(IR, truth, xyz[:60]), xyz[:60], pids[:60])
    result.point3D_ids[20:30] = INVALID_POINT3D  # lifted but unsnapped
    inliers = np.ones(60, dtype=bool)
    inliers[:10] = False

    track_before = rec.points3D[int(pids[10])].track.length()
    cid = add_ir_camera(rec, ir_camera())
    image_id, num_obs = insert_ir_image(
        rec, cid, IR_NAME.format(i=1), result, inliers, truth
    )

    assert image_id not in (1, 2) and image_id in rec.images
    image = rec.images[image_id]
    assert image.has_pose
    assert num_obs == 40  # inlier and snapped
    assert image.num_points3D == 40
    assert rec.points3D[int(pids[10])].track.length() == track_before + 1
    assert rec.points3D[int(pids[0])].track.length() == track_before  # outlier
    assert rec.points3D[int(pids[25])].track.length() == track_before  # unsnapped

    rec.write(tmp_path)
    again = pycolmap.Reconstruction(tmp_path)
    assert again.num_reg_images() == rec.num_reg_images()
    assert again.images[image_id].num_points3D == 40


def test_insert_ir_image_dedupes_point_ids():
    rec, xyz, pids = build_eo_rec()
    truth = cam_from_world((5, -3, ALT))
    # the same 3D point matched twice; the closer snap must win
    px = project(IR, truth, xyz[:20])
    result = match_result(
        np.vstack([px, px[0] + [4.0, 0.0]]),
        np.vstack([xyz[:20], xyz[0]]),
        np.concatenate([pids[:20], pids[:1]]),
    )
    result.snap_dist[:] = 1.0
    result.snap_dist[-1] = 9.0
    cid = add_ir_camera(rec, ir_camera())
    image_id, num_obs = insert_ir_image(
        rec, cid, IR_NAME.format(i=1), result, np.ones(21, bool), truth
    )
    assert num_obs == 20
    image = rec.images[image_id]
    assert image.num_points3D == 20
    assert image.points2D[0].has_point3D()  # the 1.0-distance duplicate won
    assert not image.points2D[20].has_point3D()


def test_refine_fused_fixes_eo_and_improves_ir():
    rec, xyz, pids = build_eo_rec()
    truth = cam_from_world((5, -3, ALT))
    result = match_result(project(IR, truth, xyz), xyz, pids)
    cid = add_ir_camera(rec, ir_camera())
    # insert with a perturbed pose; perfect observations should pull it back
    off = pycolmap.Rigid3d(
        pycolmap.Rotation3d(Rotation.from_euler("y", 0.2, degrees=True).as_quat()),
        np.array([1.0, -0.5, 0.7]),
    )
    perturbed = pycolmap.Rigid3d(
        (off.rotation * truth.rotation), truth.translation + off.translation
    )
    image_id, _ = insert_ir_image(
        rec, cid, IR_NAME.format(i=1), result, np.ones(len(pids), bool), perturbed
    )
    eo_before = {i: rec.images[i].cam_from_world().matrix() for i in (1, 2)}
    err_before = np.linalg.norm(
        rec.images[image_id].cam_from_world().translation - truth.translation
    )

    refine_fused(rec, [image_id], ir_camera_ids=(cid,))

    for i in (1, 2):
        assert np.allclose(
            rec.images[i].cam_from_world().matrix(), eo_before[i], atol=1e-12
        )
    err_after = np.linalg.norm(
        rec.images[image_id].cam_from_world().translation - truth.translation
    )
    assert err_after < err_before / 2


def test_align_ir_to_eo_recovers_sim3():
    """Poses expressed in an IR world offset from the EO world by a known
    Sim3 must be recovered from the cross-modal correspondences, and the
    per-image aligned poses must reproduce the true EO-world poses."""
    xyz = ground_points(300, seed=6)
    S_R = Rotation.from_euler("xyz", [0.4, -0.7, 1.2], degrees=True)
    S_t = np.array([2.0, -1.0, 0.5])
    S_s = 1.0015  # X_ir = s * R @ X_eo + t

    rng = np.random.default_rng(7)
    entries, truths = [], []
    for center in [(0, 0, ALT), (40, 10, ALT), (-30, 25, ALT), (10, -40, ALT)]:
        truth = cam_from_world(center)  # cam <- EO world
        px = project(IR, truth, xyz) + rng.normal(0, 0.5, (len(xyz), 2))
        keep = (
            (px[:, 0] >= 0) & (px[:, 0] < IR["width"])
            & (px[:, 1] >= 0) & (px[:, 1] < IR["height"])
        )
        # the same camera pose, written in the IR world: substituting
        # X_eo = R_S^-1 (X_ir - t_S)/s into X_cam = R_c X_eo + t_c gives
        # the projection-equivalent rigid pose
        #   R_ir = R_c R_S^-1,  t_ir = s t_c - R_ir t_S
        Rc = truth.rotation.matrix()
        R_ir = Rc @ S_R.inv().as_matrix()
        t_ir = S_s * truth.translation - R_ir @ S_t
        pose_ir = pycolmap.Rigid3d(
            pycolmap.Rotation3d(Rotation.from_matrix(R_ir).as_quat()), t_ir
        )
        entries.append(AlignmentEntry(pose_ir, ir_camera(), px[keep], xyz[keep]))
        truths.append(truth)

    R, t, s, stats = align_ir_to_eo(entries)
    assert abs(s - S_s) < 5e-4
    # the tilt component shared by all (planar) views is the one weakly
    # constrained direction; 4 views with 0.5 px noise leave ~0.03 deg
    assert np.degrees((R * S_R.inv()).magnitude()) < 0.05
    assert stats["reproj_px_median"] < 1.5

    for entry, truth in zip(entries, truths):
        pose = aligned_ir_pose(entry.cam_from_irworld, R, t, s)
        dq = Rotation.from_quat(pose.rotation.quat) * Rotation.from_quat(
            truth.rotation.quat
        ).inv()
        assert np.degrees(dq.magnitude()) < 0.05
        assert np.linalg.norm(pose.translation - truth.translation) < 0.5


def test_mount_delta_deg():
    a = standard_camera(EO)
    b = standard_camera(EO)
    b.cam_quat = (Rotation.from_quat(a.cam_quat) * Rotation.from_euler(
        "z", 0.5, degrees=True
    )).as_quat()
    assert abs(mount_delta_deg(a, b) - 0.5) < 1e-6
    assert mount_delta_deg(a, a) < 1e-9


# ------------------------------------------------------- matching + fusion


def test_match_ir_image_lifts_and_snaps():
    rec, xyz, pids = build_eo_rec()
    eo_sc, ir_sc = standard_camera(EO), standard_camera(IR)
    eo_image = rec.images[1]

    H = warp_homography(eo_sc, 0.0, ir_sc, 0.0, 0.0)
    eo_px = eo_sc.project(xyz.T, 0.0).T
    warped_px = cv2.perspectiveTransform(eo_px[None], H)[0]
    ir_px = ir_sc.project(xyz.T, 0.0).T
    # one extra match halfway between keypoints: lifts (planar) but
    # should not snap to any track at 16 px
    mid_xyz = (xyz[0] + xyz[1]) / 2
    mid_eo = eo_sc.project(mid_xyz[:, None], 0.0).T
    kpts0 = np.vstack([warped_px, cv2.perspectiveTransform(mid_eo[None], H)[0]])
    kpts1 = np.vstack([ir_px, ir_sc.project(mid_xyz[:, None], 0.0).T])

    partner = EoPartner(
        1, eo_image.name, np.zeros((EO["height"], EO["width"]), np.uint8),
        0.0, eo_sc, eo_image, rec.cameras[1], snap_index(eo_image, rec),
    )
    ir_img = np.zeros((IR["height"], IR["width"]), np.uint8)
    result = match_ir_image(
        ir_img, ir_sc, 0.0, [partner], FakeMatcher(kpts0, kpts1), 0.0
    )

    assert len(result.ir_xy) == len(xyz) + 1  # everything lifted
    assert result.num_raw == len(xyz) + 1
    # depth-lifted 3D agrees with the true ground points (planar -> exact)
    assert np.max(np.linalg.norm(result.xyz[:-1] - xyz, axis=1)) < 0.05
    snapped = result.point3D_ids != INVALID_POINT3D
    assert snapped[:-1].all() and not snapped[-1]
    assert set(result.point3D_ids[:-1].astype(int)) == set(pids.astype(int))


def test_fuse_ir_into_eo_end_to_end(tmp_path):
    rec, xyz, pids = build_eo_rec()
    eo_sc, ir_sc = standard_camera(EO), standard_camera(IR)
    truth = cam_from_world()  # IR colocated with EO image 1's trigger

    # stage EO jpg + IR png where the orchestrator expects them
    dirs = {}
    for folder, spec in (("pre_center_rgb", EO), ("pre_center_ir", IR)):
        d = tmp_path / folder
        d.mkdir()
        dirs[folder] = str(d)
    xx, yy = np.meshgrid(np.arange(EO["width"]), np.arange(EO["height"]))
    eo_img = ((xx / EO["width"] + yy / EO["height"]) * 127).astype(np.uint8)
    for i in (1, 2):
        cv2.imwrite(
            os.path.join(dirs["pre_center_rgb"], EO_NAME.format(i=i).split("/")[1]),
            eo_img,
        )
    cv2.imwrite(
        os.path.join(dirs["pre_center_ir"], IR_NAME.format(i=1).split("/")[1]),
        np.zeros((IR["height"], IR["width"]), np.uint8),
    )

    # a minimal IR group model: intrinsics + one posed image
    ir_rec = pycolmap.Reconstruction()
    ir_rec.add_camera_with_trivial_rig(ir_camera(camera_id=1))
    ir_rec.add_image_with_trivial_frame(
        pycolmap.Image(
            name=IR_NAME.format(i=1), keypoints=np.empty((0, 2)),
            camera_id=1, image_id=1,
        ),
        truth,
    )

    H = warp_homography(eo_sc, 0.0, ir_sc, 0.0, 0.0)
    eo_px = eo_sc.project(xyz.T, 0.0).T
    matcher = FakeMatcher(
        cv2.perspectiveTransform(eo_px[None], H)[0],
        ir_sc.project(xyz.T, 0.0).T,
    )

    base = "pre_fl00_C_20260101_00000{i}.000000"
    times = {base.format(i=1): 1000.0, base.format(i=2): 1001.0}
    report = fuse_ir_into_eo(
        rec,
        ir_rec,
        models={"pre_center_rgb": eo_sc, "pre_center_ir": ir_sc},
        image_dirs=dirs,
        times=times,
        matcher=matcher,
        run_ba=True,
    )

    assert report["num_fused"] == 1 and not report["skipped"]
    assert report["bundle_adjusted"]
    assert rec.num_reg_images() == 3
    fused = [im for im in rec.images.values() if im.name.endswith("_ir.png")]
    assert len(fused) == 1
    assert np.linalg.norm(
        fused[0].cam_from_world().translation - truth.translation
    ) < 0.2
    assert report["per_folder"]["pre_center_ir"]["fused"] == 1

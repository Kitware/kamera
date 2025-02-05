import random
import numpy as np
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple
from scipy.optimize import minimize, fminbound
from scipy.spatial.transform import Rotation
import pycolmap
from kamera.colmap_processing.camera_models import StandardCamera
from kamera.sensor_models.nav_state import NavStateINSJson


@dataclass
class VisiblePoint:
    """Point visible in a camera view."""

    point_3d: np.ndarray  # 3D point coordinates (3,)
    point_3d_id: int  # Unique ID given by Colmap of the 3D point
    point_2d: np.ndarray  # 2D observation in image (2,)
    uncertainty: float  # Point uncertainty/error
    time: float
    visible: bool = False  # Whether point is visible in this camera


@dataclass
class RotationEstimate:
    """Result of rotation estimation."""

    quaternion: np.ndarray  # Estimated rotation quaternion (x,y,z,w)
    covariance: np.ndarray  # 3x3 covariance matrix in angle-axis space
    fisher_information: np.ndarray  # Fisher information matrix
    num_inliers: int  # Number of inlier observations used
    mean_error: float  # Mean error of the estimate


def compute_jacobian(points: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of rotation with respect to angle-axis parameters.

    Args:
        points: Nx3 array of 3D points
        R: 3x3 rotation matrix
    Returns:
        3Nx3 Jacobian matrix
    """
    jac = np.zeros((3 * len(points), 3))
    for i, p in enumerate(points):
        # Skew-symmetric matrix for cross product
        skew = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
        jac[3 * i : 3 * (i + 1)] = (-R @ skew)[0]  # First row
        jac[3 * i + 1 : 3 * (i + 2)] = (-R @ skew)[1]  # Second row
        jac[3 * i + 2 : 3 * (i + 3)] = (-R @ skew)[2]  # Third row
    return jac


def compute_fisher_information(
    points: np.ndarray, R: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Compute Fisher Information Matrix without forming full weight matrix."""
    J = compute_jacobian(points, R)
    fisher = np.zeros((3, 3))
    for i in range(len(weights)):
        block = J[3 * i : 3 * (i + 3)]
        fisher += weights[i] * (block.T @ block)
    return fisher


def compute_covariance(
    R: np.ndarray, observations: List[List[VisiblePoint]]
) -> np.ndarray:
    """
    Compute covariance matrix for rotation estimate.

    Args:
        R: 3x3 rotation matrix
        observations: List of visible points per camera
    Returns:
        3x3 covariance matrix in angle-axis space
    """
    points = []
    weights = []

    for camera_obs in observations:
        for obs in camera_obs:
            points.append(obs.point_3d)
            weights.append(1.0 / (obs.uncertainty**2 + 1e-10))

    points = np.array(points)
    weights = np.array(weights)

    fisher = compute_fisher_information(points, R, weights)
    return np.linalg.inv(fisher)


def weighted_horn_alignment_partial(
    observations: np.ndarray[List[VisiblePoint]],
    sfm_quats: np.ndarray,
    ins_quats: np.ndarray,
    min_points_per_image: int = 3,
    ransac_iters: int = 100,
    error_threshold: float = 0.1,
    max_3d_points: int = 5000,
) -> RotationEstimate:
    """
    Find rotation between INS and camera frames with partial point visibility.

    Args:
        observations: List of visible points for each camera
        sfm_quats: Nx4 array of SfM-derived camera quaternions
        ins_quats: Nx4 array of INS-measured quaternions
        min_points_per_image: Minimum points needed per camera
        ransac_iters: Number of RANSAC iterations
        error_threshold: Error threshold for inlier classification

    Returns:
        RotationEstimate containing quaternion and uncertainty information
    """

    print(f"Number of observations: {len(observations)}")
    print(f"Number of sfm quats: {len(sfm_quats)}, {len(ins_quats)}")

    def compute_rotation_for_subset(
        image_indices: List[int], max_3d_points: int = 1000
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[List[int]]]:
        """Compute rotation for a subset of images."""
        points_ins = []
        points_cam = []
        weights = []
        inlier_indices = []

        for idx, image_idx in enumerate(image_indices):
            # Get visible points for this camera
            visible_obs = [obs for obs in observations[image_idx] if obs.visible]

            if len(visible_obs) < min_points_per_image:
                print("Not enough visible observations, skipping.")
                return None, None, None

            # errors = [obs.uncertainty for obs in visible_obs]
            # keep the smallest error points
            # keep_idx = np.argsort(errors)[:max_3d_points]
            # visible_obs = np.array(visible_obs)[keep_idx]

            R_ins = Rotation.from_quat(ins_quats[image_idx])
            R_sfm = Rotation.from_quat(sfm_quats[image_idx])

            for obs in visible_obs:
                # Transform point through both rotations
                p_ins = R_ins.apply(obs.point_3d)
                p_cam = R_sfm.apply(obs.point_3d)

                points_ins.append(p_ins)
                points_cam.append(p_cam)
                weights.append(1.0 / (obs.uncertainty**2 + 1e-10))
                inlier_indices.append(idx)

        if len(points_ins) < 5:  # Need minimum points for reliable estimate
            print(
                f"Number of points is {points_ins}, which is less than the min of 5 needed."
            )
            return None, None, None

        points_ins = np.array(points_ins)
        points_cam = np.array(points_cam)
        weights = np.array(weights)

        # Weighted Horn method
        centroid_ins = np.sum(weights[:, None] * points_ins, axis=0) / np.sum(weights)
        centroid_cam = np.sum(weights[:, None] * points_cam, axis=0) / np.sum(weights)

        centered_ins = points_ins - centroid_ins
        centered_cam = points_cam - centroid_cam

        H = (centered_ins.T * weights) @ centered_cam
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        errors = np.linalg.norm(R @ centered_ins.T - centered_cam.T, axis=0)

        # Identify inliers
        inliers = errors < error_threshold
        if np.sum(inliers) < min_points_per_image:
            print(
                f"Num inliers is {np.sum(inliers)}, which is less than the min required of {min_points_per_image}."
            )
            print(f"Mean error is {np.mean(errors)}, threshold is {error_threshold}.")
            return None, None, None

        mean_error = np.mean(errors[inliers])
        return R, mean_error, [inlier_indices[i] for i in np.where(inliers)[0]]

    best_R = None
    best_error = float("inf")
    best_inliers = []
    n_images = len(observations)

    for _ in range(ransac_iters):
        # Sample subset of images
        n_sample = min(5, n_images)
        image_indices = np.random.choice(n_images, n_sample, replace=False)

        R, error, inliers = compute_rotation_for_subset(
            image_indices, max_3d_points=max_3d_points
        )
        if R is not None and error < best_error:
            # Verify with all images
            R_full, error_full, inliers_full = compute_rotation_for_subset(
                inliers,
            )
            if R_full is not None and error_full < best_error:
                best_R = R_full
                best_error = error_full
                best_inliers = inliers_full

    if best_R is None:
        raise RuntimeError("Could not find valid rotation - check visibility")

    print(best_R)

    # Compute uncertainty
    # covariance = compute_covariance(best_R, observations)
    # fisher = compute_fisher_information(
    #    np.array(
    #        [
    #            obs.point_3d
    #            for i in best_inliers
    #            for obs in observations[i]
    #            if obs.visible
    #        ]
    #    ),
    #    best_R,
    #    np.array(
    #        [
    #            1.0 / (obs.uncertainty**2 + 1e-10)
    #            for i in best_inliers
    #            for obs in observations[i]
    #            if obs.visible
    #        ]
    #    ),
    # )
    covariance = np.eye(3)
    fisher = np.eye(3)

    return RotationEstimate(
        quaternion=Rotation.from_matrix(best_R).as_quat(),
        covariance=covariance,
        fisher_information=fisher,
        num_inliers=len(best_inliers),
        mean_error=best_error,
    )


def analyze_rotation_estimate(estimate: RotationEstimate) -> None:
    """
    Analyze and print information about rotation estimate quality.

    Args:
        estimate: RotationEstimate from alignment
    """
    # Extract principal uncertainties
    eigenvals, eigenvecs = np.linalg.eigh(estimate.covariance)
    std_devs = np.sqrt(eigenvals)

    print("\nRotation Alignment Analysis")
    print("==========================")
    print(f"Quaternion (x,y,z,w): {estimate.quaternion}")
    print(f"\nNumber of inliers: {estimate.num_inliers}")
    print(f"Mean error: {estimate.mean_error:.6f}")

    print("\nUncertainty Analysis:")
    print("Standard deviations (degrees):")
    for i, std in enumerate(std_devs):
        print(f"  Axis {i+1}: {np.degrees(std):.4f}°")

    # Condition number of Fisher Information
    cond = np.linalg.cond(estimate.fisher_information)
    print(f"\nFisher Information condition number: {cond:.2e}")

    # 99% confidence intervals
    conf_intervals = 2.576 * std_devs
    print("\n99% Confidence intervals (degrees):")
    for i, interval in enumerate(conf_intervals):
        print(f"  Axis {i+1}: ±{np.degrees(interval):.4f}°")


class RANSACParams:
    def __init__(
        self,
        min_samples: int = 3,
        max_iterations: int = 1000,
        inlier_threshold: float = 0.1,  # radians
        min_inliers_ratio: float = 0.5,
    ):
        self.min_samples = min_samples
        self.max_iterations = max_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inliers_ratio = min_inliers_ratio


def compute_rotation_horn(
    sfm_rotations: np.ndarray, ins_rotations: np.ndarray
) -> np.ndarray:
    """
    Compute optimal rotation matrix using Horn's method for a subset of rotations.

    Args:
        sfm_rotations: Array of SfM rotation matrices (N, 3, 3)
        ins_rotations: Array of INS rotation matrices (N, 3, 3)

    Returns:
        optimal_rotation: 3x3 rotation matrix
    """
    # Build correlation matrix
    M = sum(ins_R @ sfm_R.T for sfm_R, ins_R in zip(sfm_rotations, ins_rotations))

    # Perform SVD
    U, _, Vt = np.linalg.svd(M)

    # Ensure proper rotation (det = 1)
    det = np.linalg.det(U @ Vt)
    S = np.eye(3)
    if det < 0:
        S[2, 2] = -1

    return U @ S @ Vt


def compute_alignment_errors(
    sfm_rotations: np.ndarray, ins_rotations: np.ndarray, optimal_rotation: np.ndarray
) -> np.ndarray:
    """
    Compute angular errors between aligned rotations.

    Args:
        sfm_rotations: Array of SfM rotation matrices
        ins_rotations: Array of INS rotation matrices
        optimal_rotation: Computed optimal rotation matrix

    Returns:
        errors: Array of angular errors in radians
    """
    errors = []
    for sfm_R, ins_R in zip(sfm_rotations, ins_rotations):
        aligned_R = optimal_rotation @ sfm_R
        relative_R = aligned_R.T @ ins_R
        angle_error = abs(Rotation.from_matrix(relative_R).magnitude())
        errors.append(angle_error)
    return np.array(errors)


def register_camera_horn_ransac(
    sfm_poses: List[np.ndarray],
    ins_poses: List[np.ndarray],
    points_per_image: Optional[List] = None,
    colmap_camera: Optional[object] = None,
    ransac_params: Optional[RANSACParams] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Register camera poses using Horn's method with RANSAC for robust estimation.

    Args:
        sfm_poses: List of quaternions from Structure from Motion
        ins_poses: List of quaternions from INS measurements
        points_per_image: Optional list of 2D-3D point correspondences
        colmap_camera: Optional COLMAP camera object
        ransac_params: Optional RANSAC parameters

    Returns:
        optimal_rotation: 3x3 rotation matrix
        best_rmse: Root mean square error of the alignment
        inlier_mask: Boolean array indicating inlier poses
    """
    if ransac_params is None:
        ransac_params = RANSACParams()

    # Convert quaternions to rotation matrices
    sfm_rotations = np.array([Rotation.from_quat(q).as_matrix() for q in sfm_poses])
    ins_rotations = np.array([Rotation.from_quat(q).as_matrix() for q in ins_poses])

    num_poses = len(sfm_poses)
    best_rotation = None
    best_rmse = float("inf")
    best_inlier_mask = None

    # RANSAC iterations
    for _ in range(ransac_params.max_iterations):
        # Randomly sample pose pairs
        sample_indices = random.sample(range(num_poses), ransac_params.min_samples)
        sample_sfm = sfm_rotations[sample_indices]
        sample_ins = ins_rotations[sample_indices]

        # Compute candidate rotation using sampled pairs
        candidate_rotation = compute_rotation_horn(sample_sfm, sample_ins)

        # Compute errors for all poses using this rotation
        errors = compute_alignment_errors(
            sfm_rotations, ins_rotations, candidate_rotation
        )

        # Identify inliers
        print(np.mean(errors))
        inlier_mask = errors < ransac_params.inlier_threshold
        num_inliers = np.sum(inlier_mask)

        # Check if we have enough inliers
        if num_inliers / num_poses >= ransac_params.min_inliers_ratio:
            # Recompute rotation using all inliers
            inlier_sfm = sfm_rotations[inlier_mask]
            inlier_ins = ins_rotations[inlier_mask]
            refined_rotation = compute_rotation_horn(inlier_sfm, inlier_ins)

            # Compute RMSE for refined rotation
            refined_errors = compute_alignment_errors(
                sfm_rotations, ins_rotations, refined_rotation
            )
            rmse = np.sqrt(np.mean(refined_errors[inlier_mask] ** 2))

            # Update best solution if this is better
            if rmse < best_rmse:
                best_rotation = refined_rotation
                best_rmse = rmse
                best_inlier_mask = inlier_mask

    if best_rotation is None:
        raise RuntimeError(
            "RANSAC failed to find a good alignment. Consider adjusting parameters."
        )

    # Verify alignment using 3D points if available
    if points_per_image and colmap_camera:
        verify_alignment(points_per_image, best_rotation, colmap_camera)

    # Print some statistics about the solution
    inlier_percentage = np.sum(best_inlier_mask) / len(best_inlier_mask) * 100
    print("RANSAC statistics:")
    print(f"- Inlier percentage: {inlier_percentage:.1f}%")
    print(f"- Final RMSE: {best_rmse:.3f} radians")

    return best_rotation, best_rmse, best_inlier_mask


def verify_alignment(
    points_per_image: List, optimal_rotation: np.ndarray, colmap_camera: object
) -> None:
    """
    Verify alignment quality using 3D point reprojection.
    """
    reprojection_errors = []

    for points in points_per_image:
        for pt in points:
            point_2d = pt.image_point
            point_3d = pt.point_3d
            # Transform 3D point using optimal rotation
            transformed_point = optimal_rotation @ point_3d

            # Project 3D point using camera intrinsics
            fx, fy, cx, cy = colmap_camera.params[:4]  # Assuming standard pinhole model

            projected_x = fx * transformed_point[0] / transformed_point[2] + cx
            projected_y = fy * transformed_point[1] / transformed_point[2] + cy

            error = np.sqrt(
                (projected_x - point_2d[0]) ** 2 + (projected_y - point_2d[1]) ** 2
            )
            reprojection_errors.append(error)

    mean_reprojection_error = np.mean(reprojection_errors)
    print(f"Mean reprojection error: {mean_reprojection_error:.2f} pixels")


def iterative_alignment(
    sfm_quats: List[np.ndarray],
    ins_quats: List[np.ndarray],
    points_per_image: List[List[VisiblePoint]],
    colmap_camera: pycolmap._core.Camera,
    nav_state_provider: NavStateINSJson,
) -> StandardCamera:
    # Both quaternions are of the form (x, y, z, w) and represent a coordinate
    # system rotation.
    cam_quats = [
        (Rotation.from_quat(sfm_quats[k]) * Rotation.from_quat(ins_quats[k]))
        .inv()
        .as_quat()
        for k in range(len(ins_quats))
    ]

    K = colmap_camera.calibration_matrix()
    if colmap_camera.model.name == "OPENCV":
        fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params
    elif colmap_camera.model.name == "SIMPLE_RADIAL":
        d1 = d2 = d3 = d4 = 0
    elif colmap_camera.model.name == "PINHOLE":
        d1 = d2 = d3 = d4 = 0
    else:
        raise SystemError(f"Unexpected camera model found: {colmap_camera.model.name}")
    dist = np.array([d1, d2, d3, d4])

    organized_points_per_frame = []
    max_uncertainty = np.inf
    errs = []
    for pts in points_per_image:
        xy = []
        xyz = []
        for pt in pts:
            errs.append(pt.uncertainty)
            if pt.uncertainty < max_uncertainty:
                xy.append(pt.point_2d)
                xyz.append(pt.point_3d)

        # Eliminate points 10x the distance away from median
        max_uncertainty = np.median(errs) * 10
        # time is the same for all points, since it's a single frame
        xy = np.array(xy)
        xyz = np.array(xyz)
        organized_points_per_frame.append((xy, xyz, pts[0].time))

    def cam_quat_error(cam_quat: np.ndarray) -> float:
        cam_quat = cam_quat / np.linalg.norm(cam_quat)
        camera_model = StandardCamera(
            colmap_camera.width,
            colmap_camera.height,
            K,
            dist,
            [0, 0, 0],
            cam_quat,
            platform_pose_provider=nav_state_provider,
        )
        err = []

        # Update uncertainty based on errors
        # check quaternion alignment over all frames
        for xys, xyzs, t in organized_points_per_frame:
            xys = []
            xyzs = []
                    
            xys = np.array(xys)
            xyzs = np.array(xyzs)
            # Error in meters.
            # Rays coming out of the camera in the direction of the imaged points.
            ray_pos, ray_dir = camera_model.unproject(xys.T, t)

            # Direction coming out of the camera pointing at the actual 3-D points'
            # locations.
            ray_dir2 = xyzs.T - ray_pos
            d = np.sqrt(np.sum((ray_dir2) ** 2, axis=0))
            ray_dir2 /= d

            dp = np.minimum(np.sum(ray_dir * ray_dir2, axis=0), 1)
            dp = np.maximum(dp, -1)
            theta = np.arccos(dp)
            err_ = np.sin(theta) * d
            # err.append(np.percentile(err_, 90))
            err.append(np.median(err_))

        # print("Average uncertainty: ")
        # print(np.mean(uncertainties))
        # err = err[err < np.percentile(err, 90)]

        err = np.median(err)
        # print('RMS reproject error for quat', cam_quat, ': %0.8f' % err)
        return err

    print("Iterating through %s quaternion guesses." % len(cam_quats))
    random.shuffle(cam_quats)
    best_quat = None
    best_err = np.inf
    for cam_quat in cam_quats:
        err = cam_quat_error(cam_quat)
        if err < best_err:
            best_err = err
            best_quat = cam_quat

        if best_err < 5:
            break

    print("Best error: ", best_err)
    print("Best quat: ")
    print(cam_quat)

    print("Minimizing error over camera quaternions")

    ret = minimize(cam_quat_error, best_quat)
    best_quat = ret.x / np.linalg.norm(ret.x)
    ret = minimize(cam_quat_error, best_quat, method="BFGS")
    best_quat = ret.x / np.linalg.norm(ret.x)
    ret = minimize(cam_quat_error, best_quat, method="Powell")
    best_quat = ret.x / np.linalg.norm(ret.x)

    # Sequential 1-D optimizations.
    for i in range(4):

        def set_x(x):
            quat = best_quat.copy()
            quat = quat / np.linalg.norm(quat)
            while abs(quat[i] - x) > 1e-6:
                quat[i] = x
                quat = quat / np.linalg.norm(quat)

            return quat

        def func(x):
            return cam_quat_error(set_x(x))

        x = np.linspace(-1, 1, 100)
        x = sorted(np.hstack([x, best_quat[i]]))
        y = [func(x_) for x_ in x]
        x = fminbound(func, x[np.argmin(y) - 1], x[np.argmin(y) + 1], xtol=1e-8)
        best_quat = set_x(x)

    camera_model = StandardCamera(
        colmap_camera.width,
        colmap_camera.height,
        K,
        dist,
        [0, 0, 0],
        best_quat,
        platform_pose_provider=nav_state_provider,
    )

    return camera_model


def transfer_alignment(
    colmap_camera: pycolmap._core.Camera,
    calibrated_camera: pycolmap._core.Camera,
    nav_state_provider: NavStateINSJson,
    points_per_image: List[List[VisiblePoint]],
    calibrated_camera_model: StandardCamera,
) -> StandardCamera:
    # Both quaternions are of the form (x, y, z, w) and represent a coordinate
    # system rotation.
    skipped = 0
    total = 0
    im_pts_uv = []
    im_pts_rgb = []

    organized_points_per_frame = []
    for pts in points_per_image:
        max_uncertainty = 1
        xy = []
        xyz = []
        xyz_ids = []
        for pt in pts:
            if pt.uncertainty < max_uncertainty:
                xy.append(pt.point_2d)
                xyz.append(pt.point_3d)
                xyz_ids.append(pt.point_3d_id)
        # time is the same for all points, since it's a single frame
        xy = np.array(xy)
        xyz = np.array(xyz)
        organized_points_per_frame.append((xy, xyz, xyz_ids, pts[0].time))
    return

    # Build up pairs of image coordinates between the two cameras from image
    # pairs acquired from the same time.
    # for pts in zip(points_per_image):
    #    xys1, xyzs1, xyz1_ids, t1 = pts

    #    for _id in xyz1_ids:
    #        if calibrated_camera.has_point3D(_id):

    #    xys2, xyzs2, t2 = colocated_pts
    #    base_name = get_base_name(image_uv.name)

    #    try:
    #        t1 = basename_to_time[base_name]
    #    except KeyError:
    #        print(f"No time found for {base_name}.")
    #        continue

    #    try:
    #        image_rgb = time_to_modality[t1]["rgb"]
    #    except KeyError:
    #        print(f"No rgb image found at {t1}.")
    #        continue

    #    # Both 'uv_image' and 'image_rgb' are from the same time.
    #    pt_ids1 = image_uv.point3D_ids
    #    ind = pt_ids1 != -1
    #    xys1 = dict(zip(pt_ids1[ind], image_uv.xys[ind]))

    #    pt_ids2 = image_rgb.point3D_ids
    #    ind = pt_ids2 != -1
    #    xys2 = dict(zip(pt_ids2[ind], image_rgb.xys[ind]))

    #    match_ids = set(xys1.keys()).intersection(set(xys2.keys()))
    #    total += 1
    #    if len(match_ids) < 1:
    #        # print("No match IDs found.")
    #        skipped += 1
    #        continue

    #    for match_id in match_ids:
    #        im_pts_uv.append(xys1[match_id])
    #        im_pts_rgb.append(xys2[match_id])

    # print(
    #    f"Matched {total-skipped}/{total} image pairs, resulting in "
    #    f"{len(im_pts_uv)} matching UV and RGB points."
    # )

    # im_pts_uv = np.array(im_pts_uv)
    # im_pts_rgb = np.array(im_pts_rgb)
    ## Arbitrary cut off
    # minimum_pts_required = 10
    # if len(im_pts_rgb) < minimum_pts_required or len(im_pts_uv) < minimum_pts_required:
    #    print(
    #        "[ERROR] Not enough matching RGB/UV image points were found "
    #        f"for camera {uv_str}."
    #    )
    #    continue

    # if False:
    #    plt.subplot(121)
    #    plt.plot(im_pts_uv[:, 0], im_pts_uv[:, 1], "ro")
    #    plt.subplot(122)
    #    plt.plot(im_pts_rgb[:, 0], im_pts_rgb[:, 1], "bo")

    ## Treat as co-located cameras (they are) and unproject out of RGB and into
    ## the other camera.
    # ray_pos, ray_dir = cm_rgb.unproject(im_pts_rgb.T)
    # wrld_pts = ray_dir.T * 1e4
    # assert np.all(np.isfinite(wrld_pts)), "World points contain non-finite values."

    # colmap_camera = camera_from_camera_str[uv_str]

    # if colmap_camera.model == "OPENCV":
    #    fx, fy, cx, cy, d1, d2, d3, d4 = colmap_camera.params
    # elif colmap_camera.model == "PINHOLE":
    #    fx, fy, cx, cy = colmap_camera.params
    #    d1 = d2 = d3 = d4 = 0

    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # dist = np.array([d1, d2, d3, d4], dtype=np.float32)

    # flags = cv2.CALIB_ZERO_TANGENT_DIST
    # flags = flags | cv2.CALIB_USE_INTRINSIC_GUESS
    # flags = flags | cv2.CALIB_FIX_PRINCIPAL_POINT
    # flags = flags | cv2.CALIB_FIX_K1
    # flags = flags | cv2.CALIB_FIX_K2
    # flags = flags | cv2.CALIB_FIX_K3
    # flags = flags | cv2.CALIB_FIX_K4
    # flags = flags | cv2.CALIB_FIX_K5
    # flags = flags | cv2.CALIB_FIX_K6

    # criteria = (
    #    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    #    30000,
    #    0.0000001,
    # )

    # ret = cv2.calibrateCamera(
    #    [wrld_pts.astype(np.float32)],
    #    [im_pts_uv.astype(np.float32)],
    #    (colmap_camera.width, colmap_camera.height),
    #    cameraMatrix=K.copy(),
    #    distCoeffs=dist.copy(),
    #    flags=flags,
    #    criteria=criteria,
    # )

    # err, _, _, rvecs, tvecs = ret

    # R = np.identity(4)
    # R[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
    # cam_quat = quaternion_from_matrix(R.T)

    ## Only optimize 3/4 components of the quaternion.
    # static_quat_ind = np.argmax(np.abs(cam_quat))
    # dynamic_quat_ind = [i for i in range(4) if i != static_quat_ind]
    ## static_quat_ind = 3            # Fixing the 'w' component
    ## dynamic_quat_ind = [0, 1, 2]   # Optimizing 'x', 'y', 'z' components
    # dynamic_quat_ind = np.array(dynamic_quat_ind)
    # cam_quat = np.asarray(cam_quat)
    # cam_quat /= np.linalg.norm(cam_quat)
    # x0 = cam_quat[dynamic_quat_ind].copy()  # [x, y, z]

    # def get_cm(x):
    #    """
    #    Create a camera model with updated quaternion and intrinsic parameters.

    #    Parameters:
    #    - x: array-like, shape (N,)
    #        Optimization variables where the first 3 elements correspond to
    #        the dynamic quaternion components ('x', 'y', 'z'), optionally
    #        followed by intrinsic parameters ('fx', 'fy', etc.).

    #    Returns:
    #    - cm: StandardCamera instance
    #        Updated camera model with new parameters.
    #    """
    #    # Ensure 'x' has at least 3 elements for quaternion
    #    assert (
    #        len(x) > 2
    #    ), "Optimization variable 'x' must have at least 3 elements for quaternion."

    #    # Validate 'x[:3]' are finite numbers
    #    assert np.all(
    #        np.isfinite(x[:3])
    #    ), "Quaternion components contain non-finite values."

    #    # Initialize quaternion with fixed 'w' component
    #    cam_quat_new = np.ones(4)

    #    # Assign dynamic components from optimization variables
    #    cam_quat_new[dynamic_quat_ind] = x[:3]

    #    # Normalize to ensure it's a unit quaternion
    #    norm = np.linalg.norm(cam_quat_new)
    #    assert norm > 1e-6, "Quaternion has zero or near-zero magnitude."
    #    cam_quat_new /= norm

    #    # Extract intrinsic parameters
    #    if len(x) > 3:
    #        fx_ = x[3]
    #        fy_ = x[4]
    #    else:
    #        fx_ = fx
    #        fy_ = fy

    #    if len(x) > 5:
    #        dist_ = x[5:]
    #    else:
    #        dist_ = dist

    #    # Construct the intrinsic matrix
    #    K = np.array([[fx_, 0, cx], [0, fy_, cy], [0, 0, 1]])

    #    # Create the camera model
    #    cm = StandardCamera(
    #        colmap_camera.width,
    #        colmap_camera.height,
    #        K,
    #        dist_,
    #        [0, 0, 0],
    #        cam_quat_new,
    #        platform_pose_provider=nav_state_fixed,
    #    )
    #    return cm

    # def error(x):
    #    try:
    #        cm = get_cm(x)
    #        projected_uv = cm.project(wrld_pts.T).T  # Shape: (N, 2)

    #        # Compute Euclidean distances
    #        err = np.sqrt(np.sum((im_pts_uv - projected_uv) ** 2, axis=1))

    #        # Apply Huber loss
    #        delta = 20
    #        ind = err < delta
    #        err[ind] = err[ind] ** 2
    #        err[~ind] = 2 * (err[~ind] - delta / 2) * delta

    #        # Sort and trim the error
    #        err = sorted(err)[: len(err) - len(err) // 5]

    #        # Compute mean error
    #        mean_err = np.sqrt(np.mean(err))

    #        # Add regularization term (e.g., L2 penalty)
    #        reg_strength = 1e-3  # Adjust as needed
    #        reg_term = reg_strength * np.linalg.norm(x[:3]) ** 2

    #        total_error = mean_err + reg_term
    #        return total_error
    #    except Exception as e:
    #        print(f"Error in error function: {e}")
    #        return np.inf  # Assign a high error if computation fails

    ## Optional: Define a callback function to monitor optimization
    # def callback(xk):
    #    try:
    #        cm = get_cm(xk)
    #        projected_uv = cm.project(wrld_pts.T).T
    #        err = np.sqrt(np.sum((im_pts_uv - projected_uv) ** 2, axis=1))
    #        mean_err = np.mean(err)
    #        print(f"Current x: {xk}, Mean Error: {mean_err}")
    #    except Exception as e:
    #        print(f"Error in callback: {e}")

    # def plot_results1(x):
    #    cm = get_cm(x)
    #    err = np.sqrt(np.sum((im_pts_uv - cm.project(wrld_pts.T).T) ** 2, 1))
    #    err = sorted(err)
    #    plt.plot(np.linspace(0, 100, len(err)), err)

    # print("Optimizing error for UV models.")
    # x = x0.copy()
    ## Example bounds for [x, y, z] components
    # bounds = [
    #    (-1.0, 1.0),  # x
    #    (-1.0, 1.0),  # y
    #    (-1.0, 1.0),
    # ]  # z
    # print("First pass")
    ## Perform optimization on [x, y, z]
    # ret = minimize(
    #    error,
    #    x,
    #    method="L-BFGS-B",
    #    bounds=bounds,
    #    callback=None,  # Optional: Monitor progress
    #    options={"disp": False, "maxiter": 30000, "ftol": 1e-7},
    # )
    # assert ret.success, "Minimization of UV error failed."
    # x = np.hstack([ret.x, fx, fy])
    # print("Second pass")
    # assert np.all(np.isfinite(x)), "Input quaternion with locked fx, fy, is not finite."
    # ret = minimize(error, x, method="Powell")
    # x = ret.x
    # print("Third pass")
    # assert np.all(np.isfinite(x)), "Input quaternion for BFGS is not finite."
    # ret = minimize(error, x, method="BFGS")

    # print("Final pass")
    # if True:
    #    x = np.hstack([ret.x, dist])
    #    ret = minimize(error, x, method="Powell")
    #    x = ret.x
    #    ret = minimize(error, x, method="BFGS")
    #    x = ret.x

    # assert np.all(np.isfinite(x)), "Input quaternion for final model is not finite."
    # cm_uv = get_cm(x)
    # cm_uv.save_to_file("%s/%s.yaml" % (save_dir, uv_str))


if __name__ == "__main__":
    pass

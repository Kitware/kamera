# -----------------------------------------------------------------------------
try:
    # Use ROS tf.transformations for rotation operations.
    from tf.transformations import euler_from_quaternion, quaternion_multiply, \
        quaternion_from_matrix, quaternion_matrix, quaternion_from_euler, \
        quaternion_inverse, euler_matrix
except ImportError:
    # Instead, use the pip installed transformations.py, which isn't compatible
    # with Python 2. However, this requires some modifications to the
    # formatting.
    import transformations

    # transformations assumes a (w, x, y, z) quaterion, but the rest of the module
    # was originally written to comply with ROS tf2 operations, which assume
    # (x, y, z, w) quaternions. So, these functions manage the conversions.
    # TODO, update the constituent methods to directly use the transformations
    # package to remove these extra steps.

    def quat_xyzw_to_wxyz(quat):
        return quat[3], quat[0], quat[1], quat[2]

    def quat_wxyz_to_xyzw(quat):
        return quat[1], quat[2], quat[3], quat[0]

    def quaternion_matrix(quat):
        return transformations.quaternion_matrix(quat_xyzw_to_wxyz(quat))

    def quaternion_multiply(quat1, quat2):
        quat1 = quat_xyzw_to_wxyz(quat1)
        quat2 = quat_xyzw_to_wxyz(quat2)
        quat = transformations.quaternion_multiply(quat1, quat2)
        return quat_wxyz_to_xyzw(quat)

    def quaternion_from_matrix(R):
        if R.shape != (4, 4):
            R_ = R
            R = np.identity(4)
            R[:3, :3] = R_

        return quat_wxyz_to_xyzw(transformations.quaternion_from_matrix(R))

    def euler_from_quaternion(quat, axes='sxyz'):
        quat = quat_xyzw_to_wxyz(quat)
        return transformations.euler_from_quaternion(quat, axes=axes)

    def quaternion_from_euler(x, y, z, axes):
        return quat_wxyz_to_xyzw(transformations.quaternion_from_euler(x, y, z, axes=axes))

    def quaternion_inverse(quat):
        quat = quat_xyzw_to_wxyz(quat)
        quat = transformations.quaternion_inverse(quat)
        return quat_wxyz_to_xyzw(quat)

    def quaternion_slerp(quat1, quat2, weight, spin, shortestpath):
        quat1 = quat_xyzw_to_wxyz(quat1)
        quat2 = quat_xyzw_to_wxyz(quat2)
        quat = transformations.quaternion_slerp(quat1, quat2, weight,
                                                spin, shortestpath)
        return quat_wxyz_to_xyzw(quat)
# -----------------------------------------------------------------------------

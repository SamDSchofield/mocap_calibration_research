import copy
import numpy as np
from os import listdir
from os.path import isfile, join

import rospy
import tf2_ros
from tf import transformations
import cv2


def Ts_to_tfs(Ts):
    calibration_tfs = []
    for transform in Ts:
        _, _, rotation, translation, _ = transformations.decompose_matrix(transform)
        calibration_tfs.append(np.append(translation, rotation))
    return calibration_tfs


def tf_to_mat(transform):
    position, orientation = transform
    mat = transformations.quaternion_matrix(orientation)
    mat[:3, 3] = np.array(position)
    return mat


def tf_stamped_to_mat(transform):
    position, orientation = transform.transform.translation, transform.transform.rotation
    orientation = np.array((orientation.x, orientation.y, orientation.z, orientation.w))
    mat = transformations.quaternion_matrix(orientation)
    position = np.array((position.x, position.y, position.z))
    mat[:3, 3] = position
    return mat


# From https://eng-git.canterbury.ac.nz/mje/enmt482_assignment2_2017/scripts/import_bag.py
def load_tf_history_from_bag(bag, max_duration=100000, additional_static_tfs=()):
    """Load all the transforms from the bag to use later."""
    buffer_ = tf2_ros.Buffer(cache_time=rospy.Duration(1000000), debug=False)
    for topic, message, t in bag.read_messages(['/tf', '/tf_static']):
        for transform in message.transforms:
            buffer_.set_transform(transform, '')

        for transform in additional_static_tfs:
            transform.header.stamp = t
            buffer_.set_transform(transform, '')

    # Repeat tf_static transforms at the end time, so they actually work
    last_stamp = transform.header.stamp
    for topic, message, time in bag.read_messages(['/tf_static']):
        for t in message.transforms:
            transform = copy.deepcopy(t)
            transform.header.stamp = last_stamp
            buffer_.set_transform(transform, '')
    for transform in additional_static_tfs:
        transform.header.stamp = time
        buffer_.set_transform(transform, '')

    return buffer_


def list_bags_in_dir(path):
    bags = ["{}/{}".format(path, f) for f in listdir(path) if isfile(join(path, f)) and f.endswith(".bag")]
    return bags


def list_bag_files(base_path, directories):
    bags_strings = []
    for _dir in directories:
        path = "{}/{}".format(base_path, _dir)
        bags_strings += list_bags_in_dir(path)
    return bags_strings


def apply_T(T, points):
    """Convert an array of 3D points into homogeneous coords, left-multiply by T, then convert back."""
    flipped = False
    if points.shape[0] != 3:
        assert points.shape[1] == 3, "Points must be 3xN or Nx3"
        points = points.T
        flipped = True
    points_h = np.vstack((points, np.ones_like(points[0, :])))
    points_transformed_h = np.dot(T, points_h)
    points_transformed = points_transformed_h[:-1]
    if flipped:
        return points_transformed.T
    return points_transformed


def se3_to_T(rvec, tvec):
    """Convert an se(3) OpenCV pose to a 4x4 transformation matrix.

    OpenCV poses are an Euler-Rodrigues rotation vector and a translation vector.
    """
    R, _ = cv2.Rodrigues(rvec)
    euler_angles = transformations.euler_from_matrix(R)
    return transformations.compose_matrix(angles=euler_angles, translate=np.squeeze(tvec))


def T_to_se3(T):
    """Convert a 4x4 transformation matrix to an se(3) OpenCV pose."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3]
    return rvec, tvec


def tf_to_T(tf):
    """Convert a (tx, ty, tz, rx, ry, rz) tuple to a 4x4 transformation matrix."""
    if len(tf) == 6:
        # Single transform
        return transformations.compose_matrix(angles=tf[3:], translate=tf[:3])
    else:
        # Array of transforms
        assert tf.shape[1] == 6, "Transform array must be Nx6"
        return np.vstack([tf_to_T(row) for row in tf])


def T_to_tf(T):
    """Convert a 4x4 transformation matrix to a (tx, ty, tz, rx, ry, rz) tuple."""
    if T.shape == (4, 4):
        # Single transform
        _, _, rot, trans, _ = transformations.decompose_matrix(T)
        return np.hstack((trans, rot))
    else:
        # Array of transforms
        assert T.shape[1:] == (4, 4), "Transform array must be Nx4x4"
        return np.array([T_to_tf(row) for row in T])


def project_markers(T_object_camera, markers, camera_matrix, distortion_coeffs):
    """"""
    T_camera_object = np.linalg.inv(T_object_camera)
    rvec, tvec = T_to_se3(T_camera_object)
    markers2d, _ = cv2.projectPoints(markers, rvec, tvec, camera_matrix, distortion_coeffs)
    return np.squeeze(markers2d)


def reprojection_error_vector(T, markers, image_coords, camera_matrix, distortion_coeffs, debug=False):
    """Return the reprojection error for each marker as a vector."""
    projected_coords = project_markers(T, markers, camera_matrix, distortion_coeffs)
    difference = projected_coords - image_coords
    squared_distance = np.linalg.norm(difference, axis=1)
    return squared_distance


def reprojection_error_mean(*args, **kwargs):
    """Return the mean reprojection error."""
    return np.mean(reprojection_error_vector(*args, **kwargs))
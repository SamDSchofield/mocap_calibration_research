"""
Script for finding the misalignment between a camera rigid-body {r} and a camera base_link {c}

The following notation is use:
* A transform from one reference frame {a} to another {b} will have the variable name t_ab
* These reference frames are assigned the following letters:
    - The mocap frame {m} is the fixed frame used by the motion capture software.
    - The camera optical frame {o} has its origin at the camera's focal point, z-axis pointing along the camera's 
      optical axis, x left, y down.
    - The camera frame {c} has its origin somewhere convenient on the camera's body (typically the center or a mounting 
      point), x forward, y left, z up.
    - The camera rigid body frame {r} is defined in the motion capture software, with its origin and orientation close 
      to but not exactly the same as {c}.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.optimize
import scipy.spatial
import scipy.stats
import cv2
from tf import transformations

import calibration_common


def create_insufficient_markers_mask(c_marker_counts, o_marker_counts, expected_c_marker_count, expected_o_marker_count):
    camera_mask = c_marker_counts == expected_c_marker_count
    object_mask = o_marker_counts == expected_o_marker_count
    return np.logical_and(camera_mask, object_mask)


def calculate_t_mos(camera_rb_poses, t_rc, t_co):
    optical_frame_poses = []
    for camera_rigid_body_pose in camera_rb_poses:
        t_mc = np.matmul(camera_rigid_body_pose, t_rc)
        t_mo = np.matmul(t_mc, t_co)
        optical_frame_poses.append(t_mo)
    return optical_frame_poses


def match_checkerboard_corners(image_coords, object_coords, t_mo_prior, camera_matrix, distortion_coeffs):
    """Utilizes the fact that the image coords and object coords are in the same order bar missing corners"""
    # This is the worst case for how wrong the indices are
    diff = len(object_coords) - len(image_coords)

    _, _, rotation, translation, _ = transformations.decompose_matrix(np.linalg.inv(t_mo_prior))
    rotation = transformations.euler_matrix(*rotation)
    rotation, _ = cv2.Rodrigues(rotation[:3, :3])
    projected_coords, _ = cv2.projectPoints(object_coords, rotation, translation, camera_matrix, distortion_coeffs)
    projected_coords = projected_coords.squeeze()

    # Need to make sure the checkerboard is the right way round
    error_1 = np.linalg.norm(image_coords[0] - projected_coords[0])
    error_2 = np.linalg.norm(image_coords[0] - projected_coords[-1])
    print(np.linalg.norm(image_coords[0] - image_coords[-1]))
    if error_2 < error_1:
        projected_coords = projected_coords[::-1]
        object_coords = object_coords[::-1]
        print("flipping checkerboard")

    # If they are the same length then they are in the right order
    if len(image_coords) == len(object_coords):
        return object_coords

    matched_object_coords = []
    matched_proj_coords = []
    for i, image_coord in enumerate(image_coords):
        best_match = projected_coords[i]
        best_object_coord = object_coords[i]
        # Find the object coord that matches the image coord within the possible options
        for proj_coord, object_coord in zip(projected_coords[i:i + diff + 1], object_coords[i:i + diff + 1]):
            if np.linalg.norm(image_coord - proj_coord) < np.linalg.norm(image_coord - best_match):
                best_match = proj_coord
                best_object_coord = object_coord
        matched_proj_coords.append(best_match)
        matched_object_coords.append(best_object_coord)
    return np.array(matched_object_coords).squeeze()


def match_markers(image_coords, object_coords, camera_pose_prior, camera_matrix, distortion_coeffs):
    """Return image_coords sorted into the same order as projected_coords."""
    _, _, rotation, translation, _ = transformations.decompose_matrix(np.linalg.inv(camera_pose_prior))
    rotation = transformations.euler_matrix(*rotation)
    rotation, _ = cv2.Rodrigues(rotation[:3, :3])
    projected_coords, _ = cv2.projectPoints(object_coords, rotation, translation, camera_matrix, distortion_coeffs)
    projected_coords = projected_coords.squeeze()
    tree = scipy.spatial.cKDTree(projected_coords)
    dists, indices = tree.query(image_coords, k=1)
    return object_coords[indices]


def estimate_pose(image_coords, object_coords, camera_matrix, distortion_coeffs, use_ransac):
    if use_ransac:
        _, rotation, translation, _ = cv2.solvePnPRansac(object_coords, image_coords, camera_matrix, distortion_coeffs,
                                                     flags=cv2.SOLVEPNP_P3P, iterationsCount=500)
    else:
        _, rotation, translation = cv2.solvePnP(object_coords, image_coords, camera_matrix, distortion_coeffs,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
    rotation, _ = cv2.Rodrigues(rotation)
    t_om = transformations.compose_matrix(
        translate=translation.squeeze(), 
        angles=transformations.euler_from_matrix(rotation))
    
    t_mo = np.linalg.inv(t_om)
    return t_mo


def remove_outliers(calibration_tfs, translation_tolerance=0.1, rotation_tolerance=10):
    estimate = np.array([0, 0, 0, 0, 0, 0])
    tolerances = [translation_tolerance] * 3 + [rotation_tolerance] * 3
    estimate_mask = np.all(np.abs(calibration_tfs - estimate) < tolerances, axis=1)

    medians = np.median(calibration_tfs, axis=0)
    median_mask = np.all(np.abs(calibration_tfs - medians) < tolerances, axis=1)

    mask = estimate_mask & median_mask
    return np.array(calibration_tfs)[mask], mask


def find_image_object_correspondences(all_image_coordinates, all_object_coordinates, t_mo_priors, camera_matrix, distortion_coeffs, is_checkerboard):
    matched_markers = []

    for image_coordinates, object_coordinates, t_mo_prior in zip(all_image_coordinates, all_object_coordinates, t_mo_priors):
        if is_checkerboard:
            matched_markers.append(
                match_checkerboard_corners(image_coordinates, object_coordinates, t_mo_prior, camera_matrix, distortion_coeffs))
        else:
            matched_markers.append(
                match_markers(image_coordinates, object_coordinates, t_mo_prior, camera_matrix, distortion_coeffs))
    return np.array(matched_markers)


def calculate_t_rcs(all_image_coordinates, all_object_coordinates, t_mrs, camera_matrix, distortion_coeffs, t_co, use_ransac):
    t_rcs = []
    for image_coordinates, object_coordinates, camera_rb_pose in zip(all_image_coordinates, all_object_coordinates,
                                                                     t_mrs):
        optical_frame_pose = estimate_pose(image_coordinates, object_coordinates, camera_matrix, distortion_coeffs,
                                           use_ransac=use_ransac)
        camera_link_pose = np.matmul(optical_frame_pose, np.linalg.inv(t_co))
        calibration_transform = np.matmul(np.linalg.inv(camera_rb_pose), camera_link_pose)
        t_rcs.append(calibration_transform)
    return t_rcs


def calibrate(all_image_coordinates, all_object_coordinates, t_mrs, t_co, camera_matrix, distortion_coeffs, is_checkerboard):

    # Roughly calculate the pose of the optical frame using t_rc=0
    t_mo_priors = calculate_t_mos(t_mrs, np.identity(4), t_co)

    # Find rough image-object correspondences (we will fix these later once we have a better idea of t_rc)
    matched_object_coordinates = find_image_object_correspondences(all_image_coordinates, all_object_coordinates, t_mo_priors, camera_matrix, distortion_coeffs, is_checkerboard)

    # Calculate the calibration transforms
    t_rcs = calculate_t_rcs(all_image_coordinates, matched_object_coordinates, t_mrs, camera_matrix, distortion_coeffs, t_co, use_ransac=True)

    # Convert the calibration transforms to an (x, y, z, r, p ,y) format for outlier removal and averaging
    tf_rcs = calibration_common.Ts_to_tfs(t_rcs)

    # Remove the outliers and determine the average calibration transform
    tf_rc_inliers, _ = remove_outliers(tf_rcs)
    tf_rc_initial = np.mean(tf_rc_inliers, axis=0)
    t_rc_initial = transformations.compose_matrix(translate=tf_rc_initial[:3], angles=np.radians(tf_rc_initial[3:]))

    # Use the calculated calibration transform to get better optical-frame priors
    t_mo_priors = calculate_t_mos(t_mrs, t_rc_initial, t_co)

    # Perform better matching now we have good optical frame priors
    matched_object_coordinates = find_image_object_correspondences(all_image_coordinates, all_object_coordinates, t_mo_priors,
                                                                   camera_matrix, distortion_coeffs, is_checkerboard=False)

    # Recalculate the calibration transforms again, this time the matches should be good, so use an iterative approach
    t_rc_posteriors = calculate_t_rcs(all_image_coordinates, matched_object_coordinates, t_mrs, camera_matrix, distortion_coeffs, t_co, use_ransac=False)

    # Convert the calibration transforms to an (x, y, z, r, p ,y) format for outlier removal and averaging
    tf_rc_posteriors = calibration_common.Ts_to_tfs(t_rc_posteriors)

    # Remove the outliers and determine the average calibration transform
    tf_rc_posteriors, mask = remove_outliers(tf_rc_posteriors)
    tf_rc_posterior = np.mean(tf_rc_posteriors, axis=0)
    t_rc_posterior = transformations.compose_matrix(translate=tf_rc_posterior[:3], angles=np.radians(tf_rc_posterior[3:]))
    print(tf_rc_posterior)

    # Filter outlier frames
    count = len(all_image_coordinates)
    all_image_coordinates = all_image_coordinates[mask]
    matched_object_coordinates = matched_object_coordinates[mask]
    t_mrs = t_mrs[mask]
    print("Num inliers {}/{}".format(len(all_image_coordinates), count))

    # Project object points into camera rigid body frame beforehand, as one big array (not separated by bags)
    object_coords_r = []

    for image_coords, object_coords_M_sorted, T_MRs in zip(all_image_coordinates, matched_object_coordinates, t_mrs):
        for i in range(len(image_coords)):
            object_coords_r.append(calibration_common.apply_T(np.linalg.inv(T_MRs), np.array([object_coords_M_sorted[i]])))
    object_coords_r = np.vstack(object_coords_r)

    image_coords = np.vstack(np.vstack(all_image_coordinates))
    prior = calibration_common.T_to_tf(t_rc_posterior)

    def f(tf_rc):
        t_ro = np.dot(calibration_common.tf_to_T(tf_rc), t_co)
        return calibration_common.reprojection_error_vector(t_ro, object_coords_r, image_coords, camera_matrix,
                                                            distortion_coeffs)
    res = scipy.optimize.least_squares(f, prior, verbose=0, args=())
    return res.x


def main():
    # is_checkerboard = True
    is_checkerboard = False
    data = np.load("../data/sideways_board_data.npz")
    # data = np.load("../data/marker_data.npz")

    bag_files = data['bag_files']

    calibration_object_marker_counts = data['calibration_object_marker_counts']
    camera_marker_counts = data['camera_marker_counts']

    camera_matrix = data['camera_matrix']
    distortion_coeffs = data['distortion_coeffs']

    t_co = data['cam_to_optical_frame']
    camera_rb_poses = data['camera_rb_poses']

    all_image_coordinates = data['image_coordinates']
    all_object_coordinates = data['object_coordinates']


    # Remove data that is missing camera or calibration object markers
    is_checkerboard = True
    if is_checkerboard:
        filter_mask = create_insufficient_markers_mask(
            camera_marker_counts, calibration_object_marker_counts, 5, 5)
        mask = [0 if len(row) != 187 else 1 for row in all_image_coordinates]
        print(mask)
    else:
        filter_mask = create_insufficient_markers_mask(
            camera_marker_counts, calibration_object_marker_counts, 6, 14)
    is_checkerboard = False

    camera_rb_poses = camera_rb_poses[filter_mask]
    all_image_coordinates = all_image_coordinates[filter_mask]
    all_object_coordinates = all_object_coordinates[filter_mask]


    result = calibrate(all_image_coordinates, all_object_coordinates, camera_rb_poses, t_co, camera_matrix, distortion_coeffs, is_checkerboard=is_checkerboard)
    print(result)


if __name__ == "__main__":
    main()



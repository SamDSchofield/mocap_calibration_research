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
import random

import calibration_common




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
    if error_2 < error_1:
        projected_coords = projected_coords[::-1]
        object_coords = object_coords[::-1]
        # print("flipping checkerboard")

    # # print(np.mean(np.linalg.norm(projected_coords - image_coords, axis=1)))
    # # Corners are x-y format
    # box = cv2.boxPoints(cv2.minAreaRect(image_coords))
    #
    # combs = []
    # for corner in box:
    #    combs.append((box[0], corner))
    # combs = list(reversed(sorted(combs, key=lambda x: np.linalg.norm(x[0] - x[1]))))
    # length_points = combs[1]
    # horizontal = abs((length_points[0] - length_points[1])[0]) / 17
    #
    #
    #
    #
    # # print("h", h)



    # draw = True
    # if draw:
    #     # Find length
    #     image = np.zeros((1080, 1920), dtype="uint8")
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #     box = np.int0(box)
    #     image = cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    #     print(length_points)
    #     p1 = tuple(length_points[0])
    #     p2 = tuple(length_points[1])
    #     print(p1)
    #     image = cv2.line(image, p1, p2, (0, 255, 0))
    #     print(len(image_coords))
    #     for image_coord in image_coords:
    #         image = cv2.circle(image, (int(image_coord[0]), int(image_coord[1])), 1, (255, 0, 0))
    #
    #     cv2.imshow("", image)
    #     cv2.waitKey(0)

    return object_coords


def match_markers(image_coords, object_coords, camera_pose_prior, camera_matrix, distortion_coeffs):
    """Return image_coords sorted into the same order as projected_coords."""
    # assert len(image_coords) == len(object_coords), "Image, object coords not same length. {} != {}".format(
    #     len(image_coords), len(object_coords))

    _, _, rotation, translation, _ = transformations.decompose_matrix(np.linalg.inv(camera_pose_prior))
    rotation = transformations.euler_matrix(*rotation)
    rotation, _ = cv2.Rodrigues(rotation[:3, :3])
    projected_coords, _ = cv2.projectPoints(object_coords, rotation, translation, camera_matrix, distortion_coeffs)
    projected_coords = projected_coords.squeeze()
    tree = scipy.spatial.cKDTree(projected_coords)
    dists, indices = tree.query(image_coords, k=1)
    # print(np.mean(dists))
    # print(indices)
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


def remove_outliers(calibration_tfs, translation_tolerance=0.1, rotation_tolerance=0.174):
    """
    Angles are in radians
    """
    estimate = np.array([0, 0, 0, 0, 0, 0])
    tolerances = [translation_tolerance] * 3 + [rotation_tolerance] * 3
    estimate_mask = np.all(np.abs(calibration_tfs - estimate) < tolerances, axis=1)

    inliers = np.array(calibration_tfs)[estimate_mask]

    medians = np.median(inliers, axis=0)
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


def calibrate(all_image_coordinates, all_object_coordinates, t_mrs, t_co, camera_matrix, distortion_coeffs, verbose=True, is_checkerboard=False):
    if verbose:
        print("Calibrating")
    # Roughly calculate the pose of the optical frame using t_rc=0

    t_mo_priors = calculate_t_mos(t_mrs, np.identity(4), t_co)

    # Find rough image-object correspondences (we will fix these later once we have a better idea of t_rc)
    matched_object_coordinates = find_image_object_correspondences(all_image_coordinates, all_object_coordinates, t_mo_priors, camera_matrix, distortion_coeffs, False)

    # Calculate the calibration transforms
    t_rcs = calculate_t_rcs(all_image_coordinates, matched_object_coordinates, t_mrs, camera_matrix, distortion_coeffs, t_co, use_ransac=True)

    # Convert the calibration transforms to an (x, y, z, r, p ,y) format for outlier removal and averaging
    tf_rcs = calibration_common.Ts_to_tfs(t_rcs)

    # Remove the outliers and determine the average calibration transform
    tf_rc_inliers, _ = remove_outliers(tf_rcs)
    tf_rc_initial = np.mean(tf_rc_inliers, axis=0)
    t_rc_initial = transformations.compose_matrix(translate=tf_rc_initial[:3], angles=tf_rc_initial[3:])

    if verbose:
        print("Estimate after first pass:")
        print(tf_rc_initial)
        print("Refining estimate...")

    # Use the calculated calibration transform to get better optical-frame priors
    t_mo_priors = calculate_t_mos(t_mrs, t_rc_initial, t_co)


    # Perform better matching now we have good optical frame priors
    matched_object_coordinates = find_image_object_correspondences(all_image_coordinates, all_object_coordinates, t_mo_priors,
                                                                   camera_matrix, distortion_coeffs, is_checkerboard=is_checkerboard)

    # Recalculate the calibration transforms again, this time the matches should be good, so use an iterative approach
    t_rc_posteriors = calculate_t_rcs(all_image_coordinates, matched_object_coordinates, t_mrs, camera_matrix, distortion_coeffs, t_co, use_ransac=False)

    # Convert the calibration transforms to an (x, y, z, r, p ,y) format for outlier removal and averaging
    tf_rc_posteriors = calibration_common.Ts_to_tfs(t_rc_posteriors)

    # Remove the outliers and determine the average calibration transform
    tf_rc_posteriors, mask = remove_outliers(tf_rc_posteriors)
    tf_rc_posterior = np.mean(tf_rc_posteriors, axis=0)
    t_rc_posterior = transformations.compose_matrix(translate=tf_rc_posterior[:3], angles=tf_rc_posterior[3:])

    if verbose:
        print("Estimate after refining:")
        print(tf_rc_posterior)
        print("Optimising estimate...")

    if is_checkerboard:
        return tf_rc_posterior

    # Filter outlier frames
    count = len(all_image_coordinates)
    all_image_coordinates = all_image_coordinates[mask]
    matched_object_coordinates = matched_object_coordinates[mask]
    t_mrs = t_mrs[mask]

    if verbose:
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


def group_list(list_, group_size):
    return zip(*(iter(list_),) * group_size)


def k_fold(data_file, results_file, is_checkerboard, k=5):

    data = np.load(data_file)

    calibration_object_marker_counts = data['calibration_object_marker_counts']
    camera_marker_counts = data['camera_marker_counts']

    camera_matrix = data['camera_matrix']
    distortion_coeffs = data['distortion_coeffs']

    t_co = data['cam_to_optical_frame']
    camera_rb_poses = data['camera_rb_poses']

    all_image_coordinates = data['image_coordinates']
    all_object_coordinates = data['object_coordinates']

    bag_files = data['bag_files']


    # Remove data that is missing camera or calibration object markers
    mask = []
    for img_coords, obj_coords in zip(all_image_coordinates, all_object_coordinates):
        mask.append(len(img_coords) == len(obj_coords))
    mask = np.array(mask)
    bag_files = bag_files[mask]
    camera_rb_poses = camera_rb_poses[mask]
    all_image_coordinates = all_image_coordinates[mask]
    all_object_coordinates = all_object_coordinates[mask]

    if is_checkerboard:
        filter_mask = create_insufficient_markers_mask(
            camera_marker_counts, calibration_object_marker_counts, 6, 5)
    else:
        filter_mask = create_insufficient_markers_mask(
            camera_marker_counts, calibration_object_marker_counts, 6, 14)

    bag_files = bag_files[filter_mask]
    print(len(camera_rb_poses))
    camera_rb_poses = camera_rb_poses[filter_mask]
    print(len(camera_rb_poses))
    all_image_coordinates = all_image_coordinates[filter_mask]
    all_object_coordinates = all_object_coordinates[filter_mask]

    random.seed = 3
    distinct_bags = list(set(bag_files))
    files_per_fold = len(distinct_bags) // k
    random.shuffle(distinct_bags)

    folds = group_list(distinct_bags, files_per_fold)
    all_train_bags = []
    all_test_bags = []
    t_rcs = []
    for fold in folds:
        fold_mask = np.logical_not(np.in1d(bag_files, fold))

        fold_camera_rb_poses = camera_rb_poses[fold_mask]
        fold_image_coordinates = all_image_coordinates[fold_mask]
        fold_object_coordinates = all_object_coordinates[fold_mask]
        result = calibrate(fold_image_coordinates, fold_object_coordinates, fold_camera_rb_poses, t_co, camera_matrix, distortion_coeffs, verbose=True, is_checkerboard=is_checkerboard)

        print(result)
        t_rcs.append(result)
        train_bags = list(set(bag_files[fold_mask]))
        test_bags = list(set(bag_files[np.logical_not(fold_mask)]))
        all_train_bags.append(train_bags)
        all_test_bags.append(test_bags)

    np.savez(
        results_file,
        t_rcs=t_rcs,
        all_train_bags=all_train_bags,
        all_test_bags=all_test_bags,
        readme="""
        t_rcs=calibration transform from rigid body to camera_link.
        all_train_bags=the bags each transform was calibrated on.
        all_test_bags=the bags each transform should be tested on.
        """
    )


def main():
    # is_checkerboard = True
    is_checkerboard = False
    data = np.load("../data/5_markers.npz")
    # data = np.load("../data/5_marker_data.npz")

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
    # if is_checkerboard:
    #     filter_mask = create_insufficient_markers_mask(
    #         camera_marker_counts, calibration_object_marker_counts, 5, 5)
    # else:
    #     filter_mask = create_insufficient_markers_mask(
    #         camera_marker_counts, calibration_object_marker_counts, 6, 14)

    # print(len(camera_rb_poses))
    # camera_rb_poses = camera_rb_poses[filter_mask]
    # print(len(camera_rb_poses))
    # all_image_coordinates = all_image_coordinates[filter_mask]
    # all_object_coordinates = all_object_coordinates[filter_mask]

    result = calibrate(all_image_coordinates, all_object_coordinates, camera_rb_poses, t_co, camera_matrix, distortion_coeffs, verbose=True)
    print(result)


if __name__ == "__main__":
    # main()
    k_fold("../data/all_markers.npz", "../data/marker_calibration.npz", is_checkerboard=False, k=5)
    k_fold("../data/all_boards.npz", "../data/board_calibration.npz", is_checkerboard=True, k=5)



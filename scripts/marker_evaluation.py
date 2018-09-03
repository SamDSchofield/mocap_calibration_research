#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import cv2
import scipy
import scipy.optimize
import scipy.spatial
import scipy.stats
from tf import transformations


from calibration_common import create_insufficient_markers_mask


def calculate_t_mos(camera_rb_poses, t_rc, t_co):
    optical_frame_poses = []
    for camera_rigid_body_pose in camera_rb_poses:
        t_mc = np.matmul(camera_rigid_body_pose, t_rc)
        t_mo = np.matmul(t_mc, t_co)
        optical_frame_poses.append(t_mo)
    return optical_frame_poses


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
    output_image = np.zeros((1080, 1920), dtype="uint8")
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    # for proj, img in zip(image_coords, projected_coords[indices]):
    #     output_image = cv2.circle(output_image, (int(proj[0]), int(proj[1])), 1, (255, 0, 0))
    #     output_image = cv2.circle(output_image, (int(img[0]), int(img[1])), 1, (0, 255, 0))
    # cv2.imshow("a", output_image)
    # cv2.waitKey(0)
    return object_coords[indices]


def project_markers(t_mo, markers, camera_mat, dist_coeffs):
    rvec, _ = cv2.Rodrigues(t_mo[:3, :3])
    tvec = t_mo[:3, 3]
    markers2d, _ = cv2.projectPoints(markers, rvec, tvec, camera_mat, dist_coeffs)
    return markers2d.reshape((len(markers2d), 2))


def evaluate(t_rc, camera_rb_poses, all_image_coords, all_object_coords, t_co, camera_mat, distortion_coeffs):
    # Get the optical frame pose in the mocap frame
    t_rc = transformations.compose_matrix(translate=t_rc[:3], angles=(t_rc[3:]))
    t_mos = calculate_t_mos(camera_rb_poses, t_rc, t_co)

    total_reprojection_error = 0
    for t_mo, image_coords, object_coords in zip(t_mos, all_image_coords, all_object_coords):
        object_coords = match_markers(image_coords, object_coords, t_mo, camera_mat, distortion_coeffs)

        projected_image_coords = project_markers(np.linalg.inv(t_mo), object_coords, camera_mat, distortion_coeffs)
        total_reprojection_error += np.mean(np.linalg.norm(projected_image_coords - image_coords, axis=1))
    return total_reprojection_error / len(t_mos)


def evaluate_k_fold(marker_calibration_file, board_calibration_file, raw_data_file):
    raw_data = np.load(raw_data_file)
    bag_files = raw_data["bag_files"]

    camera_mat = raw_data["camera_matrix"]
    distortion_coeffs = raw_data["distortion_coeffs"]
    camera_marker_counts = raw_data["camera_marker_counts"]
    calibration_object_marker_counts = raw_data['calibration_object_marker_counts']

    t_co = raw_data["cam_to_optical_frame"]

    all_image_coords = raw_data["image_coordinates"]
    all_object_coords = raw_data["object_coordinates"]
    cam_rb_poses = raw_data["camera_rb_poses"]

    filter_mask = create_insufficient_markers_mask(camera_marker_counts, calibration_object_marker_counts, 6, 14)
    bag_files = bag_files[filter_mask]
    all_image_coords = all_image_coords[filter_mask]
    all_object_coords = all_object_coords[filter_mask]
    cam_rb_poses = cam_rb_poses[filter_mask]

    marker_calibration_data = np.load(marker_calibration_file)
    marker_calibrations = marker_calibration_data["t_rcs"]
    all_test_bags = marker_calibration_data["all_test_bags"]

    board_calibration_data = np.load(board_calibration_file)
    board_calibrations = board_calibration_data["t_rcs"]
    # all_test_bags = board_calibration_data["all_test_bags"]

    marker_errors = []
    board_errors = []
    for test_bags, marker_calibration, board_calibration in zip(all_test_bags, marker_calibrations, board_calibrations):
        test_mask = np.in1d(bag_files, test_bags)

        test_image_coords = all_image_coords[test_mask]
        test_object_coords = all_object_coords[test_mask]
        test_cam_rb_poses = cam_rb_poses[test_mask]

        error = evaluate(marker_calibration, test_cam_rb_poses, test_image_coords, test_object_coords, t_co, camera_mat, distortion_coeffs)
        marker_errors.append(error)

        error = evaluate(board_calibration, test_cam_rb_poses, test_image_coords, test_object_coords, t_co, camera_mat, distortion_coeffs)
        board_errors.append(error)

    print("Marker mean {}, std {}".format(np.mean(marker_errors), np.std(marker_errors)))
    print("Board mean {}, std {}".format(np.mean(board_errors), np.std(board_errors)))


if __name__ == "__main__":
    evaluate_k_fold("../data/marker_calibration.npz", "../data/board_calibration.npz", "../data/all_markers.npz")
    # evaluate_k_fold("../data/marker_calibration.npz", "../data/board_calibration.npz", "../data/all_boards.npz")

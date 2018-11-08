#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import click
import cv2
import scipy
import scipy.optimize
import scipy.spatial
import scipy.stats
from tf import transformations

import calibration_common


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
    return object_coords[indices]


def project_markers(t_mo, markers, camera_mat, dist_coeffs):
    rvec, _ = cv2.Rodrigues(t_mo[:3, :3])
    tvec = t_mo[:3, 3]
    markers2d, _ = cv2.projectPoints(markers, rvec, tvec, camera_mat, dist_coeffs)
    return markers2d.reshape((len(markers2d), 2))


def distance(marker_1, marker_2):
    return np.linalg.norm(marker_1 - marker_2)


def find_closest_marker_distance(marker_position, markers):
    best_distance = distance(marker_position, markers[0])
    best_marker = markers[0]
    for marker in markers:
        d = distance(marker_position, marker)
        if d < best_distance:
            best_distance = d
            best_marker = marker
    return best_marker, best_distance


def evaluate(t_rc, camera_rb_poses, all_image_coords, all_calibration_markers, t_co, camera_mat, distortion_coeffs):
    # Get the optical frame pose in the mocap frame
    t_rc = transformations.compose_matrix(translate=t_rc[:3], angles=(t_rc[3:]))
    t_mos = calculate_t_mos(camera_rb_poses, t_rc, t_co)

    shape = (8, 5)

    square_size = 0.035
    marker_z = 0.012
    marker_positions = [
        (-square_size, 0, -marker_z),
        (square_size * 8, 0, -marker_z),
        (square_size * 8, square_size * 5, -marker_z),
        (-square_size, square_size * 5, -marker_z),
    ]

    total_error = 0
    for t_mo, image_coords, calibration_markers in zip(t_mos, all_image_coords, all_calibration_markers):
        checkerboard_coords = np.zeros((shape[0] * shape[1], 3), np.float32)
        checkerboard_coords[:, :2] = (np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)) * square_size

        retval, rvec, tvec = cv2.solvePnP(checkerboard_coords, image_coords, camera_mat, distortion_coeffs)
        t_o_ch = calibration_common.se3_to_T(rvec, tvec)
        t_m_ch = np.matmul(t_mo, t_o_ch)
        total_distance = 0
        for marker in marker_positions:
            marker = np.matmul(t_m_ch, np.append(marker, 1))[:3]
            match, distance = find_closest_marker_distance(marker, calibration_markers)
            print(match[2] - marker[2])
            total_distance += distance
        total_distance /= 4
        print("*" * 80)
        # To handle when the checkerboard was flipped during calibration
        checkerboard_coords = checkerboard_coords[::-1]
        retval, rvec, tvec = cv2.solvePnP(checkerboard_coords, image_coords, camera_mat, distortion_coeffs)
        t_o_ch = calibration_common.se3_to_T(rvec, tvec)
        t_m_ch = np.matmul(t_mo, t_o_ch)
        other_total_distance = 0
        for marker in marker_positions:
            marker = np.matmul(t_m_ch, np.append(marker, 1))[:3]
            match, distance = find_closest_marker_distance(marker, calibration_markers)
            other_total_distance += distance
            print(match[2] - marker[2])
        other_total_distance /= 4

        if other_total_distance < total_distance:
            total_distance = other_total_distance
        total_error += total_distance
    return total_error / len(t_mos)


def evaluate_k_fold(marker_calibration_file, board_calibration_file, raw_data_file):
    raw_data = np.load(raw_data_file)
    bag_files = raw_data["bag_files"]

    camera_mat = raw_data["camera_matrix"]
    distortion_coeffs = raw_data["distortion_coeffs"]
    camera_marker_counts = raw_data["camera_marker_counts"]
    calibration_object_marker_counts = raw_data['calibration_object_marker_counts']

    t_co = raw_data["cam_to_optical_frame"]

    all_image_coords = raw_data["image_coordinates"]
    cam_rb_poses = raw_data["camera_rb_poses"]
    all_calibration_markers = raw_data["all_calibration_markers"]

    filter_mask = calibration_common.create_insufficient_markers_mask(camera_marker_counts,
                                                                      calibration_object_marker_counts, 6, 5)
    bag_files = bag_files[filter_mask]
    all_image_coords = all_image_coords[filter_mask]
    all_calibration_markers = all_calibration_markers[filter_mask]
    cam_rb_poses = cam_rb_poses[filter_mask]

    marker_calibration_data = np.load(marker_calibration_file)
    marker_calibrations = marker_calibration_data["t_rcs"]

    board_calibration_data = np.load(board_calibration_file)
    board_calibrations = board_calibration_data["t_rcs"]
    all_test_bags = board_calibration_data["all_test_bags"]

    marker_errors = []
    board_errors = []

    for test_bags, marker_calibration, board_calibration in zip(all_test_bags, marker_calibrations, board_calibrations):
        test_mask = np.in1d(bag_files, test_bags)

        test_image_coords = all_image_coords[test_mask]
        test_cam_rb_poses = cam_rb_poses[test_mask]
        test_calibration_markers = all_calibration_markers[test_mask]
        print("Marker")
        error = evaluate(marker_calibration, test_cam_rb_poses, test_image_coords, test_calibration_markers, t_co,
                         camera_mat, distortion_coeffs)
        marker_errors.append(error)

        print("Board")
        error = evaluate(board_calibration, test_cam_rb_poses, test_image_coords, test_calibration_markers, t_co,
                         camera_mat, distortion_coeffs)
        board_errors.append(error)

    print("Marker mean {}, std {}".format(np.mean(marker_errors), np.std(marker_errors)))
    print("Board mean {}, std {}".format(np.mean(board_errors), np.std(board_errors)))


def main():
    evaluate_k_fold("../data/marker_calibration_10_9_18.npz", "../data/board_calibration_10_9_18.npz",
                    "../data/all_boards_10_9_18.npz")


@click.command()
@click.argument("marker_calibration_file", type=click.Path(exists=True), nargs=1)
@click.argument("board_calibration_file", type=click.Path(exists=True), nargs=1)
@click.argument("raw_data_file", type=click.Path(exists=True), nargs=1)
def cli(marker_calibration_file, board_calibration_file, raw_data_file):
    """
    \b
    Evaluates the transformations in the given calibration files using a checkerboard.

    The marker and board calibration files are the output of the calibrate script.
    The raw_data_file is the output of the extract_checkerboard_data script.
    """

    evaluate_k_fold(marker_calibration_file, board_calibration_file, raw_data_file)


if __name__ == "__main__":
    cli()

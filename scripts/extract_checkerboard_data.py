#!/usr/bin/env python

"""
Extract data from ros bags and store it as numpy arrays. The following data should be stored

Per frame:
* bag_file
* image_coords
* object_coords
* camera_rb_pose,
* calibration_object_markers
* calibration_object_markers_counts
* camera_marker_counts,

Once:
* cam_to_optical_frame_tf
* cam_matrix
* dist coefficients


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

from __future__ import print_function, division

import numpy as np

import click
import cv2
import cv_bridge
import rosbag
import rospy
import tf2_ros
from tf import transformations

import calibration_common

DISPLAY = False  # TODO: don't use globals


def transform_markers_to_mocap_frame(checkerboard_rigid_body_poses, all_checkerboard_corners):
    all_object_coordinates = []
    board_rigid_body_to_optical_tf = transformations.compose_matrix(angles=np.radians((180, 0, 0)),
                                                                    translate=(0.035, 0, -0.012))
    for board_pose, object_coordinates in zip(checkerboard_rigid_body_poses, all_checkerboard_corners):
        mocap_to_board_optical_tf = np.matmul(board_pose, board_rigid_body_to_optical_tf)
        frame_object_coordinates = []

        for object_coord in object_coordinates:
            # Need to add a 1 to the object coord so we can do matmul with pose matrix, then remove it after
            frame_object_coordinates.append(np.matmul(mocap_to_board_optical_tf, np.append(object_coord, 1))[:3])
        all_object_coordinates.append(np.array(frame_object_coordinates))
    return all_object_coordinates


def detect_checkerboard(image, shape, size, threshold=150, max_value=255):
    _, image = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    found, corners = cv2.findChessboardCorners(image, shape, None, flags)

    if found:
        print("found")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
        corners_refined = np.squeeze(cv2.cornerSubPix(image, corners, (7, 7), (-1, -1), criteria))
        if corners_refined[0][0] > corners_refined[-1][0]:  # Because sometimes the chessboard is detected upside down
            corners_refined = corners_refined[::-1]

        object_points = np.zeros((shape[0] * shape[1], 3), np.float32)
        object_points[:, :2] = (np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)) * size
        return corners_refined, object_points

    if DISPLAY:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if corners is not None:
            for corner in corners:
                x, y = corner[0]
                image = cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0))
        cv2.imshow("image", image)
        cv2.waitKey(1)
    return None, None


def extract_data_from_bags(bag_file_paths, output_file, shape, size, threshold):
    camera_matrix = None
    distortion_coeffs = None
    camera_link_to_optical_frame_tf = None

    camera_marker_counts = []
    calibration_object_marker_counts = []
    all_image_coordinates = []
    all_checkerboard_corners = []
    all_calibration_markers = []
    camera_rigid_body_poses = []
    bag_files = []

    checkerboard_rigid_body_poses = []

    bridge = cv_bridge.CvBridge()

    for bag_file_path in bag_file_paths:
        print("*" * 80)
        print("Starting bag: {}".format(bag_file_path))
        camera_matrix = None
        distortion_coeffs = None
        camera_marker_count = 0
        checkerboard_marker_count = None
        checkerboard_markers = []

        bag = rosbag.Bag(bag_file_path)
        tf_buffer = calibration_common.load_tf_history_from_bag(bag)

        camera_link_to_optical_frame_tf = calibration_common.tf_stamped_to_mat(
            tf_buffer.lookup_transform("camera_link", "camera_color_optical_frame", rospy.Time(0)))

        for i, (topic, message, t) in enumerate(bag.read_messages()):
            if topic == "/camera/color/image_raw":
                # Check that everything is initialised
                if None not in [camera_matrix, distortion_coeffs, camera_link_to_optical_frame_tf]:

                    image = bridge.imgmsg_to_cv2(message, "mono8")
                    image_coordinates, object_coordinates = detect_checkerboard(image, shape=shape, size=size,
                                                                                threshold=threshold)

                    camera_rigid_body_pose = None
                    try:
                        camera_rigid_body_pose = calibration_common.tf_stamped_to_mat(
                            tf_buffer.lookup_transform("mocap", "Camera", t))
                    except tf2_ros.ExtrapolationException:
                        print("Tf lookup failed from mocap->Camera")

                    checkerboard_rigid_body_pose = None
                    try:
                        checkerboard_rigid_body_pose = calibration_common.tf_stamped_to_mat(
                            tf_buffer.lookup_transform("mocap", "Checkerboard", t))
                    except tf2_ros.ExtrapolationException:
                        print("Tf lookup failed from mocap->Checkerboard")

                    if None not in [camera_rigid_body_pose, checkerboard_rigid_body_pose,
                                    image_coordinates, camera_marker_count, checkerboard_marker_count]:
                        print("Storing data")
                        all_image_coordinates.append(image_coordinates)
                        all_checkerboard_corners.append(object_coordinates)
                        camera_rigid_body_poses.append(camera_rigid_body_pose)
                        bag_files.append(bag_file_path)
                        camera_marker_counts.append(camera_marker_count)
                        calibration_object_marker_counts.append(checkerboard_marker_count)
                        checkerboard_rigid_body_poses.append(checkerboard_rigid_body_pose)
                        all_calibration_markers.append(checkerboard_markers)

                    print([camera_marker_count, checkerboard_marker_count])



            elif topic == "/camera/color/camera_info":
                camera_matrix = np.array(message.K).reshape((3, 3))
                distortion_coeffs = np.array(message.D)

            elif topic == "/mocap/rigid_bodies/Checkerboard/markers":
                checkerboard_markers = [(p.x, p.y, p.z) for p in message.positions]
                checkerboard_marker_count = len(message.positions)

            elif topic == "/mocap/rigid_bodies/Camera/markers":
                camera_marker_count = len(message.positions)

            if i % 1000 == 0:
                print("Completed {}/{} messages".format(i, bag.get_message_count()))

    all_object_coordinates = transform_markers_to_mocap_frame(checkerboard_rigid_body_poses, all_checkerboard_corners)

    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        cam_to_optical_frame=camera_link_to_optical_frame_tf,

        image_coordinates=all_image_coordinates,
        object_coordinates=all_object_coordinates,
        camera_rb_poses=camera_rigid_body_poses,

        camera_marker_counts=camera_marker_counts,
        calibration_object_marker_counts=calibration_object_marker_counts,
        all_calibration_markers=all_calibration_markers,
        bag_files=bag_files,
    )
    print("Saved data to {}".format(output_file))


def main():
    bag_file_paths = calibration_common.list_bag_files(base_path="/home/sam/Desktop", directories=[
        "calibration_data_10-9-18/board",
        # "more_calibration_data/board/side",
    ])
    extract_data_from_bags(bag_file_paths, "../data/all_boards_10_9_18.npz")


@click.command()
@click.argument("outfile", type=click.File('w'))
@click.argument("infiles", type=click.Path(exists=True), nargs=-1)
@click.option("--directories/--files", default=False, help="Specify a list of directories instead of bag files.")
@click.option("--display/--dont_display", default=False, help="Specify whether to display the detection image or not.")
@click.option("--shape", default="8x5", help="The shape of the checkerboard e.g. 5x8.")
@click.option("--size", default=0.035, help="The size of the checkerboard squares in meters.")
@click.option("--threshold", default=150, help="The pixel intensity threshold.")
def cli(outfile, infiles, directories, display, shape, size, threshold):
    """
    \b
    Extracts the following data from the .bag files provided and stores it in a .npz file:
        - camera matrix
        - distortion coefficients
        - image coordinates
        - object coordinates
        - camera rigid body poses
        - camera to optical frame transform
        - camera rigid body marker counts
        - calibration object marker counts
        - bag files used
    """
    if directories:
        infiles = calibration_common.list_bag_files(directories=infiles)

    global DISPLAY
    DISPLAY = display
    shape = tuple(int(x) for x in shape.split("x"))
    extract_data_from_bags(infiles, outfile, shape, size, threshold)


if __name__ == "__main__":
    cli()

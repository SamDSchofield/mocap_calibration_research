#!/usr/bin/env python

from __future__ import print_function, division

import math
import numpy as np

import click
import cv2
import cv_bridge
import rosbag
import rospy
import tf2_ros

import calibration_common


DISPLAY = False  # TODO: don't use globals


def detect_circles_contours(image, min_r, max_r):
    """Detects circles using OpenCV's findContours and returns those within min and max radius"""
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_area = []
    min_area = math.pi * min_r ** 2
    max_area = math.pi * max_r ** 2
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        # print(area)
        if min_area < area < max_area:
            contours_area.append(con)

    contours_circles = []
    # check if contour is of circular shape
    for con in contours_area:
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if perimeter == 0:
            break
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        if 0.7 < circularity < 1.2:
            contours_circles.append(con)

    circles = np.ndarray((len(contours_circles), 3))
    if DISPLAY:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(contours_circles):
            position, radius = cv2.minEnclosingCircle(contour)
            circles[i] = (position[0], position[1], radius)
            image = cv2.circle(image, (int(position[0]), int(position[1])), radius=int(radius), color=(0, 0, 255))
        cv2.imshow("image", image)
        cv2.waitKey(1)
    return circles


def detect_mocap_markers(image, threshold_value=150, morph_size=3, min_r=3, max_r=30):
    # Threshold the image
    if DISPLAY:
        cv2.imshow("original", image)
        cv2.waitKey(1)
    _, image = cv2.threshold(image[:, :], threshold_value, 255, cv2.THRESH_BINARY)

    # Perform open morphology
    kernel = np.ones((morph_size, morph_size), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    markers = detect_circles_contours(image, min_r, max_r)
    return markers[:, :2]  # Don't want radii of markers, just their centers


def extract_data_from_bags(bag_file_paths, output_file, min_marker_r=3, max_marker_r=30, threshold=150):
    camera_matrix = None
    distortion_coeffs = None
    camera_link_to_optical_frame_tf = None

    camera_marker_counts = []
    calibration_object_marker_counts = []
    all_image_coordinates = []
    all_object_coordinates = []
    camera_rigid_body_poses = []
    bag_files = []

    bridge = cv_bridge.CvBridge()
    short_bags = []
    for bag_file_path in bag_file_paths:
        print("*" * 80)
        print("Starting bag: {}".format(bag_file_path))

        camera_matrix = None
        distortion_coeffs = None
        object_coordinates = None
        camera_marker_count = 0

        bag = rosbag.Bag(bag_file_path)
        if bag.get_message_count() < 1000:
            short_bags.append(bag_file_path)
        tf_buffer = calibration_common.load_tf_history_from_bag(bag)

        camera_link_to_optical_frame_tf = calibration_common.tf_stamped_to_mat(
            tf_buffer.lookup_transform("camera_link", "camera_color_optical_frame", rospy.Time(0)))

        for i, (topic, message, t) in enumerate(bag.read_messages()):
            if topic == "/camera/color/image_raw":
                # Check that everything is initialised
                if None not in [camera_matrix, distortion_coeffs, object_coordinates, camera_link_to_optical_frame_tf,
                                camera_marker_count]:

                    image = bridge.imgmsg_to_cv2(message, "mono8")
                    image_coordinates = detect_mocap_markers(image, threshold_value=threshold, min_r=min_marker_r, max_r=max_marker_r)

                    camera_rigid_body_pose = None
                    try:
                        camera_rigid_body_pose = calibration_common.tf_stamped_to_mat(
                            tf_buffer.lookup_transform("mocap", "Camera", t))
                    except tf2_ros.ExtrapolationException:
                        print("Tf lookup failed from mocap->Camera")
                    except tf2_ros.LookupException:
                        print("Tf lookup failed, target doesn't exist")

                    if camera_rigid_body_pose is not None:
                        all_image_coordinates.append(image_coordinates)
                        all_object_coordinates.append(object_coordinates)
                        camera_rigid_body_poses.append(camera_rigid_body_pose)
                        bag_files.append(bag_file_path)
                        camera_marker_counts.append(camera_marker_count)
                        calibration_object_marker_counts.append(len(object_coordinates))

            elif topic == "/camera/color/camera_info":
                camera_matrix = np.array(message.K).reshape((3, 3))
                distortion_coeffs = np.array(message.D)

            elif topic == "/mocap/rigid_bodies/calibration_markers/markers":
                object_coordinates = np.array([(marker.x, marker.y, marker.z) for marker in message.positions])

            elif topic == "/mocap/rigid_bodies/Camera/markers":
                camera_marker_count = len(message.positions)

            if i % 1000 == 0:
                print("Completed {}/{} messages".format(i, bag.get_message_count()))

    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        image_coordinates=all_image_coordinates,
        object_coordinates=all_object_coordinates,
        camera_rb_poses=camera_rigid_body_poses,
        cam_to_optical_frame=camera_link_to_optical_frame_tf,
        camera_marker_counts=camera_marker_counts,
        calibration_object_marker_counts=calibration_object_marker_counts,
        bag_files=bag_files,
    )
    print("Saved data to {}".format(output_file))


@click.command()
@click.argument("outfile", type=click.File('w'))
@click.argument("infiles", type=click.Path(exists=True), nargs=-1)
@click.option("--directories/--files", default=False, help="Specify a list of directories instead of bag files.")
@click.option("--display/--dont_display", default=False, help="Specify whether to display the detection image or not.")
@click.option("--min_marker_radius", default=3, help="The minimum marker radius in pixels.")
@click.option("--max_marker_radius", default=30, help="The maximum marker radius in pixels.")
@click.option("--threshold", default=150, help="The pixel intensity threshold.")
def cli(outfile, infiles, directories, display, min_marker_radius, max_marker_radius, threshold):
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
    extract_data_from_bags(infiles, outfile, min_marker_radius, max_marker_radius, threshold)


if __name__ == "__main__":
    cli()

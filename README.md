# Camera-mocap extrinsic calibration research
This code accompanies the paper "Calibration for Camera-Motion Capture Extrinsics". 
The purpose is to calculate the transformation between a camera and the mocap rigid body being used to track it.
Two methods of performing this calibration were implemented. The first uses the motion capture markers directly, as outlined
in the accompanying paper. The second is our implementation of the calibration process described in \cite{tum}.

## Quickstart
This section describes how to reproduce the results from the paper. 
The bag files used total ~50 GB, so it is much more convenient to use the pre-extracted data 
(`board_data_10_9_18.npz` and `marker_data_10_9_18.npz`).

### Euclidean error
To reproduce the Euclidean error results run the following command in the base directory:

`python scripts/board_evaluation.py data/marker_calibration_10_9_18.npz data/board_calibration_10_9_18.npz data/board_data_10_9_18.npz`

### Reprojection error
To reproduce the reprojection error results run the following command in the base directory:

`python scripts/marker_evaluation.py data/marker_calibration_10_9_18.npz data/board_calibration_10_9_18.npz data/marker_data_10_9_18.npz`


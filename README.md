# Camera-mocap extrinsic calibration research
This code accompanies the paper "Calibration for Camera-Motion Capture Extrinsics". 
The purpose is to calculate the transformation between a camera and the mocap rigid body being used to track it.
Two methods of performing this calibration were implemented. The first uses the motion capture markers directly, as outlined
in the accompanying paper. The second is our implementation of the calibration process described in \cite{tum}.

The repository consists of a number of scripts which perform different parts of the calibration pipeline.
`extract_checkerboard_data.py` and `extract_marker_data.py` are used to extra
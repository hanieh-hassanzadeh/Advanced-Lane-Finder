# **Advanced Lane Finding Project**

The goal of this project is to accurately detect the lane lines captured by the camera centered on the center of an autonomous vehicle. The first challenge is that the available cameras distort the captured pictures. Therefore, I start with camera calibration.

### Camera Calibration

To calibrate the camera I used series of chessboard images (./camera_cal/calibration*.jpg) to computed the camera matrix and distortion coefficients using cv2.calibrateCamera function. These camera properties are then stored into the ../camera_calibration_parameters.pickle file." as mtx and dist, respectively, for a later use.

Compare an example of an original and an undistorted image, using the computed camera matrix and distortion coefficients, bellow

![undistortion](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/undistortion.jpg)

### Pipeline

The pipline of this project to extract the safe lane area (the area between two lanelines) from a camera captured video is consist of several steps. First off, I will explain how to go through these steps for a single road image. These steps can then be applied to series of consequent images of a video frames.

Consider a single image shown bellow:

![test image](https://github.com/hanieh-hassanzadeh/Advanced_Lane_Find/blob/master/test_images/test2.jpg)
 
#### 1. Correct distortion.

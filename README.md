# **Advanced Lane Finding Project**

The goal of this project is to accurately detect the lane lines and their curvature, captured by the camera centered on the center of an autonomous vehicle. 

### Camera Calibration

The first challenge is that the available cameras distort the captured pictures. Therefore, I start with camera calibration. To calibrate the camera I used series of chessboard images (```./camera_cal/calibration*.jpg```) to compute the camera matrix and distortion coefficients using ```cv2.calibrateCamera``` function. These camera properties are then stored into the ```../camera_calibration_parameters.pickle``` file as mtx and dist, respectively, for later use.

Compare the example of original image and undistorted image (obtained using the computed camera matrix and distortion coefficients) bellow

![undistortion](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/undistortion.jpg)

### Pipeline

The pipline of this project to extract the safe lane area (the area between two lanelines) from a camera captured video is consist of several steps (the pipeline is coded in ```./main.py```). First off, I will explain how to go through these steps for a single road image. These steps can then be applied to series of consecutive images of the video frames.

Consider a single image shown bellow:

![original](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/test_images/test2.jpg)
 

#### 1. Correct distortion
The first step is to undistort the image by ```cv2.undistort``` function utlizing the saved camera matrix and distortion coefficients; like this undistorted image for the above original image

![Undistort](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/undist_test2.jpg)

#### 2. Perspective transform
Unlike the suggested method in the course, I decided to do the Perspective transform before Color transform. I realized that playing around various color transforms to find a better solution in detecting lines is easier once I zoom in to my region of interest (ROI: the trapezium area consisting of the lanelines only). This step is done through ```imgwarpper``` function in 'main.py' file, which takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. In this function, I used ```cv2.getPerspectiveTransform``` function by hardcoding `src` and `dst` as:

```python
src = np.float32([[200, 720], [1050, 720], [595, 450], [685, 450]])
dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
```

The resulted warped image (bird view of the ROI) is then outputted by the ```imgwarpper``` function, as following image:

![warped](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/warp_test2.jpg)


#### 3. Color transforms

I tested different combimations of Color transforms and concluded that the two most effective ones are the Strength (S) chennel of the HLS format and the Sobel gradient in X direction with kernel size of 3. The Color transforms function is called ```colorTransform``` and is located in `lineDetectionThresholds.py` file.

The correcponding image is outputed as:

![alt text](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/warp_test2.jpg)

############### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


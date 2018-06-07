# **Advanced Lane Finding Project**

The goal of this project is to accurately detect the lane lines and their curvature, captured by the camera centered on the center of an autonomous vehicle. 

### Camera Calibration

The first challenge is that the available cameras distort the captured pictures. Therefore, I started with camera calibration. To calibrate the camera I used series of chessboard images (```./camera_cal/calibration*.jpg```) to compute the camera matrix and distortion coefficients using ```cv2.calibrateCamera``` function. These camera properties are then stored into the ```../camera_calibration_parameters.pickle``` file as mtx and dist, respectively, for later use.

Compare the example of original image and undistorted image (obtained using the computed camera matrix and distortion coefficients) bellow

![undistortion](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/undistortion.jpg)

### Pipeline

The pipline of this project to extract the safe_to_drive lane area (the area between two lanelines) from a camera captured video is consist of several steps (the pipeline is coded using `Lines()` class in ```./lines.py```). First off, I will explain how to go through these steps for a single road image. These steps can then be applied to series of consecutive images of the video frames.

Consider a single image shown bellow:

![original](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/test_images/test2.jpg)
 

#### 1. Correcting distortion
The first step is to undistort the image by `cv2.undistort` function utlizing the saved camera matrix and distortion coefficients; like this undistorted image for the above original image

![Undistort](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/undist_test2.jpg)

#### 2. Perspective transform
Unlike the suggested method in the course, I decided to do the Perspective transform before Color transform. I realized that playing around various color transforms to find a better solution in detecting lines is easier once I zoom in to my region of interest (ROI: the trapezium area consisting of the lanelines only). This step is done through `imgwarpper` function in 'main.py' file, which takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. In this function, I used `cv2.getPerspectiveTransform` function by hardcoding `src` and `dst` as:

```python
src = np.float32([[200, 720], [1050, 720], [595, 450], [685, 450]])
dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
```

The resulted warped image (bird view of the ROI) is then outputted by the `imgwarpper` function in `warp.py`, as following image:

![warped](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/warp_test2.jpg)


#### 3. Color transforms

I tested different combimations of Color transforms and concluded that the two most effective ones are the Strength (S) chennel of the HLS format and the Sobel gradient in X direction with kernel size of 3. The Color transforms function is called `colorTransform` and is located in `lineDetectionThresholds.py` file.

The correcponding image is outputed as:

![alt text](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/binary_test2.jpg)

#### 4. Identifying lane-line pixels and polynomial fitting

To detect the line pixels, defined a class called `Lines()` which detects/keeps track of current and previous detected lines and does sanity chechs and so on, as explained bellow. This class is located in `lines.py`.

I used sliding-window method for line detection. In this method, starting from the bottom of the image (for each left and right line, separately), a small window sweeps the image to find the best match between the line pixels and the template window (where the lines are located, most probably). When the whole image is swept, the line pixel within each detected window are considered to find the best fitted line (all done in `firstFrame` function).

From the second image on for the video frames, the position of the previous detected line is used as the base for the search step. The final line is obtained by fitting a second order line over the new pixels and the pixels found in the last two images. This is essential to have more accurate lines by preventing an abrupt change in the position of the line (all done in `non_firstFrame` function).

It is worth notting that in the beginning, I converted the noised-pixels which appear on the edges of the image into black pixels. These nioses mostly belong to the car dashboard in the bottom of the image, the sudden shades appear on the image, and the artificial lines which were added to the edges from previous processes. Here is the detected lines by sliding windows method.

![alt text](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/lined_test2.jpg)

##### Sanity check
It is very difficult to find a generalized `src` array which represent a perfect area containing the lane lines for all the image frames specially when changing the road curve. Therefore, not all the time the lines seem parallel in the warpped image. For the very reason, I do not check in the lines are parallel, or have the same distance. Instead, since there should not be a huge difference between the position of the lines in consequent frames, I check weather the pixel position of the current line is shifted more than 25 pixels in X direction for any Y value. If this condition is satisfied I use the combination of the pixels used to fit the previous line as well as the current ditected pixels, to fit the current line. This is coded at the end of the `non_firstFrame` function.

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center

In the same function, I calculate the curvature of each line pixels by fitting a second order line through each left and right set (`np.polyfit` is used for this purpose). Suposing that a line is defined as 
`X = a * Y^2 + b * Y + c` 
the curvature is defined as
`X = mX / (mY^2) *a*(Y^2) + (mX/mY)*b*Y + c`
where mX and mY are the X and Y converted to meters from pixel space. The left and the right lines curvature are annotated on the video.


#### 6. Final image

As the last step, I unwarpped the warped image using the `Minv`, and positioned the detected lane into the original image through ```imgUnwarpper``` function in ```main.py``` file. Here, is the resulted image with the lane shown in green.

![final](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputImages/annotated_test2.jpg)

---

### Pipeline (video)

The above mentioned steps are all done by defining `Lines` class in `lines.py` that processes each image. The same applies for each video frame. I did the slicing and writting back the resulted images into a video format using `VideoFileClip` functions of `moviepy.editor` package.

Here's a link to my video result:
![video](https://github.com/hanieh-hassanzadeh/Advanced-Lane-Finder/blob/master/outputvideo/project_video_annotated.mp4)

---

### Discussion

The techniues used here work the best when the road looks like the provided video. However when there are objects on the scenery which could be mistaken by the lane lines, this code may fail detecting the right lines. The roads with cracks or sharp shades of the vihecles which are changing their laned are some examples of these kinds.

Moreover, different countries have different standards for roads design, which may not fall into the default category, considered here; such as line patterns or colors. 

Some roads do not even have correct or any lines what so ever. This could be due to the recent construction or extention of the roads. And, some lines are faded due to insufficient maintenance.

The situation could also get worse to find accurate lines in night time or under rainy and snowy condition.

It seems that a perfect solution is not easy achieve. However, I highly suggect to use deep learning algoriths, which are closer to human brain in solving problems, as the base algorithm and combine them with computer vision solution, where helpful. 

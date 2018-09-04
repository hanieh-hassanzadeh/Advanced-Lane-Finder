import numpy as np
import matplotlib.image as mpimg
import os
import glob
import pickle
import cv2
from lines import *
from lineDetectionThresholds import *
from PIL import Image
from moviepy.editor import VideoFileClip
from warp import *

class processImages:
    def __init__(self, lines, mtx, dist):
        self.lines = lines
        self.mtx = mtx
        self.dist = dist
    def __call__(self, img):
        #1. Correct undistortion
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        #2. Perspective transform 
        Wimg, M, Minv = imgwarpper(undist, self.mtx, self.dist)
        #3. Color transfors
        binary_lines = colorTransform (Wimg)
        #4. Identifying lane-line pixels and polynomial fitting
        lined = lines.findLines(binary_lines)
        #5. Unwarping the warped image and positioning the detected lane into the original image 
        annotated = imgUnwarpper(img, lined, Minv)
        return annotated

#******************************************************************
if __name__ == '__main__':
    #Load saved camera matrix and distortion coefficients
    with open('camera_calibration_parameters.pickle', 'rb') as handle:
        camCal = pickle.load(handle)
    mtx  = camCal['mtx']
    dist = camCal['dist']

    process_images = False
    process_video  = True
    #########test images##############
    if process_images:
        lines = Lines()
        #Load the test images
        testImgFiles = glob.glob('../test_images/test*.jpg')

        #If the output directory doesn't exist, create one
        try:
            os.stat("../outputImages")
        except:
            os.mkdir("../outputImages")

        for file in testImgFiles:
            img = mpimg.imread(file)
            annotated = processImage(img, mtx, dist)

    #########test video##############
    if process_video:
        #If the output directory doesn't exist, create one
        try:
            os.stat("../outputVideo")
        except:
            os.mkdir("../outputvideo")

        outputVideo = '../outputVideo/project_video_annotated.mp4'

        clip1 = VideoFileClip("../test_video/project_video.mp4")
        clipImgs = clip1.fl_image(Lines(mtx, dist)) #This function expects color images!!
        #clip = clipImgs.subclip(3,43)#26)
        clipImgs.write_videofile(outputVideo, audio=False)

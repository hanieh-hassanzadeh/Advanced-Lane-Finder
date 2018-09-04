import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.ticker as tik
import matplotlib.pyplot as plt
from warp import *
from lineDetectionThresholds import *


# Define a class to determine the position of each line 
class Lines():
    def __init__(self, mtx, dist):
        #mtx and dist properties of camera
        self.mtx = mtx
        self.dist = dist
        # was the line detected in the last iteration?
        self.detected = False  
        # y values of the lines
        self.yfitted = [] 
        # x values of the detected pixels one frames back
        self.xL_1 = [np.array([False])]
        self.xR_1 = [np.array([False])]
        # y values of the detected pixels one frames back
        self.yL_1 = [np.array([False])]
        self.yR_1 = [np.array([False])]
        # x values of the previous fit
        self.fitxL_1 = [np.array([False])]
        self.fitxR_1 = [np.array([False])]
        #fits
        self.left_fit = None
        self.right_fit = None
        #self.n = 0
    
    def __call__(self, img):
        #1. Correct undistortion
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        #2. Perspective transform 
        Wimg, M, Minv = imgwarpper(undist)
        #3. Color transfors
        binary_lines = colorTransform (Wimg)
        #4. Identifying lane-line pixels and polynomial fitting
        lined, left_rad, right_rad = self.findLines(binary_lines)
        #5. Unwarping the warped image and positioning the detected lane into the original image 
        annotated = imgUnwarpper(img, lined, Minv, int(left_rad), int(right_rad))#, offset)
        return annotated

    
    def firstFrame(self, binary_warped):
        #removing some noises
        binary_warped[-50:, :100] = 0
        binary_warped[-50:, -100:] = 0
        binary_warped[150:, -50:] = 0
        binary_warped[:5, :] = 0
        binary_warped[-5:,:] = 0
        binary_warped[:, :5] = 0
        binary_warped[:,-5:] = 0

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
 
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds  = []
        right_lane_inds = []
 
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            ##cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
            ##cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
 
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
    
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
 
        ##########visualize
        # Generate x and y values for plotting
        self.yfitted = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left_fit[0]*self.yfitted**2 + self.left_fit[1]* self.yfitted + self.left_fit[2]
        right_fitx = self.right_fit[0]*self.yfitted**2 + self.right_fit[1]*self.yfitted + self.right_fit[2]
    
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        pntL = np.array([left_fitx, self.yfitted], np.int32).T
        pntL.reshape((-1,1,2))
        out_img = cv2.polylines(out_img, [pntL], False, (200, 200, 0), 5)
        pntR = np.array([right_fitx, self.yfitted], np.int32).T
        pntR.reshape((-1,1,2))
        out_img = cv2.polylines(out_img, [pntR], False, (200, 200, 0), 5)
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, self.yfitted]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx,self.yfitted])))])
        laneArea =np.hstack((left_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        window_img = np.zeros_like(out_img)
        cv2.fillPoly(window_img, np.int_([laneArea]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.5, 0)
        #cv2.imwrite('./outputImages/lined_test2.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        self.xL_1 = leftx
        self.xR_1 = rightx
        self.yL_1 = lefty
        self.yR_1 = righty
        self.fitxL_1 = left_fitx
        self.fitxR_1 = right_fitx

        return result, 0, 0


    def non_firstFrame(self, binary_warped):
        #sliding the window to sweep the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
            self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) +   
            self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
 
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
            self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
            self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))
 
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        left_fitx = self.left_fit[0]*self.yfitted**2 + self.left_fit[1]*self.yfitted + self.left_fit[2]
        right_fitx = self.right_fit[0]*self.yfitted**2 + self.right_fit[1]*self.yfitted + self.right_fit[2]
        ################Sanity check##################################
        thresh = 25
        maskL = abs(self.fitxL_1-left_fitx)>thresh
        maskR = abs(self.fitxR_1-right_fitx)>thresh

        if any(maskL):
            self.left_fit = np.polyfit(np.concatenate([lefty, self.yL_1]), np.concatenate([leftx, self.xL_1]), 2)
            left_fitx = self.left_fit[0]*self.yfitted**2 + self.left_fit[1]*self.yfitted + self.left_fit[2]
        if any(maskR):
            self.right_fit = np.polyfit(np.concatenate([righty, self.yR_1]), np.concatenate([rightx, self.xR_1]), 2)
            right_fitx = self.right_fit[0]*self.yfitted**2 + self.right_fit[1]*self.yfitted + self.right_fit[2]

        #############################################################
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.yfitted)
        left_curverad = ((1 + (2*self.left_fit[0]*y_eval + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        right_curverad = ((1 + (2*self.right_fit[0]*y_eval + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])
        # Example values: 1926.74 1908.48radiL = curvature()
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m
        ############################################################
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, self.yfitted]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx,self.yfitted])))])
        laneArea =np.hstack((left_line_window1, right_line_window2))  
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([laneArea]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.5, 0)
        self.xL_1 = leftx
        self.xR_1 = rightx
        self.yL_1 = lefty
        self.yR_1 = righty
        self.fitxL_1 = left_fitx
        self.fitxR_1 = right_fitx
        return result, left_curverad, right_curverad
 ############################################################
    def findLines(self, img):
        if self.detected == False:
            img2,left_curverad, right_curverad = self.firstFrame(img)
            self.detected = True
        else:
            img2,left_curverad, right_curverad = self.non_firstFrame(img)
        return img2, left_curverad, right_curverad

    def distFromCenter(self):
        #Distance from center of the road
        middle = self._image_size[1] / 2
        dist_left = (middle - self._left_line.allx[self._image_size[0]-1])
        dist_right = (self._right_line.allx[self._image_size[0]-1] - middle)

        return (dist_left - dist_right) * self._left_line.meters_per_pixel_x

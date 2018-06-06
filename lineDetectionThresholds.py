import numpy as np
import cv2

def absSobel(img, orient='x', minThresh=20, maxThresh=100):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= minThresh) & (scaled_sobel <= maxThresh)] = 1
	return binary_output

def HLS(img, tr_s = (170, 255)):
	#Convert RGB to HLS and threshold to binary image using S channel
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	s_channel = hls[:,:,2]
	binary_s = np.zeros_like(s_channel)
	binary_s[(s_channel > tr_s[0]) & (s_channel <= tr_s[1])] = 1
	return binary_s

def colorTransform(img):
        #Combining Sobel in X direction and Strength component of HLS format
	sobelX  = absSobel(img, orient='x', minThresh=30, maxThresh=150)
	hls_s = HLS(img, tr_s = (140, 255))

	combined = np.zeros_like(hls_s)
	combined[(hls_s == 1) | (sobelX == 1)] = 1

	return combined


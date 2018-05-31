import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as tik
import glob
import cv2
import pickle

def uploadCalImgs():
	calImages = glob.glob('./camera_cal/calibration*.jpg')
	return calImages

def calibrCamera(imageFiles):
	# Prepare object points
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

	# Arrays for later storing object points and image points
	objpoints = []
	imgpoints = []
	for file in imageFiles:
		img = mpimg.imread(file)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
		img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
		if ret == True:
			img_size = (img.shape[1], img.shape[0])
			objpoints.append(objp) 
			imgpoints.append(corners)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	return mtx, dist

imageFiles = uploadCalImgs()
mtx, dist = calibrCamera(imageFiles)

camCalParam = {"mtx" : mtx, "dist" : dist}
pickleFile = '../camera_calibration_parameters.pickle'
with open(pickleFile, 'wb') as handle:
	pickle.dump(camCalParam, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("Camera calibration parameters (mtx and dist) were written into the ../camera_calibration_parameters.pickle file.")


################################################
#plotting an example of an original and an undistorted image, using computed the camera matrix and distortion coefficients.
img    = mpimg.imread(imageFiles[0])
undist = cv2.undistort(img, mtx, dist, None, mtx)

fig, axes = plt.subplots(1, 2, figsize=(12, 5),
    subplot_kw={'xticks': [], 'yticks': []}, facecolor='white', frameon=False)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.gca().set_title('Original')
plt.gca().set_axis_off()

plt.subplot(1, 2, 2)
plt.imshow(undist)
plt.gca().set_title('Undistorted')

plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    hspace = 0, wspace = 0)
plt.margins(0,0)
plt.savefig('../outputImages/undistortion.jpg', bbox_inches = 'tight',pad_inches = 0)

print("\n Compare an example of an original and an undistorted image, using computed the camera matrix and distortion coefficients,")
print("plotted into ../outputImages/undistortion.jpg \n")

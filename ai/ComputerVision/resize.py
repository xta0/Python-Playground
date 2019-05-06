import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2

#1. Read in the first image of a stop sign
stop1 = mpimg.imread('images/stop_sign.jpg')

print('Image shape: ', stop1.shape)
plt.imshow(stop1)

#2. Read in the second image
stop2 = mpimg.imread('images/stop_sign2.jpg')

print('Image shape: ', stop2.shape)
plt.imshow(stop2)

#3. Crop this image so that it resembles the first image

# To crop and image, you can use image slicing 
# which is just slicing off a portion of the image array

# Make a copy of the image to manipulate
image_crop = np.copy(stop2)

# Define how many pixels to slice off the sides of the original image
 
row_crop = 90
col_crop = 250

# row: [90,-90]
# col: [250,-250]
# 这两个值是被crop的大小，crop后的尺寸为：[640 - 90*2 , 960 - 250*2] = [460,460]

# Using image slicing, subtract the row_crop from top/bottom and col_crop from left/right
image_crop = stop2[row_crop:-row_crop, col_crop:-col_crop, :]

plt.imshow(image_crop)

## 4. Resize the cropped image to be the same as the first

# Use OpenCV's resize function
standardized_im = cv2.resize(image_crop, (1389, 1500))

print('Image shape: ', standardized_im.shape)

# Plot the two images side by side
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Stop sign 1')
ax1.imshow(stop1)
ax2.set_title('Standardized stop sign 2')
ax2.imshow(standardized_im)

## 5. COmpare these images

# Sum all the red channel values and compare
red_sum1 = np.sum(stop1[:,:,0])
red_sum2 = np.sum(standardized_im[:,:,0])
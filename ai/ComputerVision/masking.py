import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2

# Read in the image
image = mpimg.imread('images/car_green_screen.jpg')

# Print out the image dimensions (height, width, and depth (color))
print('Image dimensions:', image.shape)

# Display the image
# plt.imshow(image)

# Define our color selection boundaries in RGB values
lower_green = np.array([0,180,0]) 
upper_green = np.array([100,255,100])

# # Define the masked area
# inRange 函数会将位于[lower_green, upper_green]区域内的像素至为255, 区域外的元素至为0
mask = cv2.inRange(image, lower_green, upper_green)
# 得到的mask是一个只有0和255的矩阵

# Vizualize the mask
plt.imshow(mask)
# plt.show()
# plt.imshow(mask, cmap='gray')


# Mask the image to let the car show through
masked_image = np.copy(image)
#将mask等于255的区域至为非黑色，mask!=0）设置为黑色(0,0,0) rbg三通道
masked_image[mask != 0] = [0, 0, 0]

# Display it!
# plt.imshow(masked_image)
# Load in a background image, and convert it to RGB 
background_image = mpimg.imread('images/sky.jpg')
print('background_image dimensions:', background_image.shape)

# Crop it or resize the background to be the right size (450x660)
row_crop = 62
col_crop = 182
# 图片高位奇数，从row_crop+1开始
background_image_crop = background_image[row_crop+1:-row_crop, col_crop:-col_crop, :]
print('background_image_crop dimensions:', background_image_crop.shape)

# copy image
crop_background = np.copy(background_image_crop)
# Mask the cropped background so that the car area is blocked
crop_background[mask == 0 ] = [0,0,0]
plt.imshow(crop_background)

# complete_image = masked_image + crop_background
complete_image =  masked_image + crop_background
plt.imshow(complete_image)
plt.show()


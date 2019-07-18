import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2

# Read in the image
image = mpimg.imread('images/car_green_screen2.jpg')

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)#(450, 660, 3) hsv是三维数组


#
# m1 = [[1,2,3],[2,3,4]]
# m2 = [[4,5,6],[5,6,7]]
# m3 = [[7,9,9],[8,7,8]]
# m = [m1,m2,m3] 
# m是3x2x3的三维数组, 3个 2x3的二维数组
# [1,2,3]  [4,5,6] [7,9,9]
# [2,3,4]  [5,6,7] [8,7,8]

# h = m[:,:,0] -> [ [1,2],[4,5],[7,8] ] 3x2的二维数组
# s = m[:,:,1] -> [ [2,3],[5,6],[9,7] ] 3x2的二维数组
# v = m[:,:,2] -> [ [3,4],[6,7],[9,8] ] 3x2的二维数组


# 分离HSV channels
h = hsv[:,:,0] #提取二维数组的第0列, 450x660 
s = hsv[:,:,1] #提取二维数组的第1列
v = hsv[:,:,2] #提取二维数组的第2列


# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('H channel')
ax1.imshow(h, cmap='gray')
ax2.set_title('S channel')
ax2.imshow(s, cmap='gray')
ax3.set_title('V channel')
ax3.imshow(v, cmap='gray')

# mask hue channel
lower_gray = 0 
upper_gray = 62
mask = cv2.inRange(h, lower_gray, upper_gray)

# Mask the image to let the car show through
masked_image = np.copy(image)
#将mask等于255的区域至为非黑色，mask!=0）设置为黑色(0,0,0) rbg三通道
masked_image[mask != 0] = [0, 0, 0]
# plt.imshow(masked_image)
plt.show()


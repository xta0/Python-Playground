import cv2 # computer vision library

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in a day and a night image
# These are directly extracted by name -- do not change
day_image = mpimg.imread("images/20151102_074952.jpg")
night_image = mpimg.imread("images/20151102_175445.jpg")


# Make these images the same size
width = 1100
height = 600
night_image = cv2.resize(night_image, (width, height))
day_image = cv2.resize(day_image, (width, height))

# Visualize both images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('night')
ax1.imshow(night_image)
ax2.set_title('day')
ax2.imshow(day_image)

def hsv_histograms(rgb_image):
    # Convert to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Create color channel histograms
    h_hist = np.histogram(hsv[:,:,0], bins=32, range=(0, 180))
    s_hist = np.histogram(hsv[:,:,1], bins=32, range=(0, 256))
    v_hist = np.histogram(hsv[:,:,2], bins=32, range=(0, 256))
    
    # Generating bin centers
    bin_edges = h_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

    # Plot a figure with all three histograms
    fig = plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.bar(bin_centers, h_hist[0])
    plt.xlim(0, 180)
    plt.title('H Histogram')
    plt.subplot(132)
    plt.bar(bin_centers, s_hist[0])
    plt.xlim(0, 256)
    plt.title('S Histogram')
    plt.subplot(133)
    plt.bar(bin_centers, v_hist[0])
    plt.xlim(0, 256)
    plt.title('V Histogram')
    
    return h_hist, s_hist, v_hist

# def rgb_histogram(rgb_image):
#     r_hist = np.histogram(rgb_image[:,:,0],bins=256,range=(0,256))
#     g_hist = np.histogram(rgb_image[:,:,1],bins=256,range=(0,256))
#     b_hist = np.histogram(rgb_image[:,:,2],bins=256,range=(0,256))

#      # Generating bin centers
#     bin_edges = r_hist[1] #[0-256]
#     # 柱状图中心
#     bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
#     # print(bin_centers)

#     #Plot a figure with all three histograms
#     fig = plt.figure(figsize=(12,3))
#     plt.subplot(131)
#     plt.bar(bin_centers, r_hist[0])
#     plt.xlim(0, 256)
#     plt.title('R Histogram')
#     plt.subplot(132)
#     plt.bar(bin_centers, g_hist[0])
#     plt.xlim(0, 256)
#     plt.title('G Histogram')
#     plt.subplot(133)
#     plt.bar(bin_centers, b_hist[0])
#     plt.xlim(0, 256)
#     plt.title('B Histogram')
    
#     return r_hist,g_hist,b_hist

# day_r_hist, day_g_hist, day_b_hist = rgb_histogram(day_image)
# night_r_hist, night_g_hist, night_b_hist = rgb_histogram(night_image)
# plt.show()

# Call the function for "night"
night_h_hist, night_s_hist, night_v_hist = hsv_histograms(night_image)
# Call the function for "day"
day_h_hist, day_s_hist, day_v_hist = hsv_histograms(day_image)

# Which bin do most V values fall in?
# Does the Hue channel look helpful?
# What patterns can you see that might distinguish these two images?

# Out of 32 bins, if the most common bin is in the middle or high up, then it's likely day
fullest_vbin_day = np.argmax(day_v_hist[0])
fullest_vbin_night = np.argmax(night_v_hist[0])


print('Fullest Value bin for day: ', fullest_vbin_day)
print('Fullest Value bin for night: ', fullest_vbin_night)

# Sum the V component of the day image and compare the two
# Convert the night image to HSV colorspace
hsv_night = cv2.cvtColor(night_image, cv2.COLOR_RGB2HSV)

# Isolate the V component
v = hsv_night[:,:,2]
print(v.shape) #600 x 1100

# Sum the V component over all columns (axis = 0)
# axis=0 表示按列相加， axis=1表示按行相加, 没有axis表示全部相加
v_sum = np.sum(v[:,:], axis=0)
print(v_sum) #1x1100

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

ax2.set_title('Value sum over columns')
ax1.plot(v_sum)

ax2.set_title('Original image')
ax2.imshow(night_image, cmap='gray')

## 这个结果表明越亮的地方sum的值越高
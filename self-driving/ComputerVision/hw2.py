import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

# IMAGE_DIR_TRAINING = "images/traffic-light/training/"
# IMAGE_DIR_TEST = "images/traffic-light/testing/"


pics = [
"images/traffic-light/testing/green/1d5a9390-81dd-4f09-9c32-4436cb90bb95.jpg",
"images/traffic-light/testing/green/7bfade00-0869-4482-8b9a-1028910397d1.jpg",
"images/traffic-light/testing/green/6e33a55e-8f13-46d0-bd82-0a4d021e05e3.jpg",
"images/traffic-light/testing/green/0b3606b7-bf9e-49d8-8de8-801bb8374b2d.jpg",
"images/traffic-light/testing/green/3a134375-f8d2-4fb9-bd8b-e76e8c1d9305.jpg",
"images/traffic-light/testing/green/0be91609-32ba-4b7b-b460-cc47dd62b740.jpg",
"images/traffic-light/testing/yellow/3b9d130d-3725-440d-867a-7e8a04603a97.jpg",
"images/traffic-light/testing/green/1e3af45f-6fd3-4c8b-9c9e-f8e504b868d7.jpg",
"images/traffic-light/testing/green/2e368899-ff57-48f9-ae8c-4bd35cf6c402.jpg"


]




# Using the load_dataset function in helpers.py
# Load training data
# IMAGE_LIST = helpers.load_traffic_dataset(IMAGE_DIR_TRAINING)
def crop_image(image):
    row_crop = 4
    col_crop = 12   
    return image[row_crop:-row_crop,col_crop:-col_crop,:]


def testHSV(rgb_image):

    cropped_img = crop_image(rgb_image)
    # Convert to HSV
    hsv  = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)

    # HSV channels
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    sum_brightness = np.sum(hsv[:,:,2])
    area = h.shape[0]*h.shape[1]  # pixels

    # find the avg
    # create a mask using v channel
    lower_v = sum_brightness/area
    upper_v = 255
    print("lower:", lower_v, ", ",upper_v)
    mask = cv2.inRange(v, lower_v, upper_v)
    
    # Mask the image to let the color show through
    masked_image = np.copy(v)
    masked_image[mask != 255] = 0

    # Plot the original image and the three channels
    f, [(ax1, ax2, ax3), (ax4,ax5,ax6)]  = plt.subplots(2, 3, figsize=(20,10))

    ax1.set_title('cropped image')
    ax1.imshow(cropped_img)
    ax2.set_title('H channel')
    ax2.imshow(h, cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(s, cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(v, cmap='gray')
    ax5.set_title('Mask')
    ax5.imshow(mask, cmap='gray')
    ax6.set_title('V Mask')
    ax6.imshow(masked_image, cmap='gray')

# Classify image using HSV channel
def classify_image(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    # HSV channels
    h = hsv[:,:,0] #0-180
    s = hsv[:,:,1] #0-255
    v = hsv[:,:,2] #0-255

    sum_brightness = np.sum(hsv[:,:,2])
    area = h.shape[0]*h.shape[1]  # pixels

    # find the avg
    # create a mask using v channel
    lower_v = sum_brightness/area
    upper_v = 255
    # print("lower:", lower_v, ", ",upper_v)
    mask = cv2.inRange(v, lower_v, upper_v)
    
    # Mask the image to let the color show through
    masked_image = np.copy(v)
    masked_image[mask != 255] = 0
    
    # drawMask(rgb_image,v,masked_image)
    
    RH1,RH2,RS,RV = range(160,181),range(0,10),range(43,256),range(46,256)
    YH, YS, YV = range(10,60),range(43,256),range(45,256)
    GH, GS, GV = range(70,100),range(43,256),range(46,256)

    red = 0
    yellow = 0
    green = 0
    
    h,w = masked_image.shape
    for i in range(h):
        for j in range(w):
            if masked_image[i][j] != 0:
                h = hsv[i][j][0]
                s = hsv[i][j][1]
                v = hsv[i][j][2]
                if (h in RH1 or h in RH2) and (s in RS) and (v in RV):
                    red += 1
                if (h in YH) and (s in YS) and (v in YV):
                    yellow += 1
                if (h in GH) and (s in GS) and (v in GV):
                    green += 1
  
    print("red", red)
    print("green", green)
    print("yellow", yellow)


    if red > green and red > yellow:
        return [1,0,0]
    if green > red and green > yellow:
        return [0,0,1]
    if yellow > red and yellow > green:
        return [0,1,0]
    
    return [1,0,0]



def estimate_label(rgb_image):
    return classify_image(crop_image(rgb_image))

# for url in pics:
# print(len(pics))
image = helpers.load_image(pics[0])
standard_im = np.copy(image)
standard_im = cv2.resize(standard_im,(32,32))
result = estimate_label(image)
print(str(result))
testHSV(standard_im)
plt.show()



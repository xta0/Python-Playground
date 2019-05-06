import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

IMAGE_DIR_TRAINING = "images/traffic-light/training/"
IMAGE_DIR_TEST = "images/traffic-light/testing/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_traffic_dataset(IMAGE_DIR_TRAINING)


# resize all the images to 32x32
def standardize_input(image):
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im,(32,32))
    return standard_im

def one_hot_encode(label):
    color_map = ["red", "yellow", "green"]
    one_hot = [0] * len(color_map)
    try:
        one_hot[color_map.index(label)] = 1
        return one_hot
    except:
        raise TypeError('Please input red, yellow, or green. Not ', label)


def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]
        url = item[2]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label,url))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

def crop_image(image):
    row_crop = 4
    col_crop = 12    
    return image[row_crop:-row_crop,col_crop:-col_crop,:]

def randomImage():
    r = random.randint(0,723)
    y = random.randint(723,758)
    g = random.randint(758, 1187)
    
    img_r = crop_image(STANDARDIZED_LIST[r][0])
    img_y = crop_image(STANDARDIZED_LIST[y][0])
    img_g = crop_image(STANDARDIZED_LIST[g][0])
    return (img_r,img_y,img_g)

def testHSV():
    (test_im_red,test_im_yellow,test_im_green) = randomImage()
    #test_label = STANDARDIZED_LIST[image_num][1]

    # Convert to HSV
    hsv_red    = cv2.cvtColor(test_im_red, cv2.COLOR_RGB2HSV)
    hsv_green  = cv2.cvtColor(test_im_green, cv2.COLOR_RGB2HSV)
    hsv_yellow = cv2.cvtColor(test_im_yellow, cv2.COLOR_RGB2HSV)

    # Print image label
    # print('Label [red, yellow, green]: ' + str(test_label))

    # HSV channels
    # h = hsv[:,:,0]
    # s = hsv[:,:,1]
    # v = hsv[:,:,2]

    # Plot the original image and the three channels
    f, [(ax1, ax2, ax3, ax4),(ax6, ax7, ax8, ax9),(ax10, ax11, ax12, ax13)] = plt.subplots(3, 4, figsize=(20,10))

    # red
    ax1.set_title('Standardized image')
    ax1.imshow(test_im_red)
    ax2.set_title('H channel')
    ax2.imshow(hsv_red[:,:,0], cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(hsv_red[:,:,1], cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(hsv_red[:,:,2], cmap='gray')

    #yellow
    ax6.set_title('Standardized image')
    ax6.imshow(test_im_yellow)
    ax7.set_title('H channel')
    ax7.imshow(hsv_yellow[:,:,0], cmap='gray')
    ax8.set_title('S channel')
    ax8.imshow(hsv_yellow[:,:,1], cmap='gray')
    ax9.set_title('V channel')
    ax9.imshow(hsv_yellow[:,:,2], cmap='gray')
    
    #green
    ax10.set_title('Standardized image')
    ax10.imshow(test_im_green)
    ax11.set_title('H channel')
    ax11.imshow(hsv_green[:,:,0], cmap='gray')
    ax12.set_title('S channel')
    ax12.imshow(hsv_green[:,:,1], cmap='gray')
    ax13.set_title('V channel')
    ax13.imshow(hsv_green[:,:,2], cmap='gray')
    

def drawMask(rgb_image, v_image, mask_image):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.set_title('Standardized image')
    ax1.imshow(rgb_image)
    ax2.set_title('V channel')
    ax2.imshow(v_image, cmap='gray')
    ax3.set_title('Masked Image')
    ax3.imshow(mask_image, cmap='gray')
    
# testHSV()



def create_feature(rgb_image):
   # Convert to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2] #0-255
    
    #divide the matrix vertically into 3 regions
    arr = np.array_split(v,3,axis=0)
    
    #sum each region
    red = np.sum(arr[0])
    yellow = np.sum(arr[1])
    green = np.sum(arr[2])
    
    return (red,yellow,green)

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
    mask = cv2.inRange(v, lower_v, upper_v)
    
    # Mask the image to let the color show through
    masked_image = np.copy(v)
    masked_image[mask != 255] = 0
    
    # drawMask(rgb_image,v,masked_image)
    
    RH1,RH2,RS,RV = range(150,181),range(0,10),range(43,256),range(46,256)
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
                if (h in RH1 or h in RH2) and s in RS and v in RV:
                    red += 1
                if h in YH and s in YS and v in YV:
                    yellow += 1
                if h in GH and s in GS and v in GV:
                    green += 1
  
    return (red,yellow,green)


def estimate_label(rgb_image):
    #1. using hsv color pixels
    # red, yellow, green = classify_image(crop_image(rgb_image))
    #2. using brightness 
    red, yellow, green = create_feature(crop_image(rgb_image))
    
    if red > green and red > yellow:
        return [1,0,0]
    if green > red and green > yellow:
        return [0,0,1]
    if yellow > red and yellow > green:
        return [0,1,0]
    
    #default value is red
    return [1,0,0]


##########
# Test
##########
# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_traffic_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        url = image[2]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            print(url)
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))


# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as
f, matrix = plt.subplots(3, 3, figsize=(20,10))
for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        ax = matrix[i][j]
        if i*3 + j < len(MISCLASSIFIED):
            data = MISCLASSIFIED[i*3+j]
            ax.set_title(str(data[1])+"-->"+str(data[2]))
            ax.imshow(data[0])

plt.show()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import math

json_input = "example_json/input.json"


def read_from_json(inputDir):
    with open(inputDir) as json_data:
        return json.load(json_data)["image_path"]

def write_to_json(outputDir, contentsList):
    with open(outputDir, 'w') as f:
        json.dump(contentsList, f) # FIXME: Convert list into dict?

def readImage(images, imageIndex):
    img_pool_table = cv2.imread(images[imageIndex])

    # Show images
    f, axarr = plt.subplots(1, 2, figsize=(12, 4))
    axarr[0].imshow(cv2.cvtColor(img_pool_table, cv2.COLOR_BGR2RGB))
    axarr[0].set_title("Pool Table")
    axarr[0].axis("off")
    plt.show()

    return img_pool_table


def detect_table(image):
    # 1. Filter table by applying a blue mask
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL) # HSV enables for easier color identification
    lower_blue = np.array([144, 100, 100])
    upper_blue = np.array([200, 255, 255])

    mask_img = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # TODO: Remove? This should remove noise but doesn't make any difference
    #kernel = np.ones((5,5), np.uint8)
    #mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
    #mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

    result_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask_img)

    # 2. Convert the colors again
    result_img = cv2.cvtColor(result_img, cv2.COLOR_HSV2RGB)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)

    # 3. Apply Hough circles to detect balls
    result_img = cv2.medianBlur(result_img, 5)
    
    circles = cv2.HoughCircles(result_img, cv2.HOUGH_GRADIENT, dp=1, minDist=60, param1=50, param2=30, minRadius=15, maxRadius=50)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(result_img,(i[0],i[1]),i[2],(255,255,0),10)
    
    # Show images
    f, axarr = plt.subplots(1, 2, figsize=(12, 4))
    axarr[0].imshow(cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR))
    axarr[0].set_title("Pool Table")
    axarr[0].axis("off")
    plt.show()


images = read_from_json(json_input)

poolTableImage = readImage(images, 30)

detect_table(poolTableImage)


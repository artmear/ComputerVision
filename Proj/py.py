from turtle import color

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



##### Number identification #####
from enum import Enum

class BallColor(Enum):
    WHITE  = 0
    YELLOW = 1
    BLUE   = 2
    RED    = 3
    PURPLE = 4
    ORANGE = 5
    GREEN  = 6
    BROWN  = 7
    BLACK  = 8


class BallType(Enum):
    CUE     = 0
    SOLID   = 1
    STRIPED = 2


BALL_MAPPING = {
    # Cue ball
    (BallColor.WHITE,  BallType.CUE): 0,

    # Solid balls
    (BallColor.YELLOW, BallType.SOLID): 1,
    (BallColor.BLUE,   BallType.SOLID): 2,
    (BallColor.RED,    BallType.SOLID): 3,
    (BallColor.PURPLE, BallType.SOLID): 4,
    (BallColor.ORANGE, BallType.SOLID): 5,
    (BallColor.GREEN,  BallType.SOLID): 6,
    (BallColor.BROWN,  BallType.SOLID): 7,
    (BallColor.BLACK,  BallType.SOLID): 8,

    # Striped balls
    (BallColor.YELLOW, BallType.STRIPED): 9,
    (BallColor.BLUE,   BallType.STRIPED): 10,
    (BallColor.RED,    BallType.STRIPED): 11,
    (BallColor.PURPLE, BallType.STRIPED): 12,
    (BallColor.ORANGE, BallType.STRIPED): 13,
    (BallColor.GREEN,  BallType.STRIPED): 14,
    (BallColor.BROWN,  BallType.STRIPED): 15
}


# FIXME: Might need to be adjusted!!!
COLOR_RANGES = {
    BallColor.WHITE:  [(0, 0, 200), (180, 40, 255)],
    BallColor.BROWN:  [(5, 80, 50), (25, 255, 180)],
    BallColor.RED:    [(0, 90, 100), (12, 255, 255), (170, 90, 100), (180, 255, 255)], # Red can wrap around the hue range, so we need two ranges
    BallColor.ORANGE: [(8, 90, 120), (30, 255, 255)],
    BallColor.YELLOW: [(20, 120, 120), (35, 255, 255)],
    BallColor.GREEN:  [(40, 100, 80), (85, 255, 255)],
    BallColor.BLUE:   [(95, 120, 80), (130, 255, 255)],
    BallColor.PURPLE: [(130, 100, 80), (160, 255, 255)],
    BallColor.BLACK:  [(0, 0, 0), (180, 255, 16)]
}


''' Get the mean color of an image, excluding the center (number circle) '''
def get_mean_color(rgb_img):
    # Find the center and radius for the circular mask # FIXME: This is not going to work if the ball doesn't have the correct rotation
    h, w = rgb_img.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.45)

    # Create a circular mask to exclude the center area
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1) 

    # Compute the mean color using the mask
    mean_bgr = cv2.mean(rgb_img, mask=mask)[:3]
    mean_bgr = np.uint8([[mean_bgr]])

    # Convert mean color to HSV for easier color classification
    mean_hsv = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2HSV)[0][0]

    return mean_hsv


''' Determine if the ball is solid or striped, based on the ratio of white pixels in the outer area '''
def get_ball_type(hsv_img):
    # Mask to ignore center (number circle) # FIXME: This is not going to work if the ball doesn't have the correct rotation
    h, w = hsv_img.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.2)

    # Exclude the center area by drawing a filled black circle on the mask
    cv2.circle(mask, center, radius, 0, -1)

    # Detect white
    white_mask = cv2.inRange(hsv_img, (0, 0, 180), (180, 60, 255))

    # Apply center exclusion
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=mask)

    # Calculate the ratio of white pixels in the outer area
    white_ratio = np.sum(white_mask > 0) / np.sum(mask > 0)

    # TODO: Remove? (DEBUG)
    print("White ratio (outer only):", white_ratio)

    # Classify as striped if the white ratio exceeds a certain threshold (tuned based on testing)
    if white_ratio > 0.12: # TODO: Adjust threshold?
        return BallType.STRIPED
    else:
        return BallType.SOLID


''' Classify the ball color based on the mean HSV value and predefined color ranges '''
def classify_color(hsv_color):
    h, s, v = map(int, hsv_color)

    # Iterate through the color ranges and check if the HSV value falls within any of them
    for color, ranges in COLOR_RANGES.items():
        for i in range(0, len(ranges), 2):
            lower = ranges[i]
            upper = ranges[i+1]
            if (lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]):
                return color

    return None
    

# TODO: Remove? (DEBUG function to show ball roi + detected color)
def show_ball_and_color(ball_roi, color_hsv):
    color_hsv_int = np.uint8([[color_hsv]])
    display_color_rgb = cv2.cvtColor(color_hsv_int, cv2.COLOR_HSV2RGB)[0][0]

    color_img = np.full((100, 100, 3), display_color_rgb, dtype=np.uint8)
    f, axarr = plt.subplots(1, 2, figsize=(12, 4))
    axarr[0].imshow(cv2.cvtColor(ball_roi, cv2.COLOR_BGR2RGB))
    axarr[0].set_title("Ball ROI")
    axarr[0].axis("off")
    axarr[1].imshow(color_img)
    axarr[1].set_title("Dominant Color")
    axarr[1].axis("off")
    plt.show()

# TODO: Remove (DEBUG to show HSV ranges in different scale)
def print_hsv_ranges():
    for color, ranges in COLOR_RANGES.items():
        print(f"{color.name}:")
        for i in range(0, len(ranges), 2):
            lower = ranges[i]
            upper = ranges[i+1]
            lower_pct = (lower[0] * 2, (lower[1] / 255) * 100, (lower[2] / 255) * 100)
            upper_pct = (upper[0] * 2, (upper[1] / 255) * 100, (upper[2] / 255) * 100)
            #Rounded to int
            lower_pct = tuple(map(int, lower_pct))
            upper_pct = tuple(map(int, upper_pct))
            print(f"  Range {i//2 + 1}: H: {lower_pct[0]}-{upper_pct[0]}, S: {lower_pct[1]}%-{upper_pct[1]}%, V: {lower_pct[2]}%-{upper_pct[2]}%")


''' Main function to detect ball color and type, and map to ball number '''
def detect_ball_color(ball_roi):
    # Convert to HSV for easier color identification
    hsv_img = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)
    number = None

    # Compute dominant color
    color_hsv = get_mean_color(ball_roi)
    hsv_in_pct = (color_hsv[0] * 2, (color_hsv[1] / 255) * 100, (color_hsv[2] / 255) * 100)

    # Determine ball type, based on white ratio
    ball_type = get_ball_type(hsv_img)

    # Classify ball color
    classified_color = classify_color(color_hsv)

    # Map to ball number
    number = BALL_MAPPING.get((classified_color, ball_type), None)
    
    # Deal with special cases
    # TODO: Is this needed?
    
    ### TODO: Remove? (DEBUG) ###
    show_ball_and_color(ball_roi, color_hsv)
    print("Dominant color (HSV):", hsv_in_pct[0], hsv_in_pct[1], "%", hsv_in_pct[2], "%")
    print("Dominant color (RGB):", cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2RGB)[0][0])
    print("Detected BallColor:", color_hsv)
    print("Classified color:", classified_color)
    print("Ball number:", number)
    print_hsv_ranges()
    #############################

    return ball_type, number


ball_test_img = cv2.imread("ball_test.png")
print(detect_ball_color(ball_test_img))



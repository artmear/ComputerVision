import numpy as np
import cv2
import os
from debug import save_debug_image
from warp_table import warp_table, order_points_clockwise


def get_table_width_height(image, image_name, output_dir):
    # Create the table mask
    table_mask, quad = warp_table(image, image_name, output_dir)

    # Order corner points consistently (tl, tr, br, bl) top-left, top-right...
    src = order_points_clockwise(quad)
    (tl, tr, br, bl) = src

    # Compute output rectangle size
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(1, int(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(1, int(max(heightA, heightB)))

    # Define destination points for bird's-eye view transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    return maxWidth, maxHeight

#
# TODO: remove, already defined in another branch
#
def crop_ball_roi(image, x, y, r, pad=4):
    h, w = image.shape[:2]
    r = int(r) + pad
    x, y = int(x), int(y)

    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(w, x + r)
    y2 = min(h, y + r)

    roi = image[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

def detect_balls(image, image_name, output_dir, debugOutput):

    # Milimiters
    POOL_TABLE_LENGTH = 254*10
    POOL_BALL_DIAMETER = 58

    # Diameter of a ball is now aproximatly POOL_TABLE_BALL_RATIO of the width of the table
    POOL_TABLE_BALL_RATIO = POOL_BALL_DIAMETER/POOL_TABLE_LENGTH

    # Warp the table and get the transformation matrix
    width, height = get_table_width_height(image, image_name, output_dir)

    # Pool tables have width > height so we can identify the size by checking which is the biggest number:
    if (height > width):
        width = height

    table_mask, quad = warp_table(image, image_name, output_dir)
    ima = cv2.bitwise_and(image, image, mask=table_mask)

    # Before applying the HoughCircles we need to convert the image to grayscale
    # otherwise it will fail. We can also blur the image to remove some of the noise that may interfere.
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    # Detect circles using Hough Transform
    ballRadius = (width*POOL_TABLE_BALL_RATIO)/2.0
    offset = 12

    # Slightly less than a full diameter (2 * radius)
    minDist = ballRadius * 1.8
    param1 = 40
    param2 = 13
    minRadius = int(ballRadius)
    maxRadius = int(ballRadius + offset)

    print(f"[INFO] minDist={minDist}, param1={param1}, param2={param2}, minRadius={minRadius}, maxRadius={maxRadius}")

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,  minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # Draw the detected balls on the original angled image
    output_image = image.copy()
    resultingCircles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        near_black_value_threshold = 62
        for circle_idx, i in enumerate(circles[0, :], start=1):
            x, y, r = i[0], i[1], i[2]

            # Crop the image to just the detected circle.
            (hsv_image_cropped, _) = crop_ball_roi(hsv_image, x, y, 10, pad=0)
            hue_values = hsv_image_cropped[:, :, 0]

            # Find the dominant hue
            hist, bins = np.histogram(hue_values, bins=180, range=(0, 180))
            dominant_hue = np.argmax(hist)
            value_values = hsv_image_cropped[:, :, 2]
            mean_value = float(np.mean(value_values))
            is_near_black = mean_value <= near_black_value_threshold

            if debugOutput:
                base_name = os.path.splitext(image_name)[0]
                hue_debug_path = os.path.join(output_dir, f"{base_name}_hue_image_cropped_{circle_idx:02d}.png")
                cv2.imwrite(hue_debug_path, cv2.cvtColor(hsv_image_cropped, cv2.COLOR_HSV2BGR))
                print(f"[DEBUG] hue_image_cropped saved in: {hue_debug_path} and dominant hue is {dominant_hue}, mean V is {mean_value:.1f}, near_black={is_near_black}")

            if (dominant_hue > 90 and dominant_hue < 103) and (not is_near_black):
                print(f"Color is {dominant_hue}")
                continue

            

            resultingCircles.append([x,y,r])
            

            # Draw the outer circle
            cv2.circle(output_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(output_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Save output for debugging
    save_debug_image(output_image, image_name, "debug/ball_detection")
    
    return resultingCircles

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import math

##### DEBUG FUNCTIONS (TODO: Remove?) #####
def debug_dominant_color(ball_roi, hsv_img, mask, dominant_hue):
    # Display original ROI, mask, hue histogram, and a swatch of the detected dominant hue.
    masked_hue = hsv_img[:, :, 0][mask.astype(bool)]
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(ball_roi, cv2.COLOR_BGR2RGB))
    plt.title("Original Ball ROI")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask Used for Color Detection")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.hist(masked_hue, bins=180, range=(0, 180), color='blue', alpha=0.7)
    plt.axvline(dominant_hue, color='red', linestyle='--', label=f'Dominant Hue: {dominant_hue}')
    plt.title("Hue Histogram of Masked Region")
    plt.xlabel("Hue Value")
    plt.ylabel("Pixel Count")
    plt.legend()

    plt.subplot(1, 4, 4)
    if dominant_hue < 0:
        swatch_rgb = np.full((120, 120, 3), 128, dtype=np.uint8)
        swatch_title = "Dominant Color: N/A"
    else:
        hsv_color = np.uint8([[[dominant_hue, 255, 255]]])
        swatch_rgb = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        swatch_rgb = np.full((120, 120, 3), swatch_rgb, dtype=np.uint8)
        swatch_title = f"Dominant Color (Hue={dominant_hue})"
    plt.imshow(swatch_rgb)
    plt.title(swatch_title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


""" This function extracts the dominant hue value of an HSV image, by using histogram analysis """
def get_dominant_hue(hsv, combined_mask):
    hue_values = hsv[:, :, 0][combined_mask.astype(bool)]
    if len(hue_values) == 0:
        dominant_hue = -1  # cue ball or bad mask
    else:
        # Use histogram to find dominant hue (more robust than mean)
        hist, bins = np.histogram(hue_values, bins=180, range=(0, 180))
        dominant_hue = np.argmax(hist)
    return dominant_hue


""" Map the detected dominant hue and stripe presence to a specific ball number."""
def hue_to_ball_number(hue, is_striped):
    if hue is None: # Cue ball (white)
        return 0  

    # Solid balls: 1–8 | Striped balls: 9–15 (same colors, +8)
    offset = 8 if is_striped else 0

    # opencv hue ranges from 0-179
    if hue < 10 or hue > 165:   return 3 + offset   # Red
    elif 10 <= hue < 25:         return 7 + offset  # Brown
    elif 25 <= hue < 35:         return 5 + offset  # Orange
    elif 35 <= hue < 55:         return 1 + offset  # Yellow
    elif 55 <= hue < 85:         return 6 + offset  # Green
    elif 85 <= hue < 130:        return 2 + offset  # Blue
    elif 130 <= hue < 155:       return 4 + offset  # Purple
    elif 155 <= hue <= 165:      return 3 + offset  # Red (wraps around)
    else:
        return -1  # Unknown


""" This function checks for the presence of a white stripe across the ball"""
def is_striped_ball(hsv_img):
    # Stripes = significant white band across the middle of the ball.
    h, w = hsv_img.shape[:2]
    
    # Use full ball mask, no saturation filtering
    full_mask = get_circular_mask(hsv_img, 0.45)
    
    # Look for white pixels (low saturation, high brightness)
    white_px = (hsv_img[:, :, 1] < 50) & (hsv_img[:, :, 2] > 160) & full_mask.astype(bool)
    white_ratio = np.sum(white_px) / np.sum(full_mask)
    
    # Stripes have a notable white band
    return 0.10 < white_ratio < 0.65  # upper bound avoids misclassifying near-cue-ball whites


""" Mask for hue analysis: exclude glare, white stripe, and very dark pixels."""
def get_color_mask(hsv_img):
    mask = get_circular_mask(hsv_img, 0.42)
    brightness_ok  = (hsv_img[:, :, 2] > 40) & (hsv_img[:, :, 2] < 235)
    saturation_ok  = hsv_img[:, :, 1] > 45
    return mask & brightness_ok & saturation_ok


""" Create a circular mask for the given HSV image """
def get_circular_mask(hsv, radius_fraction=0.45):
    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(w, h) * radius_fraction), 255, -1)
    return mask


""" Detect single ball color and type, and map to ball number """
def detect_ball_color(ball_roi):
    hsv_img = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)

    # Special cases first
    # FIXME: This algorithm currently struggles with striped, brownish, black and white balls.

    # Stripe detection uses the unfiltered mask
    is_striped = is_striped_ball(hsv_img)

    color_mask = get_color_mask(hsv_img)
    dominant_hue = get_dominant_hue(hsv_img, color_mask)
    number = hue_to_ball_number(dominant_hue, is_striped)

    debug_dominant_color(ball_roi, hsv_img, color_mask, dominant_hue)

    return is_striped, number


""" Crop the ball region of interest (ROI) from the original image, given the detected circle parameters """
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


""" Main function to detect all balls numbers and types. Return their types and numbers """
def detect_ball_numbers(image, circles):
    balls = []
    if circles is None:
        return balls

    circles = np.asarray(circles)

    # If it's OpenCV HoughCircles output: (1, N, 3) -> (N, 3)
    if circles.ndim == 3 and circles.shape[0] == 1 and circles.shape[2] == 3:
        circles = circles[0]

    for (x, y, r) in circles:
        ball_roi, bbox = crop_ball_roi(image, x, y, r)
        if ball_roi.size == 0:
            continue
        ball_type, number = detect_ball_color(ball_roi)
        balls.append(((int(x), int(y), int(r)), bbox, ball_type, number))

    return balls


""" Visualize the detected balls by drawing bounding boxes and placing the detected number below each ball ROI """
def show_detected_balls_on_image(image, balls):
    display_img = image.copy()

    for (_, _, _), (x1, y1, x2, y2), _, number in balls:
        # Draw ROI bounding box
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Place the detected number slightly below the ROI
        label = "?" if number is None else str(number)
        text_x = x1
        text_y = min(display_img.shape[0] - 5, y2 + 20)
        cv2.putText(display_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Detected Balls (Number Under ROI)")
    ax.axis("off")
    plt.show()

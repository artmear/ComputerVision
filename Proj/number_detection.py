import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import math
from debug import debug_dominant_color, show_detected_balls_on_image

# Color hue ranges
LOWER_RED_HUE = 165 
UPPER_RED_HUE = 10
UPPER_BROWN_HUE = 25
UPPER_ORANGE_HUE = 35
UPPER_YELLOW_HUE = 55
UPPER_GREEN_HUE = 85
UPPER_BLUE_HUE = 130
UPPER_PURPLE_HUE = 155

# Ball class to store all relevant information about each detected ball
class Ball:
    def __init__(self, circle_params, bbox, ball_type, number, dominant_hue, histogram=None):
        self.circle_params = circle_params  # (x, y, r)
        self.bbox = bbox                    # (x1, y1, x2, y2)
        self.ball_type = ball_type          # solid or striped
        self.number = number                # detected ball number
        self.dominant_hue = dominant_hue    # hue value used for detection
        self.histogram = histogram          # hue histogram for debugging/confidence analysis



""" This function extracts the dominant hue value of an HSV image, by using histogram analysis """
def get_dominant_hue(hsv, combined_mask):
    hist = None
    hue_values = hsv[:, :, 0][combined_mask.astype(bool)]
    if len(hue_values) == 0:
        dominant_hue = -1  # cue ball or bad mask
    else:
        # Use histogram to find dominant hue (more robust than mean)
        hist, bins = np.histogram(hue_values, bins=180, range=(0, 180))
        dominant_hue = np.argmax(hist)
    return dominant_hue, hist



""" Map the detected dominant hue and stripe presence to a specific ball number."""
def hue_to_ball_number(hue, is_striped):
    if hue is None: # Cue ball (white)
        return 0  

    # Solid balls: 1-8 | Striped balls: 9-15 (same colors, +8)
    offset = 8 if is_striped else 0

    # opencv hue ranges from 0-179
    if hue < UPPER_RED_HUE or hue >  LOWER_RED_HUE:    return 3 + offset   # Red
    elif UPPER_RED_HUE     <= hue <  UPPER_BROWN_HUE:  return 7 + offset  # Brown
    elif UPPER_BROWN_HUE   <= hue <  UPPER_ORANGE_HUE: return 5 + offset  # Orange
    elif UPPER_ORANGE_HUE  <= hue <  UPPER_YELLOW_HUE: return 1 + offset  # Yellow
    elif UPPER_YELLOW_HUE  <= hue <  UPPER_GREEN_HUE:  return 6 + offset  # Green
    elif UPPER_GREEN_HUE   <= hue <  UPPER_BLUE_HUE:   return 2 + offset  # Blue
    elif UPPER_BLUE_HUE    <= hue <= UPPER_PURPLE_HUE: return 4 + offset  # Purple
    elif UPPER_PURPLE_HUE  <= hue <= LOWER_RED_HUE:    return 3 + offset  # Red (wraps around)
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
    dominant_hue, histogram = get_dominant_hue(hsv_img, color_mask)
    number = hue_to_ball_number(dominant_hue, is_striped)

    #debug_dominant_color(ball_roi, hsv_img, color_mask, dominant_hue) # TODO: Remove (DEBUG)

    return is_striped, number, dominant_hue, histogram



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



def get_hue_confidence(ball):
    """Calculate confidence score (0-1) for hue-based ball detection.
    
    High confidence: hue is far from range boundaries (centered in its range)
    Low confidence: hue is near boundaries (ambiguous/could be another ball)
    """
    number = ball.number 
    dominant_hue = ball.dominant_hue
    
    # Invalid cases: no number detected or bad hue value
    if number is None or number < 0 or dominant_hue is None or dominant_hue < 0:
        return 0.0
    
    # Cue ball (white): always high confidence if detected as 0
    if number == 0:
        return 1.0
    
    # For numbered balls, find the hue range and calculate distance from boundaries
    # Distance from closest boundary = confidence (normalized to 0-1)
    
    offset = 8 if number > 8 else 0  # striped balls have offset
    color_number = number - offset   # get base color number (1-8)
    
    # Map colors to their hue ranges (based on hue_to_ball_number logic)
    hue_ranges = {
        1: (UPPER_ORANGE_HUE, UPPER_YELLOW_HUE),    # Yellow
        2: (UPPER_GREEN_HUE, UPPER_BLUE_HUE),       # Blue
        3: (UPPER_RED_HUE, UPPER_BROWN_HUE),        # Red (or LOWER_RED_HUE to 180)
        4: (UPPER_BLUE_HUE, UPPER_PURPLE_HUE),      # Purple
        5: (UPPER_BROWN_HUE, UPPER_ORANGE_HUE),     # Orange
        6: (UPPER_YELLOW_HUE, UPPER_GREEN_HUE),     # Green
        7: (UPPER_RED_HUE, UPPER_BROWN_HUE),        # Brown
    }
    
    if color_number not in hue_ranges:
        return 0.5  # Unknown color, medium confidence
    
    lower, upper = hue_ranges[color_number]
    
    # Special case: red wraps around (0-10 and 165-179)
    if color_number == 3:
        # Check if hue is closer to lower or upper part of red range
        dist_to_lower = dominant_hue
        dist_to_upper = LOWER_RED_HUE - dominant_hue
        min_dist = min(dist_to_lower, dist_to_upper)
        range_width = (180 - LOWER_RED_HUE) + UPPER_RED_HUE
    else:
        min_dist = min(dominant_hue - lower, upper - dominant_hue)
        range_width = upper - lower
    
    # Confidence: distance from nearest boundary, normalized by range width
    # Max when centered, decreases toward boundaries
    if range_width <= 0:
        return 0.5
    
    confidence = max(0.0, 1.0 - (abs(min_dist) / (range_width / 2.0)))
    return confidence



def get_second_best_number(ball):
    number, dominant_hue, hist = ball.number, ball.dominant_hue, ball.histogram
    
    # No histogram data, cannot determine alternative number
    if hist is None or len(hist) == 0:
        return None
    
    # Get hues sorted by frequency (descending)
    sorted_hue_indices = np.argsort(hist)[::-1]  # Reverse to get descending order
    
    # Skip the first one (dominant hue) and try subsequent peaks
    for i in range(1, len(sorted_hue_indices)):
        candidate_hue = sorted_hue_indices[i]
        candidate_number = hue_to_ball_number(candidate_hue, is_striped=(number > 8))
        
        # Return the first alternative that maps to a different number
        if candidate_number != number and candidate_number is not None and candidate_number >= 0:
            return candidate_number
    
    # No alternative number found
    return None



def handle_duplicate_balls(balls):
    final_balls = []

    # 1. Find duplicate numbers
    seen_numbers = set()
    duplicate_numbers = set()
    for ball in balls:
        number = ball.number
        if number in seen_numbers:
            duplicate_numbers.add(number)
        seen_numbers.add(number)

    # 2. Find all balls with duplicate numbers
    all_duplicate_balls = dict()
    duplicate_balls = []
    unique_balls = []
    for ball in balls:
        number = ball.number
        if number in duplicate_numbers:
            all_duplicate_balls[number] = all_duplicate_balls.get(number, []) + [ball] # e.g. {3: [ball1, ball2], 5: [ball3, ball4, ball5]}
            duplicate_balls.append(ball)
        else:
            unique_balls.append(ball)

    print("all_duplicate_balls:", all_duplicate_balls) # TODO: Remove (DEBUG)

    # 3. Compare duplicate balls and rank them by confidence
    for number, balls_with_same_number in all_duplicate_balls.items():
        if len(balls_with_same_number) == 1:
            final_balls.append(balls_with_same_number[0]) # No duplicates, keep as is
        else:
            # Sort by confidence (descending)
            sorted_balls = sorted(balls_with_same_number, key=get_hue_confidence, reverse=True)
            best_ball = sorted_balls[0]
            final_balls.append(best_ball) # Keep the most confident ball

            # For the others, we assign a second-best number
            for other_ball in sorted_balls[1:]:
                second_best_number = get_second_best_number(other_ball)
                other_ball.number = second_best_number # Assign second-best number (or set to None)
                final_balls.append(other_ball)

    final_balls.extend(unique_balls) # Add unique balls that had no duplicates
    return final_balls


""" Main function to detect all balls numbers and types. Return their types and numbers """
def detect_ball_numbers(image, circles):
    balls = []
    if circles is None:
        return balls

    circles = np.asarray(circles)

    # For each detected circle, crop the ROI, detect color and type, and store results in a Ball object
    for (x, y, r) in circles:
        ball_roi, bbox = crop_ball_roi(image, x, y, r)
        if ball_roi.size == 0:
            continue
        ball_type, number, dominant_hue, histogram = detect_ball_color(ball_roi)
        balls.append(Ball((int(x), int(y), int(r)), bbox, ball_type, number, dominant_hue, histogram=histogram))
        #balls.append(((int(x), int(y), int(r)), bbox, ball_type, number, dominant_hue)) # TODO: Remove (OLD)

    balls = handle_duplicate_balls(balls) # FIXME: Currently not working well

    return balls




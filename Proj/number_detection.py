import numpy as np
import cv2
from debug import debug_dominant_color, show_detected_balls_on_image, save_debug_image

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
    def __init__(self, circle_params, bbox, ball_type, number, dominant_color=None, xmin=None, xmax=None, ymin=None, ymax=None):
        self.circle_params = circle_params    # (x, y, r)
        self.bbox = bbox                      # (x1, y1, x2, y2)
        self.ball_type = ball_type            # solid or striped
        self.number = number                  # detected ball number
        self.dominant_color = dominant_color  # Detected dominant color in HSV (for debugging)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def to_dict(self):
        """Convert Ball object to a dictionary suitable for JSON serialization."""
        return {
            "number": self.number,
            "xmin" : self.xmin,
            "xmax" : self.xmax,
            "ymin" : self.ymin,
            "ymax" : self.ymax
        }



"""Calculate dominant color using median in BGR space (avoids hue wrapping issues)"""
def get_dominant_color(hsv_img, mask):
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    # Extract masked pixels and calculate median for each channel
    masked_pixels = bgr_img[mask > 0]
    if len(masked_pixels) == 0:
        return None
    median_bgr = np.median(masked_pixels, axis=0)[:3]
    median_bgr = np.uint8([[median_bgr]])
    median_hsv = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2HSV)
    return median_hsv


"""Calculate dominant color using only non-white pixels (for balls misclassified as white)."""
def get_dominant_color_non_white(hsv_img, mask):
    # Mask out white/near-white pixels (low saturation, high value)
    non_white = (hsv_img[:, :, 1] > 50) | (hsv_img[:, :, 2] < 150)
    combined_mask = cv2.bitwise_and(mask, mask, mask=non_white.astype(np.uint8) * 255)
    
    # Fall back to original mask if too few non-white pixels
    if np.sum(combined_mask > 0) < 20:
        return get_dominant_color(hsv_img, mask)
    
    return get_dominant_color(hsv_img, combined_mask)


"""Compute a white score for a ball ROI (higher = whiter). Used to find the cue ball."""
def compute_white_score(ball_roi):
    hsv_img = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)
    mask = get_circular_mask(hsv_img, 0.30)
    masked_s = hsv_img[:, :, 1][mask > 0]
    masked_v = hsv_img[:, :, 2][mask > 0]
    if len(masked_s) == 0:
        return -1
    avg_s = np.mean(masked_s)
    avg_v = np.mean(masked_v)
    # White = high brightness with low saturation
    return avg_v - avg_s



""" This function checks for the presence of a white stripe across the ball"""
def is_striped_ball(hsv_img):
    # Stripes = significant white band across the middle of the ball.
    h, w = hsv_img.shape[:2]
    
    # Check the middle horizontal band of the ball for white pixels
    # Stripes typically run horizontally across the middle
    band_start = int(h * 0.35)
    band_end = int(h * 0.65)
    middle_band = hsv_img[band_start:band_end, :]
    
    # Look for white pixels in the band (very low saturation)
    white_px = (middle_band[:, :, 1] < 40) & (middle_band[:, :, 2] > 100)
    white_ratio = np.sum(white_px) / (middle_band.shape[0] * middle_band.shape[1])
    
    is_striped = white_ratio > 0.05  # If >5% of middle band is white-ish
    
    # DEBUG # TODO: Remove
    #print(f"  Stripe check - white_ratio in middle band: {white_ratio:.3f}, is_striped: {is_striped}") 
    
    return is_striped



""" Create a circular mask for the given HSV image """
def get_circular_mask(hsv, radius_fraction=0.45):
    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(w, h) * radius_fraction), 255, -1)
    return mask



""" Calibrate white balance using the cue ball (white ball) as reference. Assumes the brightest, least saturated ball is the cue ball. """
def calibrate_white_balance(bgr_image, circles, pad=4):
    if circles is None or len(circles) == 0:
        return bgr_image  # No circles, return original
        
    # Find the whitest ball (lowest saturation + highest brightness)
    white_ball_candidate = None
    best_white_score = -1
    
    for (x, y, r) in circles:
        roi, _ = crop_ball_roi(bgr_image, x, y, r, pad=pad)
        if roi.size == 0:
            continue
        
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = get_circular_mask(roi_hsv, 0.4)
        
        # Score: high brightness, low saturation
        avg_s = cv2.mean(roi_hsv[:, :, 1], mask=mask)[0]
        avg_v = cv2.mean(roi_hsv[:, :, 2], mask=mask)[0]
        
        white_score = avg_v - avg_s  # Higher is whiter
        
        if white_score > best_white_score:
            best_white_score = white_score
            white_ball_candidate = roi
    
    if white_ball_candidate is None:
        return bgr_image
    
    # Calculate white balance correction using the cue ball
    gray_white = cv2.cvtColor(white_ball_candidate, cv2.COLOR_BGR2GRAY)
    target_gray = 255  # Ideal white
    current_mean = np.mean(gray_white)
    
    if current_mean < 10:  # Avoid division by near-zero
        return bgr_image
    
    # Apply correction to entire image
    scale = target_gray / current_mean
    calibrated = cv2.convertScaleAbs(bgr_image.astype(np.float32) * scale)
    calibrated = np.uint8(np.clip(calibrated, 0, 255))
    
    return calibrated



'''Map dominant HSV color to ball number'''
def color_to_ball_number(dominant_color, is_striped):
    if dominant_color is None:
        return None
    
    h, s, v = dominant_color[0][0]  # Extract HSV values
    
    # Map color ranges to ball numbers   
    color_match = None

    # White/Cue ball detection (very low saturation, high brightness)
    if s < 50 and v > 180:
        color_match = "white"
    # Black (check before colors to avoid false positives)
    elif v < 120: # Black key indicator: LOW brightness (V < 120), saturation can vary after calibration
        color_match = "black"
    # Red (wraps around hue range)
    elif (h <= UPPER_RED_HUE or h >= LOWER_RED_HUE) and s > 40:
        color_match = "red"
    # Brown
    elif UPPER_RED_HUE < h <= UPPER_BROWN_HUE and s > 30:
        color_match = "brown"
    # Orange
    elif UPPER_BROWN_HUE < h <= UPPER_ORANGE_HUE and s > 40:
        color_match = "orange"
    # Yellow
    elif UPPER_ORANGE_HUE < h <= UPPER_YELLOW_HUE and s > 40:
        color_match = "yellow"
    # Green
    elif UPPER_YELLOW_HUE < h <= UPPER_GREEN_HUE and s > 40:
        color_match = "green"
    # Blue
    elif UPPER_GREEN_HUE < h <= UPPER_BLUE_HUE and s > 40:
        color_match = "blue"
    # Purple
    elif UPPER_BLUE_HUE < h <= UPPER_PURPLE_HUE and s > 30:
        color_match = "purple"
    
    # Map color to ball number
    color_to_number = {
        "yellow": 1,
        "blue":   2,
        "red":    3,
        "purple": 4,
        "orange": 5,
        "green":  6,
        "brown":  7,
        "black":  8,
        "white":  0  # Cue ball
    }
    
    if color_match not in color_to_number:
        return None
    
    base_number = color_to_number[color_match]
    
    # Striped balls are numbered base number + 8
    if is_striped and 1 <= base_number <= 7:
        return base_number + 8
    
    return base_number



""" Detect single ball color and type, and map to ball number """
def detect_ball_color(ball_roi, use_non_white=False):
    hsv_img = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)

    # Stripe detection uses the unfiltered mask
    is_striped = is_striped_ball(hsv_img)

    # Detect dominant color and number
    circular_mask = get_circular_mask(hsv_img, 0.30)
    if use_non_white:
        dominant_color = get_dominant_color_non_white(hsv_img, circular_mask)
    else:
        dominant_color = get_dominant_color(hsv_img, circular_mask)
    number = color_to_ball_number(dominant_color, is_striped)

    # TODO: DEBUG (Remove later)
    #h, s, v = dominant_color[0][0]
    #print(f"Ball detected - H:{h}, S:{s}, V:{v} | Striped:{is_striped} | Number:{number}") 
    #if dominant_color is not None:
        #debug_dominant_color(ball_roi, hsv_img, circular_mask, dominant_color) 

    return is_striped, number, dominant_color



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
def detect_ball_numbers(image, circles, img_name, debug=False):
    # Apply white balance calibration using detected cue ball
    image = calibrate_white_balance(image, circles)
    
    h, w = image.shape[:2]
    
    # First pass: detect all balls and compute white scores
    ball_data = []  # (ball_roi, bbox, circle, xmin, xmax, ymin, ymax, white_score)
    for (x, y, r) in circles:
        ball_roi, bbox = crop_ball_roi(image, x, y, r)
        xmin = bbox[0]/w
        xmax = bbox[2]/w
        ymin = bbox[1]/h
        ymax = bbox[3]/h
        if ball_roi.size == 0:
            continue
        white_score = compute_white_score(ball_roi)
        ball_data.append((ball_roi, bbox, (int(x), int(y), int(r)), xmin, xmax, ymin, ymax, white_score))
    
    # Find the whitest ball (cue ball candidate)
    cue_ball_idx = -1
    if ball_data:
        cue_ball_idx = max(range(len(ball_data)), key=lambda i: ball_data[i][7])
    
    # Second pass: classify each ball
    balls = []
    for i, (ball_roi, bbox, circle, xmin, xmax, ymin, ymax, white_score) in enumerate(ball_data):
        if i == cue_ball_idx:
            # This is the whitest ball -> cue ball (number 0)
            ball_type = False  # not striped
            number = 0
            hsv_img = cv2.cvtColor(ball_roi, cv2.COLOR_BGR2HSV)
            circular_mask = get_circular_mask(hsv_img, 0.30)
            dominant_color = get_dominant_color(hsv_img, circular_mask)
            #if dominant_color is not None: # TODO: Remove (DEBUG)
                #debug_dominant_color(ball_roi, hsv_img, circular_mask, dominant_color)
        else:
            # For non-cue balls, detect normally first
            ball_type, number, dominant_color = detect_ball_color(ball_roi)
            # If this ball was also classified as white, reclassify using non-white pixels
            if number == 0:
                ball_type, number, dominant_color = detect_ball_color(ball_roi, use_non_white=True)
        
        balls.append(Ball(circle, bbox, ball_type, number, dominant_color, xmin, xmax, ymin, ymax))

    # Save output for debugging
    if debug:
        output_img = image.copy()
        for ball in balls:
            (x1, y1, x2, y2) = ball.bbox
            number = ball.number

            # Draw ROI bounding box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Place the detected number slightly below the ROI
            label = "?" if number is None else str(number)
            text_x = x1
            text_y = min(output_img.shape[0] - 5, y2 + 20)
            cv2.putText(output_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        save_debug_image(output_img, img_name, "debug/number_identification")

    return balls




import argparse
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt


# Input and output
INPUT_DIR = "example_json/input.json"
OUTPUT_DIR = "output" 

# Debug
DEBUG_OPTION = None
DEBUG_OUTPUT_DIR = "debug" 


####################################################
#######                                      #######
#######              WARP TABLE              #######
#######                                      #######
####################################################
LOWER_BLUE = np.array([90, 60, 60], dtype=np.uint8)
UPPER_BLUE = np.array([144, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 5000
HULL_SIMPLIFY_EPS = 2.0

def order_points_clockwise(pts):
    pts = np.asarray(pts, dtype=np.float32)

    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts.reshape(-1, 2)

    c = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(angles)]

    s = pts.sum(axis=1)
    start = np.argmin(s)
    pts = np.roll(pts, -start, axis=0)

    return pts.astype(np.float32)

def fit_lines_and_intersect(hull, image_shape):
    h, w = image_shape[:2]

    hull = cv2.approxPolyDP(hull, HULL_SIMPLIFY_EPS, True)

    points = hull.reshape(-1, 2).astype(np.float32)

    if len(points) < 8:
        return None

    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = order_points_clockwise(box)
    tl, tr, br, bl = box

    box_sides = [(tl, tr), (tr, br), (br, bl), (bl, tl)]

    side_models = []
    for a, b in box_sides:
        direction = b - a
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        side_models.append({"a": a, "b": b, "dir": direction / norm})

    side_segments = [[], [], [], []]

    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        seg_vec = p2 - p1
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 2.0:
            continue

        seg_dir = seg_vec / seg_len
        mid = 0.5 * (p1 + p2)

        best_side = -1
        best_score = float("inf")

        for side_idx, side in enumerate(side_models):
            align = abs(float(np.dot(seg_dir, side["dir"])))
            if align < 0.65:
                continue

            side_vec = side["b"] - side["a"]
            dist = abs(float(np.cross(
                np.array([side_vec[0], side_vec[1], 0.0], dtype=np.float32),
                np.array([mid[0] - side["a"][0], mid[1] - side["a"][1], 0.0], dtype=np.float32),
            )[2])) / (np.linalg.norm(side_vec) + 1e-6)

            score = dist + (1.0 - align) * 80.0
            if score < best_score:
                best_score = score
                best_side = side_idx

        if best_side >= 0:
            side_segments[best_side].append((seg_len, p1, p2))

    if any(len(group) == 0 for group in side_segments):
        return None

    def side_points(group, keep=6):
        chosen = sorted(group, key=lambda item: item[0], reverse=True)[:keep]
        pts = []
        for _, a, b in chosen:
            pts.append(a)
            pts.append(b)
        return np.array(pts, dtype=np.float32)

    side_point_sets = [side_points(group) for group in side_segments]
    if any(len(pts) < 2 for pts in side_point_sets):
        return None

    def fit_line(pts):
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        return (float(np.asarray(vx).item()), float(np.asarray(vy).item()),
                float(np.asarray(x0).item()), float(np.asarray(y0).item()))

    def intersect(line1, line2):
        vx1, vy1, x1, y1 = line1
        vx2, vy2, x2, y2 = line2
        denom = vx1 * vy2 - vy1 * vx2
        if abs(denom) < 1e-6:
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
        return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)

    side_lines = [fit_line(pts) for pts in side_point_sets]

    corner0 = intersect(side_lines[3], side_lines[0])
    corner1 = intersect(side_lines[0], side_lines[1])
    corner2 = intersect(side_lines[1], side_lines[2])
    corner3 = intersect(side_lines[2], side_lines[3])

    corners = [corner0, corner1, corner2, corner3]
    if any(corner is None for corner in corners):
        return None

    corners = order_points_clockwise(np.array(corners, dtype=np.float32))
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners.reshape(-1, 1, 2).astype(np.int32)

def approx_quad_from_hull(hull):
    perimeter = cv2.arcLength(hull, True)

    eps = 0.02
    best = None

    for _ in range(12):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)

        if len(approx) == 4:
            return approx

        best = approx
        eps *= 1.3

    if best is not None and len(best) == 4:
        return best

    return None

def quad_internal_angles_deg(quad):
    pts = order_points_clockwise(quad).astype(np.float32)
    if len(pts) != 4:
        return None

    angles = []
    for i in range(4):
        prev_pt = pts[(i - 1) % 4]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % 4]

        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return None

        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosang))))

    return np.array(angles, dtype=np.float32)

def quad_is_near_right_angles(quad, tolerance_deg=25.0):
    angles = quad_internal_angles_deg(quad)
    if angles is None:
        return False
    return bool(np.all(np.abs(angles - 90.0) <= tolerance_deg))

def select_table_quad(hull, image_shape, right_angle_tolerance_deg=25.0):
    quad = fit_lines_and_intersect(hull, image_shape)

    if quad is not None and quad_is_near_right_angles(quad, tolerance_deg=right_angle_tolerance_deg):
        return quad

    quad_approx = approx_quad_from_hull(hull)
    if quad_approx is not None:
        return quad_approx

    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)

def compute_topview_homography(quad):
    src = order_points_clockwise(quad)   # tl, tr, br, bl
    tl, tr, br, bl = src

    top_len    = float(np.linalg.norm(tr - tl))
    bottom_len = float(np.linalg.norm(br - bl))
    left_len   = float(np.linalg.norm(bl - tl))
    right_len  = float(np.linalg.norm(br - tr))

    h_ratio_base = max(top_len, bottom_len) / max(min(top_len, bottom_len), 1.0)
    v_ratio_base = max(left_len, right_len) / max(min(left_len, right_len), 1.0)

    AMPLIFIER = 1.8

    h_ratio = h_ratio_base ** AMPLIFIER
    v_ratio = v_ratio_base ** AMPLIFIER

    avg_h_adjusted = ((top_len + bottom_len) / 2.0) * v_ratio
    avg_v_adjusted = ((left_len + right_len) / 2.0) * h_ratio

    #print(avg_h_adjusted)
    #print(avg_v_adjusted)

    if avg_v_adjusted > avg_h_adjusted:
        src_reordered = np.array([bl, tl, tr, br], dtype=np.float32)
        short_side = int(round((top_len + bottom_len) / 2.0))
    else:
        src_reordered = src
        short_side = int(round((left_len + right_len) / 2.0))

    width  = short_side * 2
    height = short_side

    dst = np.array([
        [0,         0         ],
        [width - 1, 0         ],
        [width - 1, height - 1],
        [0,         height - 1],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_reordered, dst, method=0)
    return H, (width, height)

def compute_per_side_padding(quad, dst_size, padding_min=0.06, padding_max=0.1):
    src = order_points_clockwise(quad)
    tl, tr, br, bl = src

    ref = dst_size[1]

    top_len = np.linalg.norm(tr - tl)
    bottom_len = np.linalg.norm(br - bl)
    left_len = np.linalg.norm(bl - tl)
    right_len = np.linalg.norm(br - tr)

    def _pad(edge_len, opposite_len):
        ratio = edge_len / (opposite_len + 1e-6)
        t = float(np.clip((ratio - 1.0), 0.0, 1.0))
        frac = padding_max - t * (padding_max - padding_min)
        return int(round(ref * frac))

    pad_top = _pad(top_len, bottom_len)
    pad_bottom = _pad(bottom_len, top_len)
    pad_left = _pad(left_len, right_len)
    pad_right = _pad(right_len, left_len)

    return pad_top, pad_right, pad_bottom, pad_left


def adjust_homography_no_crop(H, src_image_shape):
    h, w = src_image_shape[:2]
    max_canvas_w = w * 4
    max_canvas_h = h * 4

    corners = np.array([
        [0,     0    ],
        [w - 1, 0    ],
        [w - 1, h - 1],
        [0,     h - 1],
    ], dtype=np.float32).reshape(-1, 1, 2)

    corners_dst = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # Clamp to avoid degenerate homographies blowing up the canvas
    corners_dst[:, 0] = np.clip(corners_dst[:, 0], -max_canvas_w, max_canvas_w)
    corners_dst[:, 1] = np.clip(corners_dst[:, 1], -max_canvas_h, max_canvas_h)

    x_min = corners_dst[:, 0].min()
    y_min = corners_dst[:, 1].min()
    x_max = corners_dst[:, 0].max()
    y_max = corners_dst[:, 1].max()

    tx = -x_min if x_min < 0 else 0.0
    ty = -y_min if y_min < 0 else 0.0

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1],
    ], dtype=np.float64)

    H_adjusted = T @ H

    canvas_w = min(int(np.ceil(x_max + tx)), max_canvas_w)
    canvas_h = min(int(np.ceil(y_max + ty)), max_canvas_h)

    return H_adjusted, (canvas_w, canvas_h)

def _to_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _make_debug_tile(img, label, tile_w=560, tile_h=360, title_h=42):
    bgr = _to_bgr(img)

    if bgr is None:
        resized_canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    else:
        h, w = bgr.shape[:2]

        scale = min(tile_w / w, tile_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        resized_canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        x0 = (tile_w - new_w) // 2
        y0 = (tile_h - new_h) // 2

        resized_canvas[y0:y0+new_h, x0:x0+new_w] = resized

    canvas = np.zeros((tile_h + title_h, tile_w, 3), dtype=np.uint8)

    canvas[title_h:title_h+tile_h, 0:tile_w] = resized_canvas

    cv2.rectangle(canvas, (0, 0), (tile_w, title_h), (0, 0, 0), thickness=cv2.FILLED)

    cv2.putText(
        canvas,
        label,
        (12, int(title_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return canvas

def _compose_debug_grid(tiles, columns=2):
    if not tiles:
        return np.zeros((360, 560, 3), dtype=np.uint8)

    tile_h, tile_w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / columns))
    total_slots = rows * columns

    if len(tiles) < total_slots:
        blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tiles = tiles + [blank] * (total_slots - len(tiles))

    row_imgs = []
    for row_idx in range(rows):
        start = row_idx * columns
        end = start + columns
        row_imgs.append(cv2.hconcat(tiles[start:end]))

    return cv2.vconcat(row_imgs)

def save_debug_pipeline(
    image, image_name, output_dir,
    blue_mask, blue_only, hull, quad, table_mask,
    topview,
):
    os.makedirs(output_dir, exist_ok=True)

    contour_stage = image.copy()
    cv2.drawContours(contour_stage, [hull], -1, (0, 255, 0), 2)
    cv2.drawContours(contour_stage, [quad], -1, (0, 0, 255), 3)

    tiles = [
        _make_debug_tile(image, "01 Original"),
        _make_debug_tile(blue_mask, "02 Blue Mask"),
        _make_debug_tile(blue_only, "03 Largest Blue"),
        _make_debug_tile(contour_stage, "04 Hull + Quad"),
        _make_debug_tile(table_mask, "05 Table Mask"),
        _make_debug_tile(topview, "06 Top-View"),
    ]

    debug_panel = _compose_debug_grid(tiles, columns=2)
    base_name = os.path.splitext(image_name)[0]
    debug_path = os.path.join(output_dir, f"{base_name}_debug_pipeline.jpg")
    cv2.imwrite(debug_path, debug_panel)
    print(f"[DEBUG] pipeline saved in: {debug_path}")

def warp_table(image, image_name, output_dir, debug=False):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_img, LOWER_BLUE, UPPER_BLUE)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"Blue area not found {image_name}")
        return None

    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    if not contours:
        return None

    blue_cnt = max(contours, key=cv2.contourArea)
    blue_only = np.zeros_like(blue_mask)
    cv2.drawContours(blue_only, [blue_cnt], -1, 255, thickness=cv2.FILLED)
    hull = cv2.convexHull(blue_cnt)
    quad = select_table_quad(hull, image.shape, right_angle_tolerance_deg=25.0)
    table_mask = np.zeros_like(blue_only)
    cv2.drawContours(table_mask, [quad], -1, 255, thickness=cv2.FILLED)
    H, dst_size = compute_topview_homography(quad)
    pad_top, pad_right, pad_bottom, pad_left = compute_per_side_padding(quad, dst_size)

    padded_w = dst_size[0] + pad_left + pad_right
    padded_h = dst_size[1] + pad_top + pad_bottom

    T_pad = np.array([
        [1, 0, pad_left],
        [0, 1, pad_top],
        [0, 0, 1],
    ], dtype=np.float64)

    H_padded = T_pad @ H

    topview = cv2.warpPerspective(image, H_padded, (padded_w, padded_h))

    if output_dir is not None:
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        save_path = os.path.join(output_dir, f"{base_name}.jpg")
        cv2.imwrite(save_path, topview)

    if debug:
        save_debug_pipeline(
            image=image,
            image_name=image_name,
            output_dir="debug/table_mask",
            blue_mask=blue_mask,
            blue_only=blue_only,
            hull=hull,
            quad=quad,
            table_mask=table_mask,
            topview=topview,
        )

    return (table_mask, quad)




####################################################
#######                                      #######
#######            BALL DETECTION            #######
#######                                      #######
####################################################
def save_debug_image(image, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]
    debug_path = os.path.join(output_dir, f"{base_name}_debug.png")

    cv2.imwrite(debug_path, image)
    print(f"[DEBUG] image saved in: {debug_path}")



def get_table_width_height(image, image_name, output_dir):
    # Create the table mask
    table_mask, quad = warp_table(image, image_name, output_dir, debug=False)

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



""" Calculate Intersection over Union (IoU) between two bounding boxes."""
def is_top_view(quad, tolerance=0.15):
    src = order_points_clockwise(quad)
    (tl, tr, br, bl) = src

    # Compute the lengths of all four sides
    top_len = float(np.linalg.norm(tr - tl))
    bottom_len = float(np.linalg.norm(br - bl))
    left_len = float(np.linalg.norm(bl - tl))
    right_len = float(np.linalg.norm(br - tr))

    # Compute ratio of opposite sides (1.0 = perfectly rectangular). In a top view, approximately, top = bottom and left = right
    h_ratio = min(top_len, bottom_len) / max(top_len, bottom_len, 1e-6)
    v_ratio = min(left_len, right_len) / max(left_len, right_len, 1e-6)

    is_top = h_ratio > (1.0 - tolerance) and v_ratio > (1.0 - tolerance)

    return is_top


def detect_balls(image, image_name, output_dir, debugOutput, hueDebug=False):
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

    table_mask, quad = warp_table(image, image_name, output_dir, debug=False)
    ima = cv2.bitwise_and(image, image, mask=table_mask)

    # Check if this is a top-down view to select appropriate parameters
    top_view = is_top_view(quad)

    # Compute the 6 pocket locations to filter out balls inside holes for top view
    src = order_points_clockwise(quad)
    (tl, tr, br, bl) = src
    top_len = np.linalg.norm(tr - tl) # Calculate the length of the top edge of the table (distance between top-left and top-right corners)
    bottom_len = np.linalg.norm(br - bl)
    left_len = np.linalg.norm(bl - tl)
    right_len = np.linalg.norm(br - tr)
    
    pockets = [tl, tr, br, bl]
    if (top_len + bottom_len) > (left_len + right_len): # Check if the table is in landscape orientation (width > height)
        pockets.append((tl + tr) / 2.0)
        pockets.append((bl + br) / 2.0)
    else:
        pockets.append((tl + bl) / 2.0)
        pockets.append((tr + br) / 2.0)

    # Before applying the HoughCircles we need to convert the image to grayscale
    # otherwise it will fail. We can also blur the image to remove some of the noise that may interfere.
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    # Detect circles using Hough Transform
    ballRadius = (width*POOL_TABLE_BALL_RATIO)/2.0
    offset = 21

    # Slightly less than a full diameter (2 * radius)
    minDist = ballRadius * 1.825
    minRadius = int(ballRadius)
    maxRadius = int(ballRadius + offset)

    if top_view:
        # we need more sensitive detection parameters in the top view
        param1 = 25
        param2 = 12
    else:
        param1 = 35
        param2 = 16

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,  minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # Draw the detected balls on the original angled image
    output_image = image.copy()
    resultingCircles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        #near_black_value_threshold = 58
        near_black_value_threshold = 62
        for circle_idx, i in enumerate(circles[0, :], start=1):
            x, y, r = i[0], i[1], i[2]

            # Reject balls that are inside or too close to holes (pockets) when in top-view
            if top_view:
                in_hole = False
                for pocket in pockets:
                    dist = np.linalg.norm(np.array([x, y]) - pocket)
                    if dist < r * 2.5:
                        in_hole = True
                        break
                if in_hole:
                    continue

            # Crop the image to just the detected circle.
            (hsv_image_cropped, _) = crop_ball_roi(hsv_image, x, y, 10, pad=0)
            hue_values = hsv_image_cropped[:, :, 0]

            # Find the dominant hue
            hist, bins = np.histogram(hue_values, bins=180, range=(0, 180))
            dominant_hue = np.argmax(hist)
            value_values = hsv_image_cropped[:, :, 2]
            mean_value = float(np.mean(value_values))
            is_near_black = mean_value <= near_black_value_threshold

            if hueDebug:
                base_name = os.path.splitext(image_name)[0]
                hue_debug_path = os.path.join(output_dir, f"{base_name}_hue_image_cropped_{circle_idx:02d}.png")
                cv2.imwrite(hue_debug_path, cv2.cvtColor(hsv_image_cropped, cv2.COLOR_HSV2BGR))
                print(f"[DEBUG] hue_image_cropped saved in: {hue_debug_path} and dominant hue is {dominant_hue}, mean V is {mean_value:.1f}, near_black={is_near_black}")

            if (dominant_hue > 85 and dominant_hue < 106) and (not is_near_black):
                #print(f"Color is {dominant_hue}")
                continue

            resultingCircles.append([x,y,r])
        
            # Draw the outer circle
            cv2.circle(output_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(output_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Save output for debugging
    if debugOutput:
        save_debug_image(output_image, image_name, "debug/ball_detection")
    
    return resultingCircles





####################################################
#######                                      #######
#######        NUMBER IDENTIFICATION         #######
#######                                      #######
####################################################
# Color hue ranges
LOWER_RED_HUE = 170
UPPER_RED_HUE = 8
UPPER_BROWN_HUE = 20
UPPER_ORANGE_HUE = 29
UPPER_YELLOW_HUE = 55
UPPER_GREEN_HUE = 85
UPPER_BLUE_HUE = 115
UPPER_PURPLE_HUE = 179


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


def debug_dominant_color(ball_roi, hsv_img, mask, dominant_color):
    # Display original ROI, mask, hue histogram, and a swatch of the detected dominant color.
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
    h, s, v = dominant_color[0][0]
    plt.axvline(h, color='red', linestyle='--', label=f'Dominant Hue: {h}')
    plt.title("Hue Histogram of Masked Region")
    plt.xlabel("Hue Value")
    plt.ylabel("Pixel Count")
    plt.legend()

    plt.subplot(1, 4, 4)
    if dominant_color is None:
        swatch_rgb = np.full((120, 120, 3), 128, dtype=np.uint8)
        swatch_title = "Dominant Color: N/A"
    else:
        h, s, v = dominant_color[0][0]
        swatch_rgb = cv2.cvtColor(dominant_color, cv2.COLOR_HSV2RGB)[0][0]
        swatch_rgb = np.full((120, 120, 3), swatch_rgb, dtype=np.uint8)
        swatch_title = f"Dominant Color (H={h}, S={s}, V={v})"
    plt.imshow(swatch_rgb)
    plt.title(swatch_title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

""" Visualize the detected balls by drawing bounding boxes and placing the detected number below each ball ROI """
def show_detected_balls_on_image(image, balls):
    display_img = image.copy()

    for ball in balls:
        (x1, y1, x2, y2) = ball.bbox
        number = ball.number

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



""" This function checks for the presence of a white stripe. Striped balls have a significant portion of white over their entire visible surface. """
def is_striped_ball(hsv_img):    
    # Get a circular mask covering most of the ball
    mask = get_circular_mask(hsv_img, radius_fraction=0.40)
    
    area = np.sum(mask > 0)
    if area == 0:
        return False
        
    # Look for white pixels within the circular mask (low saturation, high brightness)
    white_px = (hsv_img[:, :, 1] < 50) & (hsv_img[:, :, 2] > 100) & (mask > 0)
    
    white_ratio = np.sum(white_px) / area
    
    # Since we are evaluating the whole ball, a striped ball should have more than a given threshold of white pixels.
    is_striped = white_ratio > 0.15 
        
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
def detect_ball_color(ball_roi, use_non_white=False, debug=False):
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

    if dominant_color is not None and debug: 
        debug_dominant_color(ball_roi, hsv_img, circular_mask, dominant_color) 

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
def detect_ball_numbers(image, circles, img_name, debug=False, debug_output=False):
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
            if dominant_color is not None and debug: 
                debug_dominant_color(ball_roi, hsv_img, circular_mask, dominant_color)
        else:
            # For non-cue balls, detect normally first
            ball_type, number, dominant_color = detect_ball_color(ball_roi, debug=debug)
            # If this ball was also classified as white, reclassify using non-white pixels
            if number == 0:
                ball_type, number, dominant_color = detect_ball_color(ball_roi, use_non_white=True, debug=debug)

        balls.append(Ball(circle, bbox, ball_type, number, dominant_color, xmin, xmax, ymin, ymax))

    # Save output for debugging
    if debug_output:
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





####################################################
#######                                      #######
#######           MAIN DETECTION             #######
#######                                      #######
####################################################


''' Reads the image paths from the input json file '''
def read_from_json(input_dir):
    with open(input_dir) as json_data:
        return json.load(json_data)["image_path"]
 

''' Writes the output data to a json file in the output directory '''
def write_to_json(output_path, contents):
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Create the output directory if it doesn't exist
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(contents, file, ensure_ascii=False, indent=4)


''' Reads the image from the given path and returns the image and its name '''
def read_image(images, image_index):
    full_path = images[image_index]
    img_pool_table = cv2.imread(full_path)
    image_name = os.path.basename(full_path)
    return img_pool_table, image_name


''' Entry point for the full detection pipeline
    - Reads the images from the paths in the input json file  
    - Detects the balls and their numbers
    - Warps the table to a top-down view
    - Saves the results to the output directory (in json and images)
'''
def run_full_detection(json_input_dir, output_dir):
    # 1. Read the image paths array from the json file
    image_paths = read_from_json(json_input_dir)

    # 2. Loop through each image path and run the detection pipeline
    i = 0
    output_data_list = []
    for image_path in image_paths:
        print(f"[INFO] Processing image {i+1}/{len(image_paths)}: {image_path}")

        # 2.1 Read the image
        table_image, img_name = read_image(image_paths, i)

        # 2.2 Detect balls
        circles = detect_balls(table_image, img_name, output_dir=None, debugOutput=False) 

        # 2.3 Detect ball numbers
        balls = detect_ball_numbers(table_image, circles, img_name) 

        # 2.4 Warp the table and save the warped image
        warp_table(table_image, img_name, output_dir=output_dir+"/top_view", debug=False) 

        # 2.5. Create JSON object with output data
        output_data = {
            "image_path": image_path,
            "num_balls": len(balls),
            "balls": [ball.to_dict() for ball in balls]
        }
        output_data_list.append(output_data)

        i += 1

    # 2.6 Save the results to the output directory (in json)
    output_json_path = os.path.join(output_dir, "detection_results.json")
    write_to_json(output_json_path, output_data_list)


''' Main function to parse command-line arguments and run the appropriate pipeline mode (full or debug) based on the provided options.'''
def main():
    parser = argparse.ArgumentParser(description="Pool table ball detection pipeline with debug options", formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
                                        Examples:
                                          # Run full pipeline (default)
                                          python pipeline.py

                                          # Run debug for all images
                                          python pipeline.py --debug all

                                          # Run debug for specific image (index 2)
                                          python pipeline.py --debug 2

                                          # Specify custom input/output directories
                                          python pipeline.py --input custom_input.json --output custom_output

                                          # Full debug with custom paths
                                          python pipeline.py --debug all --input data.json --output results --debug-dir debug_output""")
    
    parser.add_argument("--debug", "-d", type=str, default=None, metavar="OPTION", help="Enable debug mode. Pass 'all' for all images or an image index (e.g., '0', '1'). Default: None (run full pipeline)")
    parser.add_argument("--input", "-i", type=str, default=INPUT_DIR, metavar="PATH", help=f"Input JSON file path (default: {INPUT_DIR})")
    parser.add_argument("--output", "-o", type=str, default=OUTPUT_DIR, metavar="PATH", help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--debug-dir", type=str, default=DEBUG_OUTPUT_DIR, metavar="PATH", help=f"Debug output directory (default: {DEBUG_OUTPUT_DIR})")
    args = parser.parse_args()
    
    # Determine debug mode
    debug_mode = args.debug is not None
    debug_option = None
    if debug_mode:
        if args.debug.lower() == "all":
            debug_option = None  # None means debug all images
            print(f"[DEBUG] Running debug pipeline for ALL images")
        else:
            try:
                debug_option = int(args.debug)
                print(f"[DEBUG] Running debug pipeline for image index: {debug_option}")
            except ValueError:
                parser.error(f"Invalid debug option '{args.debug}'. Use 'all' or an integer index.")
    
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)
    
    try:
        if debug_mode:
            os.makedirs(args.debug_dir, exist_ok=True)
            # Debug mode
            images = read_from_json(args.input)
            
            if debug_option is not None:
                # Debug specific image
                if debug_option < 0 or debug_option >= len(images):
                    print(f"Error: Image index {debug_option} out of range [0, {len(images)-1}]")
                    return
                
                table_image, img_name = read_image(images, debug_option)
                circles = detect_balls(table_image, img_name, output_dir=args.debug_dir+"/top_view", debugOutput=False)
                balls = detect_ball_numbers(table_image, circles, img_name, debug=True)
                show_detected_balls_on_image(table_image, balls)
            else:
                # Debug all images
                for i in range(len(images)):
                    table_image, img_name = read_image(images, i)
                    circles = detect_balls(table_image, img_name, output_dir=args.debug_dir+"/top_view", debugOutput=True)
                    balls = detect_ball_numbers(table_image, circles, img_name, debug=False, debug_output=True)
                    warp_table(table_image, img_name, output_dir=args.debug_dir+"/top_view", debug=True)
        else:
            # Normal pipeline execution
            print("[INFO] Running full detection pipeline...")
            run_full_detection(args.input, args.output)
            print("[INFO] Pipeline completed successfully!")
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
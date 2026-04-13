import numpy as np
import cv2
import os
from debug import visualize_table_mask_debug

DEBUG = True

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

    print(avg_h_adjusted)
    print(avg_v_adjusted)

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

def warp_table(image, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(
        hsv_img,
        LOWER_BLUE,
        UPPER_BLUE
    )

    contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print(f"Blue area not found {image_name}")
        return None

    contours = [
        c for c in contours
        if cv2.contourArea(c) > MIN_CONTOUR_AREA
    ]

    if not contours:
        return None

    blue_cnt = max(contours, key=cv2.contourArea)

    blue_only = np.zeros_like(blue_mask)

    cv2.drawContours(
        blue_only,
        [blue_cnt],
        -1,
        255,
        thickness=cv2.FILLED
    )

    hull = cv2.convexHull(blue_cnt)

    quad = select_table_quad(
        hull,
        image.shape,
        right_angle_tolerance_deg=25.0
    )

    table_mask = np.zeros_like(blue_only)

    cv2.drawContours(
        table_mask,
        [quad],
        -1,
        255,
        thickness=cv2.FILLED
    )

    H, dst_size = compute_topview_homography(quad)

    pad_top, pad_right, pad_bottom, pad_left = compute_per_side_padding(
        quad,
        dst_size
    )

    padded_w = dst_size[0] + pad_left + pad_right
    padded_h = dst_size[1] + pad_top + pad_bottom

    T_pad = np.array([
        [1, 0, pad_left],
        [0, 1, pad_top],
        [0, 0, 1],
    ], dtype=np.float64)

    H_padded = T_pad @ H

    topview = cv2.warpPerspective(
        image,
        H_padded,
        (padded_w, padded_h)
    )

    if DEBUG:
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

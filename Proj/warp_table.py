import numpy as np
import cv2
import os
from debug import visualize_table_mask_debug

def order_points_clockwise(pts):
    pts = np.asarray(pts, dtype=np.float32)

    # Handle OpenCV contour format (N, 1, 2) -> (N, 2)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts.reshape(-1, 2)

    # Compute centroid of all points
    c = np.mean(pts, axis=0)

    # Compute angle of each point relative to the centroid
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])

    # Sort points by angle (counterclockwise order)
    pts = pts[np.argsort(angles)]

    # Rotate order so that sequence starts at top-left point
    # (top-left is approximated as the point with smallest x + y)
    s = pts.sum(axis=1)
    start = np.argmin(s)
    pts = np.roll(pts, -start, axis=0)

    return pts.astype(np.float32)

def fit_lines_and_intersect(hull, image_shape):
    h, w = image_shape[:2]
    points = hull.reshape(-1, 2).astype(np.float32)
    if len(points) < 8:
        return None

    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = order_points_clockwise(box)
    tl, tr, br, bl = box

    box_sides = [
        (tl, tr),
        (tr, br),
        (br, bl),
        (bl, tl),
    ]

    side_models = []
    for a, b in box_sides:
        direction = b - a
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        side_models.append({
            "a": a,
            "b": b,
            "dir": direction / norm,
        })

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
            dist = abs(
                float(
                    np.cross(
                        np.array([side_vec[0], side_vec[1], 0.0], dtype=np.float32),
                        np.array([mid[0] - side["a"][0], mid[1] - side["a"][1], 0.0], dtype=np.float32),
                    )[2]
                )
            ) / (np.linalg.norm(side_vec) + 1e-6)

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
        return (
            float(np.asarray(vx).item()),
            float(np.asarray(vy).item()),
            float(np.asarray(x0).item()),
            float(np.asarray(y0).item()),
        )

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

    quad = None
    best = None
    best_diff = 10**9

    for eps_factor in np.linspace(0.005, 0.10, 120):
        approx = cv2.approxPolyDP(hull, eps_factor * perimeter, True)
        diff = abs(len(approx) - 4)

        if diff < best_diff:
            best = approx
            best_diff = diff

        if len(approx) == 4:
            quad = approx
            break

    if quad is None and best is not None and len(best) == 4:
        quad = best

    return quad

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

        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = float(np.clip(cosang, -1.0, 1.0))
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

def get_correct_orientation(warped_square_img):
    h, w = warped_square_img.shape[:2]
    gray = cv2.cvtColor(warped_square_img, cv2.COLOR_BGR2GRAY)

    margin = max(5, int(min(h, w) * 0.01))
    half_roi = max(7, int(min(h, w) * 0.015))
    depth = max(15, int(min(h, w) * 0.03))

    x0 = max(0, w // 2 - half_roi)
    x1 = min(w, w // 2 + half_roi)
    y0 = max(0, h // 2 - half_roi)
    y1 = min(h, h // 2 + half_roi)

    boxes = {
        "top": (x0, margin, x1, margin + depth),
        "bottom": (x0, h - margin - depth, x1, h - margin),
        "left": (margin, y0, margin + depth, y1),
        "right": (w - margin - depth, y0, w - margin, y1),
    }

    mid_top = gray[boxes["top"][1]:boxes["top"][3], boxes["top"][0]:boxes["top"][2]]
    mid_bottom = gray[boxes["bottom"][1]:boxes["bottom"][3], boxes["bottom"][0]:boxes["bottom"][2]]
    mid_left = gray[boxes["left"][1]:boxes["left"][3], boxes["left"][0]:boxes["left"][2]]
    mid_right = gray[boxes["right"][1]:boxes["right"][3], boxes["right"][0]:boxes["right"][2]]

    score_horizontal = float(np.mean(mid_top) + np.mean(mid_bottom))
    score_vertical = float(np.mean(mid_left) + np.mean(mid_right))

    orientation = "HORIZONTAL" if score_horizontal < score_vertical else "VERTICAL"
    return orientation, score_horizontal, score_vertical, boxes

def _to_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _make_debug_tile(img, label, tile_w=560, tile_h=360):
    bgr = _to_bgr(img)
    if bgr is None:
        bgr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tile = cv2.resize(bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile_w, 42), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(tile, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return tile

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
    image,
    image_name,
    output_dir,
    blue_mask,
    blue_only,
    hull,
    quad,
    table_mask,
    warped_probe,
    warped_final,
    orientation,
    score_horizontal,
    score_vertical,
    sample_boxes,
):
    os.makedirs(output_dir, exist_ok=True)

    contour_stage = image.copy()
    cv2.drawContours(contour_stage, [hull], -1, (0, 255, 0), 2)
    cv2.drawContours(contour_stage, [quad], -1, (0, 0, 255), 3)

    probe_stage = warped_probe.copy()
    for name, (x0, y0, x1, y1) in sample_boxes.items():
        color = (0, 255, 0) if name in ("top", "bottom") else (0, 165, 255)
        cv2.rectangle(probe_stage, (x0, y0), (x1, y1), color, 2)

    probe_text = f"ori={orientation} H={score_horizontal:.1f} V={score_vertical:.1f}"
    cv2.rectangle(probe_stage, (0, 0), (probe_stage.shape[1], 28), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(probe_stage, probe_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    tiles = [
        _make_debug_tile(image, "01 Original"),
        _make_debug_tile(blue_mask, "02 Blue Mask"),
        _make_debug_tile(blue_only, "03 Largest Blue"),
        _make_debug_tile(contour_stage, "04 Hull + approxPolyDP/Hull Lines intersection"),
        _make_debug_tile(table_mask, "05 Table Mask"),
        _make_debug_tile(probe_stage, "06 Warp Probe"),
        _make_debug_tile(warped_final, "07 Final Warp"),
    ]

    debug_panel = _compose_debug_grid(tiles, columns=2)
    base_name = os.path.splitext(image_name)[0]
    debug_path = os.path.join(output_dir, f"{base_name}_debug_pipeline.jpg")
    cv2.imwrite(debug_path, debug_panel)
    print(f"[DEBUG] pipeline saved in: {debug_path}")

def warp_table(image, image_name, output_dir, debugOutput = True):
    os.makedirs(output_dir, exist_ok=True)

    # Convert image to HSV and isolate blue table cloth
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 60, 60], dtype=np.uint8)
    upper_blue = np.array([144, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # Find largest blue contour (assumed table region)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Blue area not found {image_name}")
        return None

    blue_cnt = max(contours, key=cv2.contourArea)
    blue_only = np.zeros_like(blue_mask)
    cv2.drawContours(blue_only, [blue_cnt], -1, 255, thickness=cv2.FILLED)

    # Build convex hull around detected table and estimate a 4-corner polygon
    hull = cv2.convexHull(blue_cnt)
    quad = select_table_quad(hull, image.shape, right_angle_tolerance_deg=25.0)

    # Create final table mask from detected quadrilateral
    table_mask = np.zeros_like(blue_only)
    cv2.drawContours(table_mask, [quad], -1, 255, thickness=cv2.FILLED)

    # Save debug visualization of the table mask
    visualize_table_mask_debug(
        image=image,
        image_name=image_name,
        output_dir="debug/table_mask",
        table_mask=table_mask,
    )

    return (table_mask, quad)

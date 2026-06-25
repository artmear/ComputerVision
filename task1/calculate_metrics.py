import json
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

def calculate_iou(box1, box2):
    """Calculate Intersection Over Union between two bounding boxes.
    Boxes are in format [xmin, ymin, xmax, ymax] (normalized coordinates)."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def match_predictions_to_groundtruth(predictions, ground_truth, iou_threshold=0.5):
    """Match predicted bboxes to ground truth using greedy IOU matching.
    Returns matched pairs and unmatched predictions/ground truth."""
    matched_pairs = []
    used_gt = set()
    used_pred = set()
    
    # Calculate IOU for all pairs and sort by IOU descending
    iou_scores = []
    for i, pred in enumerate(predictions):
        pred_box = [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']]
        for j, gt in enumerate(ground_truth):
            gt_box = [gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax']]
            iou = calculate_iou(pred_box, gt_box)
            if iou > 0:
                iou_scores.append((iou, i, j))
    
    # Greedy matching: sort by IOU and assign highest first
    iou_scores.sort(reverse=True)
    for iou, pred_idx, gt_idx in iou_scores:
        if iou < iou_threshold:
            break
        if pred_idx not in used_pred and gt_idx not in used_gt:
            matched_pairs.append((pred_idx, gt_idx, iou))
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
    
    unmatched_pred = [i for i in range(len(predictions)) if i not in used_pred]
    unmatched_gt = [j for j in range(len(ground_truth)) if j not in used_gt]
    
    return matched_pairs, unmatched_pred, unmatched_gt

def extract_image_number(image_path):
    """Extract the base number from image filename (e.g., '100' from '100_png.rf.xyz.jpg')."""
    filename = Path(image_path).name
    # Extract leading hex or decimal number
    match = re.match(r'^([0-9a-f]+)', filename.lower())
    return match.group(1) if match else None

def calculate_metrics(ground_truth_path, predictions_path):
    """Calculate all metrics comparing predictions to ground truth."""
    
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
    
    with open(predictions_path, 'r') as f:
        predictions_data = json.load(f)
    
    # Create mappings by image number for matching
    pred_by_number = {}
    for item in predictions_data:
        num = extract_image_number(item['image_path'])
        if num:
            pred_by_number[num] = item
    
    # Metrics accumulators
    count_errors = []
    count_correct = 0
    total_images = 0
    ious = []
    undetected_ratios = []
    correct_class_ratios = []
    incorrect_class_ratios = []
    
    for gt_image in ground_truth_data:
        image_path = gt_image['image_path']
        print("Evaluating image:", image_path)
        print("Total images evaluated so far:", total_images)
        gt_balls = gt_image['balls']
        gt_count = gt_image['num_balls']
        
        # Find matching prediction by image number
        gt_number = extract_image_number(image_path)
        pred_image = pred_by_number.get(gt_number)
        
        if pred_image is None:
            print(f"No prediction found for image: {image_path} (number: {gt_number})")
            continue
        
        pred_balls = pred_image['balls']
        pred_count = pred_image['num_balls']
        
        total_images += 1
        
        # Metric 1 & 2: Ball Count MAE and Accuracy
        count_error = abs(gt_count - pred_count)
        count_errors.append(count_error)
        if count_error == 0:
            count_correct += 1
        
        # Metric 3: BBox Detection IOU (through matching)
        matched, unmatched_pred, unmatched_gt = match_predictions_to_groundtruth(
            pred_balls, gt_balls, iou_threshold=0.0
        )
        
        # Calculate IOU for matched detections
        for pred_idx, gt_idx, iou in matched:
            ious.append(iou)
        
        # Metric 4: Undetected balls (false negatives)
        num_undetected = len(unmatched_gt)
        undetected_ratio = num_undetected / max(gt_count, 1)
        undetected_ratios.append(undetected_ratio)
        
        # Metric 5 & 6: Classification accuracy for matched detections
        num_correctly_classified = 0
        num_incorrectly_classified = 0
        
        for pred_idx, gt_idx, iou in matched:
            pred_number = pred_balls[pred_idx]['number']
            gt_number = gt_balls[gt_idx]['number']
            
            if pred_number == gt_number:
                num_correctly_classified += 1
            else:
                num_incorrectly_classified += 1
        
        # Calculate ratios relative to total ground truth balls
        if gt_count > 0:
            correct_class_ratio = num_correctly_classified / gt_count
            incorrect_class_ratio = num_incorrectly_classified / gt_count
        else:
            correct_class_ratio = 0
            incorrect_class_ratio = 0
        
        correct_class_ratios.append(correct_class_ratio)
        incorrect_class_ratios.append(incorrect_class_ratio)
    
    # Calculate final metrics
    ball_count_mae = np.mean(count_errors) if count_errors else 0
    ball_count_accuracy = count_correct / max(total_images, 1)
    bbox_detection_iou = np.mean(ious) if ious else 0
    avg_undetected = np.mean(undetected_ratios) if undetected_ratios else 0
    avg_correctly_classified = np.mean(correct_class_ratios) if correct_class_ratios else 0
    avg_incorrectly_classified = np.mean(incorrect_class_ratios) if incorrect_class_ratios else 0
    
    return {
        'Ball Count MAE': ball_count_mae,
        'Ball Count Accuracy': ball_count_accuracy,
        'BBox Detection IOU': bbox_detection_iou,
        'Average % of undetected balls': avg_undetected,
        'Average % of correctly classified balls': avg_correctly_classified,
        'Average % of incorrectly classified balls': avg_incorrectly_classified,
        'Total Images': total_images,
        'Total Detections': len(ious)
    }

if __name__ == '__main__':
    # Paths
    ground_truth_path = 'roboflow_dataset/roboflow_balls.json'
    predictions_path = 'output/detection_results.json'
    
    metrics = calculate_metrics(ground_truth_path, predictions_path)
    
    print("=" * 50)
    print("DETECTION METRICS")
    print("=" * 50)
    print(f"Ball Count MAE:  {metrics['Ball Count MAE']:.4f}")
    print(f"Ball Count Accuracy:  {metrics['Ball Count Accuracy']:.4f}")
    print(f"BBox Detection IOU:  {metrics['BBox Detection IOU']:.4f}")
    print(f"Average % of undetected balls: {metrics['Average % of undetected balls']:.4f}")
    print(f"Average % of correctly classified balls:  {metrics['Average % of correctly classified balls']:.4f}")
    print(f"Average % of incorrectly classified balls:  {metrics['Average % of incorrectly classified balls']:.4f}")
    print("=" * 50)
    print(f"Total Images Evaluated: {metrics['Total Images']}")
    print(f"Total Matched Detections: {metrics['Total Detections']}")
    print("=" * 50)
    
    # Save results to JSON
    output_file = 'output/metrics_results.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {output_file}")

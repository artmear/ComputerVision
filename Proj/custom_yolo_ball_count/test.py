"""
Every metric consumes the standard torchvision detection format, so the same
functions work for ANY model (Faster R-CNN, SSD, RetinaNet, FCOS, ...):

    predictions   = [{"boxes": [N, 4] xyxy, "scores": [N], "labels": [N]}, ...]
    ground_truths = [{"boxes": [M, 4] xyxy, "labels": [M]}, ...]

Counting metrics measure the actual objective; detection metrics explain why.
"""

import os
import json

import torch
import torchvision
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_jpeg

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
DEBUG_DIR = "debug_out"

def save_debug_image(raw, pred, keep, image_path, debug_dir):
    boxes = pred["boxes"][keep].cpu()
    scores = pred["scores"][keep].cpu()
    labels = [f"{s:.2f}" for s in scores.tolist()]

    img = raw[:3].cpu()
    annotated = draw_bounding_boxes(img, boxes, labels=labels, colors="red", width=2)

    os.makedirs(debug_dir, exist_ok=True)
    out_path = os.path.join(debug_dir, os.path.basename(image_path))
    if not out_path.lower().endswith((".jpg", ".jpeg")):
        out_path += ".jpg"
    write_jpeg(annotated, out_path)
    return out_path

# ---------------------------------------------------------------------------
# Shared helpers (label 1 is "ball" and 0 is background)
# ---------------------------------------------------------------------------

def prediction_counts(predictions, conf_threshold):
    counts = [int(((prediction["scores"] >= conf_threshold) & (prediction["labels"] == 1)).sum()) for prediction in predictions]
    return torch.tensor(counts, dtype=torch.float32)


def ground_truth_counts(ground_truths):
    return torch.tensor([len(ground_truth["boxes"]) for ground_truth in ground_truths], dtype=torch.float32)


def _tp_fp_fn(predictions, ground_truths, conf_threshold, iou_threshold):
    """Greedy IoU matching (highest-confidence first) totalled over all images."""
    tp = fp = fn = 0
    for prediction, ground_truth in zip(predictions, ground_truths):
        keep = (prediction["scores"] >= conf_threshold) & (prediction["labels"] == 1)
        boxes = prediction["boxes"][keep][prediction["scores"][keep].argsort(descending=True)]
        gt = ground_truth["boxes"]

        if len(gt) == 0:
            fp += len(boxes)
            continue
        if len(boxes) == 0:
            fn += len(gt)
            continue

        ious = box_iou(boxes, gt)
        matched = torch.zeros(len(gt), dtype=torch.bool)
        for i in range(len(boxes)):
            row = ious[i].clone()
            row[matched] = 0
            best_iou, best_j = row.max(0)
            if best_iou >= iou_threshold:
                tp += 1
                matched[best_j] = True
            else:
                fp += 1
        fn += int((~matched).sum())
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Counting metrics (the real objective)
# ---------------------------------------------------------------------------

def mae(predictions, ground_truths, conf_threshold=CONF_THRESHOLD):
    err = prediction_counts(predictions, conf_threshold) - ground_truth_counts(ground_truths)
    return err.abs().mean().item()


def rmse(predictions, ground_truths, conf_threshold=CONF_THRESHOLD):
    err = prediction_counts(predictions, conf_threshold) - ground_truth_counts(ground_truths)
    return err.pow(2).mean().sqrt().item()


def exact_count_accuracy(predictions, ground_truths, conf_threshold=CONF_THRESHOLD):
    err = prediction_counts(predictions, conf_threshold) - ground_truth_counts(ground_truths)
    return (err == 0).float().mean().item()


def count_bias(predictions, ground_truths, conf_threshold=CONF_THRESHOLD):
    """Signed mean error: negative => undercounts, positive => overcounts."""
    err = prediction_counts(predictions, conf_threshold) - ground_truth_counts(ground_truths)
    return err.mean().item()


# ---------------------------------------------------------------------------
# Detection metrics (diagnostic: explain why a model counts well or badly)
# ---------------------------------------------------------------------------

def precision(predictions, ground_truths, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    tp, fp, _ = _tp_fp_fn(predictions, ground_truths, conf_threshold, iou_threshold)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(predictions, ground_truths, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    tp, _, fn = _tp_fp_fn(predictions, ground_truths, conf_threshold, iou_threshold)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1(predictions, ground_truths, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    tp, fp, fn = _tp_fp_fn(predictions, ground_truths, conf_threshold, iou_threshold)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def mean_average_precision(predictions, ground_truths):
    """mAP@0.5 and COCO-style mAP@0.5:0.95 (uses unfiltered predictions)."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(predictions, ground_truths)
    result = metric.compute()
    return {"mAP@0.5": result["map_50"].item(), "mAP@0.5:0.95": result["map"].item()}


# ---------------------------------------------------------------------------
# Inference runner + aggregation (works with any torchvision detector)
# ---------------------------------------------------------------------------

def run_inference(model, annotations, images_dir, device, transform):
    model.eval()
    predictions, ground_truths = [], []
    for ann in annotations:

        image_path = os.path.join(images_dir, ann["file_name"])
        raw_image = torchvision.io.decode_image(image_path)

        image = torchvision.tv_tensors.Image(raw_image)
        image = transform(image).to(device)
        with torch.no_grad():
            prediction = model([image])[0]

            if DEBUG_DIR is not None:
                keep = (prediction["scores"] >= CONF_THRESHOLD) & (prediction["labels"] == 1)
                save_debug_image(raw_image, prediction, keep, image_path, DEBUG_DIR)

        predictions.append({k: prediction[k].cpu() for k in ("boxes", "scores", "labels")})

        boxes = [(b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]) for b in ann["balls"]]
        ground_truths.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
        })

    return predictions, ground_truths


def evaluate(predictions, ground_truths):
    return {
        "MAE": mae(predictions, ground_truths),
        "RMSE": rmse(predictions, ground_truths),
        "ExactCount": exact_count_accuracy(predictions, ground_truths),
        "Bias": count_bias(predictions, ground_truths),
        "Precision": precision(predictions, ground_truths),
        "Recall": recall(predictions, ground_truths),
        "F1": f1(predictions, ground_truths),
        **mean_average_precision(predictions, ground_truths),
    }


def main():
    #from detection_faster_rcnn_resnet50 import load_model, inference_transform
    from detection_ssd import load_model, inference_transform

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    val_dir = "datasets_and_train_script/datasets/validation"
    with open(os.path.join(val_dir, "_annotations.json")) as f:
        annotations = json.load(f)

    #model = load_model("ball_detector_faster_rcnn_resnet50.pt", device)
    model = load_model("ball_detector_ssd.pt", device)
    predictions, ground_truths = run_inference(
        model, annotations, os.path.join(val_dir, "images"), device, inference_transform
    )

    for name, value in evaluate(predictions, ground_truths).items():
        print(f"{name:12s}: {value:.4f}")


if __name__ == "__main__":
    main()

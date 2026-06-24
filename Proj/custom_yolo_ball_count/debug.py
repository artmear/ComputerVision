import json
import os

import torch
import torchvision
from torchvision.transforms import v2 as transforms

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_jpeg


class BallDetector(torch.nn.Module):
    # Background + ball
    NUM_CLASSES = 2

    def __init__(self):
        super().__init__()
        # Faster R-CNN with a COCO-pretrained ResNet50-FPN backbone.
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)

        # Replace the classification/box head for our single "ball" class.
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.NUM_CLASSES)

    def forward(self, images, targets=None):
        # In training mode returns the loss dict; in eval mode returns detections.
        return self.model(images, targets)



INPUT_JSON = "input.json"
OUTPUT = "ball_counts.json"
WEIGHTS = "ball_detector.pt"
THRESHOLD = 0.5
DEBUG_DIR = "debug_out"

# Same preprocessing as the validation pipeline in train.py: Faster R-CNN
# normalizes/resizes internally, so we only convert to a float image.
inference_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])


def load_model(weights_path, device):
    model = BallDetector().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def count_balls(model, image_path, device, threshold, debug_dir=None):
    raw = torchvision.io.decode_image(image_path)
    image = torchvision.tv_tensors.Image(raw)
    image = inference_transform(image).to(device)

    with torch.no_grad():
        pred = model([image])[0]

    # Label 1 is "ball" (0 is background); keep confident detections only.
    keep = (pred["scores"] >= threshold) & (pred["labels"] == 1)

    if debug_dir is not None:
        save_debug_image(raw, pred, keep, image_path, debug_dir)

    return int(keep.sum().item())


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


def main():
    # Select the device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = load_model(WEIGHTS, device)

    with open(INPUT_JSON) as f:
        image_paths = json.load(f)["image_path"]

    results = []
    for image_path in image_paths:
        n_balls = count_balls(model, image_path, device, THRESHOLD, debug_dir=DEBUG_DIR)
        results.append({"image_path": image_path, "num_balls": n_balls})
        print(f"{image_path}: {n_balls:3d}")

    print(f"\nAnnotated images written to {DEBUG_DIR}/")

    if OUTPUT:
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nWrote results to {OUTPUT}")


if __name__ == "__main__":
    main()

import json
import os

import torch
import torchvision
from torchvision.transforms import v2 as transforms

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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


# Configuration
INPUT_JSON = "input.json"
OUTPUT = "ball_counts.json"
WEIGHTS = "ball_detector.pt"
THRESHOLD = 0.5

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


def count_balls(model, image_path, device, threshold):
    image = torchvision.tv_tensors.Image(torchvision.io.decode_image(image_path))
    image = inference_transform(image).to(device)

    with torch.no_grad():
        pred = model([image])[0]

    # Label 1 is "ball" (0 is background); keep confident detections only.
    keep = (pred["scores"] >= threshold) & (pred["labels"] == 1)
    return int(keep.sum().item())


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
        n_balls = count_balls(model, image_path, device, THRESHOLD)
        results.append({"image_path": image_path, "num_balls": n_balls})
        print(f"{image_path}: {n_balls:3d}")

    if OUTPUT:
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nWrote results to {OUTPUT}")


if __name__ == "__main__":
    main()

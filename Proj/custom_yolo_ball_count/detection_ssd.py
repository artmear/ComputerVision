import json
import os

import torch
import torchvision
from torchvision.transforms import v2 as transforms

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels


class BallDetector(torch.nn.Module):
    # Background + ball
    NUM_CLASSES = 2

    def __init__(self, use_local=True):
        super().__init__()
        if use_local:
            self.model = ssd300_vgg16(weights=None, weights_backbone=None)
        else:
            self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

        # SSD has no roi_heads; instead swap the multi-scale classification head
        # for our single "ball" class, keeping the same anchors and channels.
        in_channels = retrieve_out_channels(self.model.backbone, (300, 300))
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        self.model.head.classification_head = SSDClassificationHead(
            in_channels, num_anchors, self.NUM_CLASSES
        )

    def forward(self, images, targets=None):
        # In training mode returns the loss dict; in eval mode returns detections.
        return self.model(images, targets)


# Configuration
INPUT_JSON = "input.json"
OUTPUT = "ball_counts.json"
WEIGHTS = "ball_detector_ssd.pt"
THRESHOLD = 0.5

# SSD300 normalizes/resizes to 300x300 internally, so we only convert to a
# float image (same as the Faster R-CNN pipeline).
inference_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])


def load_model(weights_path, device):
    model = BallDetector(use_local=True).to(device)
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

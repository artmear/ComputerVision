import torch
from torchvision.transforms import v2 as transforms
import torchvision
import torchmetrics
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os, random, shutil
import json

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels

""" 
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
"""

class BallDetector(torch.nn.Module):
    # Background + ball
    NUM_CLASSES = 2

    def __init__(self):
        super().__init__()
        # SSD300 with a COCO-pretrained VGG16 backbone.
        weights = SSD300_VGG16_Weights.DEFAULT
        self.model = ssd300_vgg16(weights=weights)

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

device = None
metric = None

class PoolBallsDataset:
    def __init__(self, root, files, transform=None):
        self.root = root
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        annotation = self.files[i]
        image_path = os.path.join(self.root, annotation["image_path"])
        image = torchvision.tv_tensors.Image(torchvision.io.decode_image(image_path))

        bboxes = []
        for ball in annotation["balls"]:
            x1 = ball["x"]
            y1 = ball["y"]
            x2 = ball["x"] + ball["width"]
            y2 = ball["y"] + ball["height"]
            bboxes.append((x1, y1, x2, y2))
        bboxes = torchvision.tv_tensors.BoundingBoxes(
            bboxes, format='XYXY', canvas_size=image.shape[1:]
        )

        if self.transform:
            image, bboxes = self.transform(image, bboxes)

        # Faster R-CNN target: all boxes are class 1 ("ball"); 0 is background.
        target = {
            "boxes": bboxes,
            "labels": torch.ones(len(bboxes), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    return batch


# Faster R-CNN normalizes/resizes internally, so we only need float images + box-aware augmentation.
train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToDtype(torch.float32, scale=True),
])

val_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])


def one_epoch(model, optimizer, dataloader, is_training):
    avg_loss = avg_metric = 0
    for batch in dataloader:
        images = [b[0].to(device) for b in batch]
        targets = [{k: v.to(device) for k, v in b[1].items()} for b in batch]

        if is_training:
            model.train()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(dataloader)
        else:
            model.eval()
            with torch.no_grad():
                preds = model(images)
            avg_metric += metric(preds, targets)['map_50'].item() / len(dataloader)

    return avg_loss, avg_metric


def main():
    global device, metric

    dataset_annotations = "datasets/merged/_annotations.json"
    with open(dataset_annotations) as json_data:
        annotations_json_array = json.load(json_data)

    # Split annotations into train/validation
    random.seed(42)
    split_ratio = 0.8
    random.shuffle(annotations_json_array)
    n_train = int(len(annotations_json_array) * split_ratio)
    train_annotations = annotations_json_array[:n_train]
    val_annotations = annotations_json_array[n_train:]

    # Save validation set (images + annotations) to a separate folder for testing
    val_dir = 'datasets/validation'
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    for annotation in val_annotations:
        shutil.copy(annotation['image_path'], os.path.join(val_dir, 'images', os.path.basename(annotation['image_path'])))
    with open(os.path.join(val_dir, '_annotations.json'), 'w') as f:
        json.dump(val_annotations, f)

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print('using device:', device)

    epochs = 10

    model = BallDetector().to(device)
    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=1e-4,
    )
    metric = torchmetrics.detection.MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    batch_size = 4
    num_workers = 4

    train_dataset = PoolBallsDataset('.', train_annotations, train_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers,
    )

    val_dataset = PoolBallsDataset('.', val_annotations, val_transform)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size, collate_fn=collate_fn, num_workers=num_workers,
    )

    train_history = {'loss': []}
    val_history = {'metric': []}
    for epoch in range(epochs):
        avg_loss, _ = one_epoch(model, optimizer, train_dataloader, True)
        train_history['loss'].append(avg_loss)
        print(f'Epoch {epoch+1:2d}/{epochs} - Train loss: {avg_loss}')

        _, avg_metric = one_epoch(model, optimizer, val_dataloader, False)
        val_history['metric'].append(avg_metric)
        print(f'Epoch {epoch+1:2d}/{epochs} - Val   mAP50: {avg_metric}')

    torch.save(model.state_dict(), 'ball_detector.pt')

    # Visualize predictions on validation set
    model.eval()
    batch = next(iter(val_dataloader))
    images = [b[0].to(device) for b in batch]
    targets = [b[1] for b in batch]
    with torch.no_grad():
        preds = model(images)
    for i in range(min(4, len(images))):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i].permute(1, 2, 0).cpu())
        keep = preds[i]['scores'] >= 0.5
        boxes = preds[i]['boxes'][keep].cpu()
        for x1, y1, x2, y2 in boxes:
            plt.gca().add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linestyle='--'))
        for x1, y1, x2, y2 in targets[i]['boxes'].cpu():
            plt.gca().add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='b'))
        plt.title(f'Total balls: {len(boxes)}')
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/validation_predictions.png')


if __name__ == '__main__':
    main()

import torch
from torchvision.transforms import v2 as transforms
from torch.nn import functional as F
import torchvision
import torchmetrics
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os, random, re

class MyYOLO(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        # ResNet-18 backbone up to the last convolutional feature map (N, 512, H/32, W/32).
        resnet = torchvision.models.resnet18(weights='DEFAULT')
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])

        # We're currently freezing the encoder (which is one of the parts that we took of resnet)
        # TODO: Check if unfreezing and retraining will lead to higher improvement in ball detection.
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Single YOLO-like head: 1 objectness + 4 bbox values per cell => 5 channels.
        self.pred_head = torch.nn.Sequential(
            torch.nn.LazyConv2d(256, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.LazyConv2d(5, 1),
        )

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.yscale = output_shape[0] / input_shape[0]
        self.xscale = output_shape[1] / input_shape[1]

    def forward(self, x):

        # Extract the features using the resnet backbone (Transfer learning part)
        enc = self.encoder(x)

        # Force the output shape if it is not yet the same (e.g., 7x7 for 224x224 input).
        if enc.shape[-2:] != self.output_shape:
            enc = F.adaptive_avg_pool2d(enc, self.output_shape)

        # Predict the values of the bouding boxes
        pred = self.pred_head(enc)

        
        scores = torch.sigmoid(pred[:, 0:1]) # Scores for each bounding box
        rel_bboxes = torch.sigmoid(pred[:, 1:5]) # Bounding boxes

        # Convert predicted [cx, cy, w, h] in cell-relative space into absolute XYXY boxes.
        height, width = x.shape[2:]
        xx = torch.arange(0, width, 1 / self.xscale, device=x.device, dtype=rel_bboxes.dtype)
        yy = torch.arange(0, height, 1 / self.yscale, device=x.device, dtype=rel_bboxes.dtype)
        xx, yy = torch.meshgrid(xx, yy, indexing='xy')
        x1 = rel_bboxes[:, 0] / self.xscale + xx[None] - rel_bboxes[:, 2] * width / 2
        y1 = rel_bboxes[:, 1] / self.yscale + yy[None] - rel_bboxes[:, 3] * height / 2
        x2 = rel_bboxes[:, 0] / self.xscale + xx[None] + rel_bboxes[:, 2] * width / 2
        y2 = rel_bboxes[:, 1] / self.yscale + yy[None] + rel_bboxes[:, 3] * height / 2
        abs_bboxes = torch.stack((
            x1.clamp(0, width - 1),
            y1.clamp(0, height - 1),
            x2.clamp(0, width - 1),
            y2.clamp(0, height - 1),
        ), 1)
        return scores, abs_bboxes

    def postprocess(self, scores, abs_bboxes, conf_threshold=0.25, iou_threshold=0.5, topk=100):
        # Convert grid predictions to per-image detections with confidence threshold + NMS.
        batch_size, _, grid_h, grid_w = scores.shape
        scores = scores.reshape(batch_size, grid_h * grid_w)
        abs_bboxes = abs_bboxes.reshape(batch_size, 4, grid_h * grid_w).permute(0, 2, 1)

        results = []
        for i in range(batch_size):
            score_i = scores[i]
            boxes_i = abs_bboxes[i]

            keep = score_i >= conf_threshold
            if keep.sum() == 0:
                results.append({"boxes": boxes_i[:0], "scores": score_i[:0]})
                continue

            score_i = score_i[keep]
            boxes_i = boxes_i[keep]

            if topk is not None and score_i.numel() > topk:
                topk_scores, topk_idx = torch.topk(score_i, k=topk)
                score_i = topk_scores
                boxes_i = boxes_i[topk_idx]

            nms_idx = torchvision.ops.nms(boxes_i, score_i, iou_threshold)
            results.append({"boxes": boxes_i[nms_idx], "scores": score_i[nms_idx]})

        return results

    def detect(self, x, conf_threshold=0.25, iou_threshold=0.5, topk=100):
        scores, abs_bboxes = self.forward(x)
        return self.postprocess(scores, abs_bboxes, conf_threshold, iou_threshold, topk)


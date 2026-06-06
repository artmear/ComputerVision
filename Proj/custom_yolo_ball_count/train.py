import torch
from torchvision.transforms import v2 as transforms
from torch.nn import functional as F
import torchvision
import torchmetrics
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os, random, re
import json

from custom_model import MyYOLO


device = None
metric = None


class PoolBallsDataset:
    def __init__(self, root, fold, transform=None, files=None):
        self.root = root
        if files is not None:
            self.files = files
        else:
            dataset_annotations = "datasets/merged/_annotations.json"
            with open(dataset_annotations) as json_data:
                self.files = json.load(json_data)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        annotation = self.files[i]
        image_path = os.path.join(self.root, annotation["image_path"])
        image = torchvision.tv_tensors.Image(torchvision.io.decode_image(image_path))

        x1 = annotation["x"] # xmin
        y1 = annotation["y"] # ymin
        x2 = annotation["x"] + annotation["width"] # xmax
        y2 = annotation["y"] + annotation["height"] # ymax
        bboxes = [(x1, y1, x2, y2)]
        bboxes = torchvision.tv_tensors.BoundingBoxes(bboxes, format='XYXY', canvas_size=image.shape[1:])
        if self.transform:
            image, bboxes = self.transform(image, bboxes)
        return image, bboxes


def collate_fn(batch):
    return batch


train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomAffine(30, (0.2, 0.2), (0.8, 1.2))], 0.5),
    transforms.ToDtype(torch.float32, True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
#
#train_dataset = PoolBallsDataset('.', 'train', train_transform, files=train_annotations)


# Define transformations to be applied to validation data
#val_transform = transforms.Compose([
#    transforms.ToImage(),
#    transforms.Resize((256, 256)),
#    transforms.CenterCrop((224, 224)),
#    transforms.ToDtype(torch.float32, True),
#    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#])
#
## Define dataloader for validation data
#val_dataset = OxfordPetDataset('.', 'val', val_transform)
#
#
## hyperparameters
#batch_size = 8
#num_workers = 4
#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
#
## create validation dataset + dataloader
#val_dataset = PoolBallsDataset('.', 'val', train_transform, files=val_annotations)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, collate_fn=collate_fn, num_workers=num_workers)



def ground_truth_to_masks(batch_bboxes, input_shape, output_shape, device):
    scores_masks = torch.zeros(len(batch_bboxes), 1, *output_shape, dtype=bool, device=device)
    bboxes_masks = torch.zeros(len(batch_bboxes), 4, *output_shape, dtype=float, device=device)
    yscale = output_shape[0]/input_shape[0]
    xscale = output_shape[1]/input_shape[1]
    for i, bboxes in enumerate(batch_bboxes):
        for x1, y1, x2, y2 in bboxes:
            i1 = int(torch.floor(x1*xscale))
            j1 = int(torch.floor(y1*yscale))
            i2 = int(torch.ceil(x2*xscale))
            j2 = int(torch.ceil(y2*yscale))
            scores_masks[i, :, j1:j2, i1:i2] = 1
            bboxes_masks[i, 0, j1:j2, i1:i2] = x1
            bboxes_masks[i, 1, j1:j2, i1:i2] = y1
            bboxes_masks[i, 2, j1:j2, i1:i2] = x2
            bboxes_masks[i, 3, j1:j2, i1:i2] = y2
    return scores_masks, bboxes_masks.to(torch.float32)


def bbox_iou_loss(pred_boxes, target_boxes, object_mask):
    pred_boxes = pred_boxes.permute(0, 2, 3, 1).reshape(-1, 4)
    target_boxes = target_boxes.permute(0, 2, 3, 1).reshape(-1, 4)
    positive = object_mask.reshape(-1)

    if not positive.any():
        return pred_boxes.sum() * 0

    ious = torchvision.ops.box_iou(pred_boxes[positive], target_boxes[positive]).diagonal()
    return 1 - ious.mean()


def one_epoch(model, optimizer, dataloader, is_training):
  model.train() if is_training else model.eval()
  avg_loss = avg_metric = 0
  for batch in dataloader:
    images = torch.stack([b[0] for b in batch]).to(device)
    bboxes = [b[1] for b in batch] # bboxes = [b[2] for b in batch]
    # (1) call ground_truth_to_masks() to convert the bboxes to a mask
    # (2) do the forward pass to obtain the predicted scores and bboxes
    # (3) do the loss and backward pass: use BCE for scores + IoU for bboxes
    scores_masks, bboxes_masks = ground_truth_to_masks(
        bboxes, images.shape[-2:], model.output_shape, device
    )
    scores_preds, bboxes_preds = model(images)
    score_loss = F.binary_cross_entropy(scores_preds, scores_masks.float())
    bbox_loss = bbox_iou_loss(bboxes_preds, bboxes_masks, scores_masks)
    loss = score_loss + bbox_loss
    if is_training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss += loss.item() / len(dataloader)
    # convert ground-truth and predictions to format that torchmetrics likes
    preds = [{'boxes': bboxes.flatten(1).T, 'scores': scores[0].flatten(), 'labels': torch.zeros(7*7, dtype=int)} for scores, bboxes in zip(scores_preds, bboxes_preds)]
    true = [{'boxes': boxes, 'labels': torch.zeros(len(boxes), dtype=int)} for boxes in bboxes]
    avg_metric += metric(preds, true)['map_50'].item() / len(dataloader)
  return avg_loss, avg_metric


def main():
  global device, metric

  dataset_annotations = "datasets/merged/_annotations.json"
  with open(dataset_annotations) as json_data:
      annotations_json_array = json.load(json_data)

  # split annotations into train/validation
  random.seed(42)
  split_ratio = 0.8
  random.shuffle(annotations_json_array)
  n_train = int(len(annotations_json_array) * split_ratio)
  train_annotations = annotations_json_array[:n_train]
  val_annotations = annotations_json_array[n_train:]

  # Device
  if torch.cuda.is_available():
      device = 'cuda'
  elif torch.backends.mps.is_available():
      device = 'mps'
  else:
      device = 'cpu'
  print('using device:', device)

  # Number of epochs
  epochs = 10

  input_shape = (224, 224)
  output_shape = (7, 7)
  model = MyYOLO(input_shape, output_shape).to(device)
  optimizer = torch.optim.Adam(
      (param for param in model.parameters() if param.requires_grad),
      lr=1e-4,
  )
  metric = torchmetrics.detection.MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

  train_dataset = PoolBallsDataset('.', 'train', train_transform, files=train_annotations)
  batch_size = 8
  num_workers = 4
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size,
      shuffle=True,
      collate_fn=collate_fn,
      num_workers=num_workers,
  )

  # create validation dataset + dataloader
  val_dataset = PoolBallsDataset('.', 'val', train_transform, files=val_annotations)
  val_dataloader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size,
      collate_fn=collate_fn,
      num_workers=num_workers,
  )

  train_history = {'loss': [], 'metric': []}
  val_history = {'loss': [], 'metric': []}
  for epoch in range(epochs):

    # compute train
    avg_loss, avg_metric = one_epoch(model, optimizer, train_dataloader, True)
    train_history['loss'].append(avg_loss)
    train_history['metric'].append(avg_metric)
    print(f'Epoch {epoch+1:2d}/{epochs} - Train loss: {avg_loss} - Train metric: {avg_metric}')

    # compute validation statistics
    avg_loss, avg_metric = one_epoch(model, optimizer, val_dataloader, False)
    val_history['loss'].append(avg_loss)
    val_history['metric'].append(avg_metric)
    print(f'Epoch {epoch+1:2d}/{epochs} - Val   loss: {avg_loss} - Val   metric: {avg_metric}')

  # Visualize predictions on validation set
  model.eval()

  batch = next(iter(val_dataloader))
  images = torch.stack([b[0] for b in batch]).to(device)
  bboxes = [b[1] for b in batch]
  with torch.no_grad():
      scores_preds, bboxes_preds = model(images)
  for i in range(4):
      plt.subplot(2, 2, i+1)
      image = images[i].permute(1, 2, 0).cpu()*torch.tensor([[[0.229, 0.224, 0.225]]]) + torch.tensor([[[0.485, 0.456, 0.406]]])
      plt.imshow(image)
      image_scores = scores_preds[i, 0].flatten()
      image_bboxes = bboxes_preds[i].flatten(1).T
      image_bboxes = image_bboxes[image_scores >= 0.5]
      image_scores = image_scores[image_scores >= 0.5]
      from torchvision.ops import nms
      if image_bboxes.numel() > 0:
          ix = nms(image_bboxes, image_scores, 0.10)
          image_bboxes = image_bboxes[ix]
      for x1, y1, x2, y2 in image_bboxes.cpu():
          plt.gca().add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linestyle='--'))
      for x1, y1, x2, y2 in bboxes[i].cpu():
          plt.gca().add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='b'))
  plt.tight_layout()
  os.makedirs('output', exist_ok=True)
  plt.savefig('output/validation_predictions.png')


if __name__ == '__main__':
  main()
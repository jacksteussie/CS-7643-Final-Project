import os
import torch
from const import PROJECT_ROOT, DOTA_DIR, DOTA_MOD_DIR

from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class DotaDataset(torch.utils.data.Dataset):
    def __init__(self, folder="train", transforms = None):
        self.root = PROJECT_ROOT
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(DOTA_MOD_DIR if DOTA_MOD_DIR else DOTA_DIR, "images", folder))))
        self.labels = list(sorted(os.listdir(os.path.join(DOTA_MOD_DIR if DOTA_MOD_DIR else DOTA_DIR, "labels", folder))))
        
    def __len__(self):
        return len(self.imgs)
    
    def read_labels(file_path):
        labels = []
        coordinates_list = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                class_number = int(parts[0])
                coordinates = list(map(float, parts[1:]))
                labels.append(class_number)
                coordinates_list.append(coordinates_list)
        return labels, coordinates
    
    def __getitem__(self, idx):
        # Bounding box definition: https://docs.ultralytics.com/datasets/obb/#yolo-obb-format 
        # The link says that coords are all normalized to 1

        assert idx < len(self.imgs)
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        img = read_image(img_path)
        labels, box_coords = self.read_labels(label_path)

        img = tv_tensors.Image(img)
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(box_coords, format="XYXYXYXY", canvas_size=F.get_size(img)) # TODO: are normalized coords ok here?
        target["labels"] = labels
        target["image_id"] = idx
        
        # Calculate area according to box_coords (normalized)
        # x1 y1 x2 y2 x3 y3 x4 y4
        # 0  1  2  3  4  5  6  7
        l1 = abs(box_coords[:, 0] - box_coords[:, 6]) # x1 - x4
        l2 = abs(box_coords[:, 7] - box_coords[:, 1]) # y4 - y1
        h = torch.sqrt(l1 ** 2 + l2 ** 2)
        l3 = abs(box_coords[:, 4] - box_coords[:, 6]) # x3 - x4
        l4 = abs(box_coords[:, 5] - box_coords[:, 7]) # y3 - y4
        w = torch.sqrt(l3 ** 2 + l4 ** 2)
        target["area"] = h * w

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

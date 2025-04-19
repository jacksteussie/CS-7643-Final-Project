import os
import torch
from const import PROJECT_ROOT, DOTA_DIR, DOTA_MOD_DIR
import logging

from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from utils import norm_quad_to_rotated_rect

logger = logging.getLogger(__name__)

class DotaDataset(torch.utils.data.Dataset):
    def __init__(self, folder="train", transforms=None):
        self.root = PROJECT_ROOT
        self.transforms = transforms
        base_dir = DOTA_MOD_DIR if DOTA_MOD_DIR else DOTA_DIR
        image_dir = os.path.join(base_dir, "images", folder)
        label_dir = os.path.join(base_dir, "labels", folder)

        logger.debug(f"BASE_DIR: {base_dir}")
        logger.debug(f"IMAGE_DIR: {image_dir}")
        logger.debug(f"LABEL_DIR: {label_dir}")

        all_imgs = sorted(os.listdir(image_dir))
        all_labels = sorted(os.listdir(label_dir))
        self.imgs = []
        self.labels = []

        for img, label in zip(all_imgs, all_labels):
            if self.is_valid_label(os.path.join(label_dir, label)):
                self.imgs.append(os.path.join(image_dir, img))
                self.labels.append(os.path.join(label_dir, label))
            else:
                logger.info(f"⚠️ Skipping invalid label: {os.path.join(label_dir, label)}")

        logger.info(f"Found {len(self.labels)} valid labels out of {len(all_labels)}.")

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def read_labels(file_path):
        labels = []
        coordinates = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                class_number = int(parts[0])
                coords = list(map(float, parts[1:]))
                labels.append(class_number)
                coordinates.append(coords)
        return labels, coordinates

    @staticmethod
    def is_valid_label(label_path):
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    coords = list(map(float, parts[1:]))
                    if any(c < 0.0 or c > 1.0 for c in coords):
                        return False
            return True
        except Exception as e:
            logger.info(f"⚠️ Error reading {label_path}: {e}")
            return False

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        img = read_image(img_path)  # shape: [3, H, W], dtype: uint8
        img = tv_tensors.Image(img)

        labels, box_coords = self.read_labels(label_path)
        box_coords = torch.tensor(box_coords, dtype=torch.float32)

        h, w = F.get_size(img)
        rotated_boxes = norm_quad_to_rotated_rect(box_coords, w, h)  # [N, 5]

        target = {
            "boxes": rotated_boxes,                            # [cx, cy, w, h, angle]
            "labels": torch.tensor(labels, dtype=torch.int64)  # class indices
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target




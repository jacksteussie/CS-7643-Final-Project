import os
import torch
from const import PROJECT_ROOT, DOTA_DIR, DOTA_MOD_DIR
import logging

from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from utils import norm_quad_to_aabb

logger = logging.getLogger(__name__)

class DotaDataset(torch.utils.data.Dataset):
    def __init__(self, folder="train", transforms=None):
        self.root = PROJECT_ROOT
        self.transforms = transforms
        base_dir = DOTA_MOD_DIR if DOTA_MOD_DIR else DOTA_DIR
        image_dir = os.path.join(base_dir, "images", folder)
        label_dir = os.path.join(base_dir, "labels", folder)

        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.transform = self.weights.transforms()

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
        img = img.float() / 255.0

        labels, box_coords = self.read_labels(label_path)
        box_coords = torch.tensor(box_coords, dtype=torch.float32)

        if box_coords.ndim == 1:
            box_coords = box_coords.unsqueeze(0)

        h, w = F.get_size(img)
        aabb_boxes = norm_quad_to_aabb(box_coords, w, h)  # [N, 4]

        if aabb_boxes.ndim != 2 or aabb_boxes.shape[1] != 4:
            logger.warning(f"⚠️ Skipping item at index {idx} due to invalid box shape: {aabb_boxes.shape}")
            return self.__getitem__((idx + 1) % len(self))

        target = {
            "boxes": aabb_boxes,
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
    # def __getitem__(self, idx):
    #     img_path   = self.imgs[idx]
    #     label_path = self.labels[idx]

    #     # 1) load image as [C,H,W] float tensor in [0,1]
    #     img = read_image(img_path).float().div(255.0)

    #     # 2) read your normalized quad coords → absolute AABB
    #     labels, quads = self.read_labels(label_path)
    #     quads = torch.tensor(quads, dtype=torch.float32)
    #     if quads.ndim == 1:
    #         quads = quads.unsqueeze(0)

    #     # get width & height from tensor
    #     _, h, w = img.shape
    #     boxes = norm_quad_to_aabb(quads, w, h)  # [N,4]

    #     # 3) shift labels so 0 → background, real classes start at 1
    #     labels = torch.tensor(labels, dtype=torch.int64).add(1)

    #     target = {
    #         "boxes":  boxes,
    #         "labels": labels,
    #     }

    #     # 4) apply any detection-style transforms (e.g. GeneralizedRCNNTransform)
    #     if self.transforms:
    #         img, target = self.transforms(img, target)

    #     return img, target



class DotaDataModule(LightningDataModule):
    def __init__(self, batch_size=4, num_workers=4, transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

    def setup(self, stage=None):
        self.train_dataset = DotaDataset(folder="train", transforms=self.transforms)
        self.val_dataset = DotaDataset(folder="val", transforms=self.transforms)

        test_label_dir = os.path.join(DOTA_MOD_DIR if DOTA_MOD_DIR else DOTA_DIR, "labels", "test")
        if os.path.exists(test_label_dir) and len(os.listdir(test_label_dir)) > 0:
            self.test_dataset = DotaDataset(folder="test", transforms=self.transforms)
        else:
            self.test_dataset = None
            logger.warning("⚠️ Test dataset skipped — no label files found.")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)

    # @staticmethod
    # def collate_fn(batch):
    #     return tuple(zip(*batch))  # Needed for detection models
    @staticmethod
    def collate_fn(batch):
        imgs, tgts = zip(*batch)
        return list(imgs), list(tgts)
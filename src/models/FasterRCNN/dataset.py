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
from pathlib import Path

from utils import norm_quad_to_aabb

logger = logging.getLogger(__name__)


class DotaDataset(torch.utils.data.Dataset):
    def __init__(self, folder="train", transforms=None):
        self.transforms = transforms

        base_dir = Path(DOTA_MOD_DIR or DOTA_DIR)
        image_dir = base_dir / "images" / folder
        label_dir = base_dir / "labels" / folder

        labels_map = {
            p.stem: p
            for p in sorted(label_dir.glob("*.txt"))
            if self.is_valid_label(p)
        }

        self.imgs = []
        self.labels = []

        for img_path in sorted(image_dir.glob("*")):
            stem = img_path.stem
            label_path = labels_map.get(stem)
            if label_path is None:
                logger.debug(f"⚠️ No label for image {img_path.name}, skipping.")
                continue

            self.imgs.append(img_path)
            self.labels.append(label_path)

        logger.info(f"Found {len(self.labels)} valid labeled images in '{folder}'.")

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


        img = read_image(img_path)  # [3, H, W], uint8 tensor
        img = tv_tensors.Image(img).float() / 255.0

        raw_labels, box_coords = self.read_labels(label_path)
        box_coords = torch.tensor(box_coords, dtype=torch.float32)

        if len(raw_labels) == 0:
             logger.warning(f"⚠️ Skipping index {idx} ({label_path.name}), no objects found after read_labels.")
             if idx == len(self) - 1: 
                 next_idx = 0
             else:
                 next_idx = idx + 1
             if next_idx == idx:
                 raise RuntimeError(f"Dataset contains only invalid samples, or infinite loop detected starting at index {idx}.")
             return self.__getitem__(next_idx)

        # Ensure box_coords tensor is [N, 8] where N is number of objects
        if box_coords.ndim == 1: 
             box_coords = box_coords.unsqueeze(0) 

        # Get image dimensions
        h, w = F.get_size(img)

        # Convert normalized quadrilateral coordinates [N, 8] to pixel AABB [N, 4]
        # This is your 'norm_quad_to_aabb' function call
        aabb_boxes = norm_quad_to_aabb(box_coords, w, h)


        # --- Validate output of norm_quad_to_aabb ---
        if aabb_boxes.ndim != 2 or aabb_boxes.shape[1] != 4 or aabb_boxes.shape[0] != len(raw_labels):
             logger.warning(f"⚠️ Skipping index {idx} ({label_path.name}), invalid box shape after norm_quad_to_aabb: {aabb_boxes.shape}, expected {len(raw_labels)}x4")
             if idx == len(self) - 1:
                 next_idx = 0
             else:
                 next_idx = idx + 1
             if next_idx == idx: raise RuntimeError(f"Dataset contains only invalid samples after norm_quad_to_aabb, or infinite loop detected starting at index {idx}.")
             return self.__getitem__(next_idx)
        
        target_labels = [label + 1 for label in raw_labels] # <--- IMPORTANT REINDEXING

        target = {
            "boxes": tv_tensors.BoundingBoxes(aabb_boxes, format="XYXY", canvas_size=(w, h)),
            "labels": torch.tensor(target_labels, dtype=torch.int64)
        }


        return img, target


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

    @staticmethod
    def collate_fn(batch):
        imgs, tgts = zip(*batch)
        return list(imgs), list(tgts)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches # Import patches for Rectangle

    CLASS_NAMES = [
        "UNUSED_LABEL_0",     # Index 0: Placeholder (not used by labels 1-16)
        "plane",              # Index 1: Corresponds to raw label 0
        "ship",               # Index 2: Corresponds to raw label 1
        "storage tank",       # Index 3: Corresponds to raw label 2
        "baseball diamond",   # Index 4: Corresponds to raw label 3
        "tennis court",       # Index 5: Corresponds to raw label 4
        "basketball court",   # Index 6: Corresponds to raw label 5
        "ground track field", # Index 7: Corresponds to raw label 6
        "harbor",             # Index 8: Corresponds to raw label 7
        "bridge",             # Index 9: Corresponds to raw label 8
        "large vehicle",      # Index 10: Corresponds to raw label 9
        "small vehicle",      # Index 11: Corresponds to raw label 10
        "helicopter",         # Index 12: Corresponds to raw label 11
        "roundabout",         # Index 13: Corresponds to raw label 12
        "soccer ball field",  # Index 14: Corresponds to raw label 13
        "swimming pool",      # Index 15: Corresponds to raw label 14
        "container crane"     # Index 16: Corresponds to raw label 15
    ]

    # Quick sanity check with visualization
    dm = DotaDataModule(batch_size=2, num_workers=0, transforms=None)
    dm.setup()
    loader = dm.train_dataloader()

    if len(loader) == 0:
        print("DataLoader is empty. Check dataset paths and filtering.")
    else:
        try:
            imgs, tgts = next(iter(loader))
        except StopIteration:
            print("DataLoader yielded no batches. Check dataset.")
            exit() # Exit if no data

        print(f"Loaded batch with {len(imgs)} images.")

        for i, (img, tgt) in enumerate(zip(imgs, tgts)):
            print(f"\nProcessing Sample {i}:")
            if not isinstance(img, torch.Tensor) or img.ndim != 3:
                 print(f"  Skipping sample {i}: Invalid image format. Expected Tensor, got {type(img)}")
                 continue

            if not isinstance(tgt, dict) or 'boxes' not in tgt or 'labels' not in tgt:
                 print(f"  Skipping sample {i}: Invalid target format. Expected dict with 'boxes' and 'labels', got {type(tgt)}")
                 continue

            print(f"  Image shape: {img.shape}")
            print(f"  Target keys: {tgt.keys()}")
            print(f"  Number of boxes: {len(tgt['boxes'])}")
            print(f"  Labels: {tgt['labels'].tolist()}")

            np_img = img.cpu().permute(1, 2, 0).numpy()

            fig, ax = plt.subplots(1, figsize=(10, 10)) 
            ax.imshow(np_img)

            boxes = tgt['boxes']
            labels = tgt['labels']

            if len(boxes) != len(labels):
                print(f"  Warning: Mismatch between number of boxes ({len(boxes)}) and labels ({len(labels)}) for sample {i}.")
                continue

            for box, label_idx in zip(boxes, labels):
                if not isinstance(box, torch.Tensor) or len(box) != 4:
                     print(f"  Skipping invalid box: {box}")
                     continue

                x1, y1, x2, y2 = box.cpu().tolist()
                label_idx = label_idx.item()

                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          linewidth=1, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)

                if 0 < label_idx < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[label_idx]
                else:
                    class_name = f"IDX:{label_idx}"

                ax.text(x1, y1 - 5, class_name,
                        color='white', fontsize=8,
                        bbox=dict(facecolor='lime', alpha=0.5, pad=0.1, edgecolor='none')) 

            ax.set_title(f"Sample {i}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            if i >= 2: 
                 break
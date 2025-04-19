import logging
import torch
import torch.utils
import torch.utils.data
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2, FasterRCNN
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.resnet import ResNet50_Weights
import hydra
from hydra.utils import instantiate 
from omegaconf import DictConfig, OmegaConf

from .datasets import DotaDataset
from .model import FastRCNNDFLPredictor, RotatedRoIHeads

def train_step(model: FasterRCNN, ):
    # TODO: train one epoch over training set
    ...

def val_step():
    # TODO: validate with the val dataset
    ...

def test_step():
    # TODO: test on holdout (a.k.a. test) set (should this maybe be a separate script? we could just make a single "experiment" script here)
    ...

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("🔧 Training Config:")
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = DotaDataset()
    val_dataset = DotaDataset(folder="val")

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )

    match cfg.model.backbone:
        case "mobilenet":
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
            )
        case "resnet":
            model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
            )
        case _:
            raise ValueError(f"Unknown backbone '{cfg.model.backbone}'")

    # Replace the head
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNDFLPredictor(in_feats, cfg.model.num_classes, reg_max=cfg.model.reg_max)

    # Replace RoIHeads with rotated-aware variant
    orig = model.roi_heads
    model.roi_heads = RotatedRoIHeads(
        box_roi_pool=model.roi_heads.box_roi_pool,
        box_head=model.roi_heads.box_head,
        box_predictor=model.roi_heads.box_predictor,

        fg_iou_thresh=cfg.model.fg_iou_thresh,
        bg_iou_thresh=cfg.model.bg_iou_thresh,
        batch_size_per_image=cfg.model.batch_size_per_image,
        positive_fraction=cfg.model.positive_fraction,
        bbox_reg_weights=None,  # you can later add this to the config if you want to tune it

        score_thresh=cfg.model.score_thresh,
        nms_thresh=cfg.model.nms_thresh,
        detections_per_img=cfg.model.detections_per_image,  # You could also make this configurable
        reg_max=cfg.model.reg_max
    )

    model.to(device)

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Set up parameter groups
    param_groups = []
    for group_cfg in cfg.optimizer.param_groups:
        name = group_cfg.name
        if name == "backbone":
            params = [p for _, p in model.backbone.named_parameters() if p.requires_grad]
        elif name == "head":
            params = [p for _, p in model.roi_heads.named_parameters() if p.requires_grad]
        else:
            raise ValueError(f"Unknown param group name: {name}")

        group_cfg_clean = OmegaConf.to_container(group_cfg, resolve=True)
        group_cfg_clean.pop("name")
        group_cfg_clean["params"] = params
        param_groups.append(group_cfg_clean)

    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    optimizer_cfg.pop("param_groups", None)

    optimizer = instantiate(optimizer_cfg, param_groups, _convert_="partial")
    lr_scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    for epoch in range(cfg.training.epochs):
        train_step(model, optimizer, data_loader_train, device, epoch)
        lr_scheduler.step()
        val_step(model, data_loader_val, device=device)


if __name__ == "__main__":
    train()
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
from .model import FastRCNNDFLPredictor

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

    # instantiate configured values
    num_classes = cfg.model.num_classes
    model_backbone = cfg.model.backbone # resnet or mobilenet
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_dataset = DotaDataset()
    val_dataset = DotaDataset(folder="val")

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=torch.utils.collate_fn # TODO: is this needed?
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        # collate_fn=torch.utils.collate_fn # TODO: is this needed?
    )

    match model_backbone:
        case "mobilenet":
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None, # debating on whether to use COCO or not...
                weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
            )
        case "resnet":
            model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weigts_backbone=ResNet50_Weights.IMAGENET1K_V2,
            )
        case _:
            raise ValueError(f"backbone name in the configuration file was set as {model_backbone}")
    
    # replace the head of the model
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNDFLPredictor(in_feats, num_classes, reg_max=16)

    for p in model.backbone.parameters():
        p.requires_grad = False

    # we have the option to use different optimization parameters for the backbone and the head
    param_groups = []

    for group_cfg in cfg.optimizer.param_groups:
        name = group_cfg.name
        params = []

        if name == "backbone":
            params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        elif name == "head":
            params = [p for n, p in model.roi_heads.named_parameters() if p.requires_grad]
        else:
            raise ValueError(f"Unknown param group name: {name}")

        group_cfg_clean = OmegaConf.to_container(group_cfg, resolve=True)
        group_cfg_clean.pop("name")
        group_cfg_clean["params"] = params
        param_groups.append(group_cfg_clean)

    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    optimizer_cfg.pop("param_groups", None)

    optimizer = instantiate(
        optimizer_cfg,
        param_groups,
        _convert_="partial"
    )
    
    lr_scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    for epoch in range(epochs):
        # train for one epoch
        train_step(model, optimizer, data_loader_train, device, epoch)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the val dataset
        val_step(model, data_loader_val, device=device)

if __name__ == "__main__":
    train()